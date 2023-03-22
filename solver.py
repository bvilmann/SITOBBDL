# Control
import control as ctrl

# numerical packages
import numpy as np
from numpy.linalg import inv, det, eigvals, eig, norm 
from numpy import sqrt, log, real, imag
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy import stats

# plot packages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib import cm
prop_cycle = plt.rcParams['axes.prop_cycle']
clrs = prop_cycle.by_key()['color']

#
# from scipy.stats import norm
import SITOBBDS_utils as utils
from numerical_methods import NumericalMethods
import datetime
import copy

import pandas as pd

# https://pypi.org/project/sysidentpy/
# class ParamWrapper:
#     def __init__(self,val:float,base:float,unit:str=None):
#         self.v = val
#         self.base = base
#         self.unit = unit
#         return
#
#     def __get__(self):
#         return self.val

class ParamWrapper:
    def __init__(self,params,pu:bool=True):
        # ------ Load default parameters ------
        self.Vbase = 66e3
        self.Sbase = 100e6
        if pu: self.V = self.Vbase/self.Vbase
        self.R = 1
        self.Rin = 0.05 # BRK = 0.05
        self.Rload = 100 # BRK = 0.05
        self.L = 1e-3
        self.C1 = 0.5e-6
        self.C2 = 0.5e-6
        self.V_amp = self.Vbase * np.sqrt(2)
        self.f = 50
        self.omega = 2*np.pi*self.f
        self.phi = 0

        # ------ Get custom parameters ------
        if params is not None:
            for k, v in params.items():
                setattr(self,k,v)

        # ------ Per unitizing ------
        if pu:
            self.Ibase = self.Sbase / (np.sqrt(3) * self.Vbase)
            self.Zbase = self.Vbase**2 / self.Sbase           
    
            # Adjusting 
            for k in ['Rin','Rload','R','L','C1','C2']:
                setattr(self, k, getattr(self, k)/self.Zbase)


        return

class SITOBBDS:
    def __init__(self):
        self.method = NumericalMethods()
        self.models = ['c1_s0_o3',
                       'c1_s0_o3_load',
                       'c1_s0_o2']
        return
    
    def load_model(self,model=None,p=None):
        if model is None: model = self.model
        if p is None: p = self.p
        
        if isinstance(model, (list,tuple)):
            # ------ CUSTOM MODEL ------ 
            model='Custom'
            self.n = n = 1
            raise NotImplementedError('')
            
        elif isinstance(model,str) and model == 'c1_s0_o3':
            # ------ Single core, no screen, 3rd order ------ 
            self.n = n = 3
            A = np.array([[-p.R/p.L, 1/p.L, -1/p.L],
                            [-1/(p.C1), -1/(p.Rin * p.C1), 0],
                            [1/(p.C2), 0, 0],
                          ])
            
            B = np.array([[0], [1 / (p.Rin * p.C1)], [0]])

        elif isinstance(model,str) and model == 'c1_s0_o3_load':
            # ------ Single core, no screen, 3rd order ------ 
            self.n = n = 3
            A = np.array([[-p.R/p.L, 1/p.L, -1/p.L],
                            [-1/(p.C1), -1/(p.Rin * p.C1), 0],
                            [1/(p.C2), 0, -1/(p.Rload*p.C2)],
                          ])
            
            B = np.array([[0], [1 / (p.Rin * p.C1)], [0]])


        elif isinstance(model,str) and model == 'c1_s0_o2':
            # ------ Single core, no screen, 2rd order ------ 
            self.n = n = 2
            A = np.array([[-p.R/p.L,-1/(p.L)],
                          [1/(p.C2),0]])

            B = np.array([[1/p.L],[0]])
           
        # Generalized output matrices
        C = np.eye(n)
        D = np.zeros(1)
                
        return A, B, C, D

    
    def get_model(self,model,
                   discretize:bool=False,
                   dt:float=None,
                   params:dict=None,
                   pu:bool=True,
                   silence=False):
        """
        :param model:
        :param discretize:
        :param dt:
        :param params:
        :return:
        """
        # ------ Load parameters------
        self.p = p = ParamWrapper(params,pu=pu)

        # ------ Load model ------
        self.model = model
        A, B, C, D = self.load_model(model)

        # ------ Discretization ------
        if discretize:
            if dt is None:
                raise ValueError('dt must be defined to discretize the system')
            else:                
                # A, B, C, D, dt = scipy.signal.cont2discrete((A, B, C, D),dt)
                A_d,B_d,C,D = self.c2d(A,B,C,D,dt)
                
        # ------ Data storage ------
        # Store state and relevant data
        self.A, self.B, self.C, self.D = A, B, C, D
        self.discrete = discretize
        self.dt = dt
        if discretize:            
            self.condition_number = np.linalg.cond(A_d)
            self.lambd = np.linalg.eigvals(A_d)
            self.A_unstable = np.real(self.lambd).any() > 1
        else:
            self.condition_number = np.linalg.cond(A)
            self.lambd = np.linalg.eigvals(A)
            self.A_unstable = np.real(self.lambd).any() > 0

        # ------ Print statements ------
        # print model configuration
        if not silence:
            if discretize:
                print('',f'Model: {model}','A=',A_d,'B=',B_d,'C=',C,'D=',D,'Lambdas=',*list(self.lambd),'',f'Condition number:\t{self.condition_number}',f'A matrix unstable:\t{self.A_unstable}',sep='\n') 
            else:
                print('',f'Model: {model}','A=',A,'B=',B,'C=',C,'D=',D,'Lambdas=',*list(self.lambd),'',f'Condition number:\t{self.condition_number}',f'A matrix unstable:\t{self.A_unstable}',sep='\n') 
        
        return
    
    
    def create_input(self, t1, t2, dt, amp=None, phi=None, t0:float = 0, mode:str='sin'):

        # if Sx is None: Sx = lambda: sx_random*((sx,1)[sx is None])*np.random.randn(m) + mu_x
        # if Sy is None: Sy = lambda: sy_random*((sy,1)[sy is None])*np.random.randn(m) + mu_y        

        if mode not in ['sin','step','impulse']:
            raise ValueError("Mode must be 'sin','step', or 'impulse'")

        
        # Parameter selection
        if phi is None: phi = self.p.phi
        if amp is None: amp = self.p.V
        
        # Function evaluation
        if mode == 'sin':
            u = lambda t: amp*np.sin(self.p.omega*t+phi)*(0,1)[t>=t0]
        elif mode == 'step':
            u = lambda t: amp*(0,1)[t>=t0]
        elif mode == 'impulse':
            u = lambda t: amp*(0,1)[t0<= t and t < t0 + dt]
        
        # Evaluate u(k) for each k
        time = np.arange(t1,t2+dt,dt)
        uk = np.array([u(k) for k in time])

        return u, uk
    
    #============================= Evaluate functions =============================#
    def evaluate_condition_number(self,params,n=100):
        data = {'Rin':[],
                'Rload':[],
                'condition':[],
                }

        for rin in np.linspace(0.05,100,n):
            print('asdf')
            for rload in np.linspace(0.2,1e6,n):
                print(rin,rload)
                params['Rin'] = rin
                params['Rload'] = rload
                

                # ------ Load parameters------
                p = ParamWrapper(params,pu=True)
        
                # ------ Load model ------
                A, B, C, D = self.load_model(p=p)
            
                data['Rin'].append(rin)                
                data['Rload'].append(rload)                
                data['condition'].append(np.linalg.cond(A))                                
                
        return pd.DataFrame(data,columns=list(data.keys()))
    
    #============================= STAT METHODS =============================#
    def quantile_confidence_interval(self, data, alpha):
        # Calculate the sample mean and standard deviation
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
    
        # Calculate the critical value for the desired level of confidence and degree of freedom
        n_samples = len(data)
        df = n_samples - 1
        if n_samples <= 30:
            critical_value = stats.t.ppf((1-alpha)/2, df)
        else:
            critical_value = stats.norm.ppf((1-alpha)/2)
    
        # Calculate the standard error of the sample mean
        standard_error = sample_std / np.sqrt(n_samples)
    
        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = sample_mean - (critical_value * standard_error)
        upper_bound = sample_mean + (critical_value * standard_error)
    
        # Return the lower and upper bounds as a tuple
        return (lower_bound, upper_bound)

    
    
    #============================= PLOT METHODS =============================#

    def plot_simulations(self,t,sims,labels:list=None,colors=clrs,figsize=(9,6),grid:bool=True,save=None,file_extension='pdf',path=None):
        fig, ax = plt.subplots(sims[0].shape[0],1,figsize=figsize,dpi=200,sharex=True)


        # input validation
        if labels is not None and len(sims) != len(labels):
            raise ValueError('Number of simulations and labels are not having same length')
        
        # Define linestyles
        lss=['-','--','-.',':']        
        
        for k, sim in enumerate(sims):
            for i in range(sim.shape[0]):
                idxs = ~np.isnan(sim[i]) & ~(abs(sim[i]) > 2)
                if labels is not None:
                    ax[i].plot(t[idxs],sim[i,idxs],color=colors[k],ls=lss[k%4],label=labels[k])
                else:
                    ax[i].plot(t[idxs],sim[i,idxs],color=colors[k],ls=lss[k%4])
        
        # Add grid lines
        if grid: [ax[i].grid() for i in range(sims[0].shape[0])]
                

        if labels is not None:
            ax[1].legend(ncols=2)
            
        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_sim.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_sim.{file_extension}',dpi=200)
            plt.close()
        return

    def plot_estimation_summary(self,thetahat,thetahat0,save=None,file_extension='pdf',path=None):
        fig, ax = plt.subplots(1,4,dpi=300,figsize=(9,3))
        error = abs(thetahat-self.A)
        titles = ['System\n$\\theta$','Initial guess\n$\\theta_0$','Estimation\n$\\hat{\\theta}$','Error\nnorm$(|\\hat{\\theta}-\\theta|)$'] # ='+str(round(np.linalg.norm(error),2))+'
        for i, data in enumerate([self.A,thetahat0,thetahat,error]):
            # if 'Error' in titles[i]:
            #     im = ax[i].imshow(data, norm=LogNorm(),cmap=cm.Reds)
            #     im = ax[i].imshow(data, norm=LogNorm(),cmap=cm.Reds)

            # else:
            #     im = ax[i].imshow(data,cmap=cm.Reds)
                
            im = ax[i].imshow(data,cmap=cm.Reds)
            ax[i].set(title=titles[i])            

            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)            
            cbar = fig.colorbar(im, cax=cax)
            
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_summ.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_summ.{file_extension}',dpi=200)
            plt.close()

        return
    
    def plot_eigenvalues(self,systems,labels=None,save=None,file_extension='pdf',path=None):
        # 
        z_xy = lambda z: ([real(x) for x in z],[imag(x) for x in z])

    
        # input validation
        if labels is not None and len(systems) != len(labels):
            raise ValueError('systems and labels are not having same length')
        
        # Custom markers
        markers = ['o','x','*','.']
        
        # 
        fig, ax = plt.subplots(1,1,dpi=300)
        # Go through all the systems
        for i, sys in enumerate(systems):
            # Get the eigenvalues
            lambd = eigvals(sys)

            x,y = z_xy(lambd)
            print(x,y)
            if labels is not None:
                ax.scatter(x,y,label=labels[i],zorder=1e3-i,alpha=0.75,marker=markers[i])
            else:
                ax.scatter(x,y,zorder=1e3-i,alpha=0.75,marker=markers[i%4])         
                    
        ax.axvline(0,color='k',lw=0.75)
        ax.axvline(1,color='k',lw=0.75)
        ax.axhline(0,color='k',lw=0.75)
        ax.grid()
        ax.legend(loc='upper left')

        # plt.gca().set_aspect('equal', adjustable='box')
        ax.set(xlabel='Real Part',ylabel='Imaginary Part')

        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_eig.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_eig.{file_extension}',dpi=200)
            plt.close()

        
        return

    
    def plot_covariance_in_time(self,covs):
        if isinstance(covs,np.ndarray):
            covs = [covs]
        
        fig, ax = plt.subplots(len(covs),1,dpi=200)
        if len(covs) > 1:
            for i, cov in enumerate(covs):
                print(cov.shape)
                x,y = cov.shape[0]*cov.shape[0], cov.shape[2]
                c = abs(cov.reshape((x,y)))
                im = ax[i].imshow(c,norm=LogNorm())
                ax[i].set_xticks([i for i in range(x)])
                ax[i].set_xticklabels([f'{i%cov.shape[0]}{i//cov.shape[0]}' for i in range(x)])        
        
                # Colorbar
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
        else:
            cov = covs[0]
            x,y = cov.shape[0]*cov.shape[0], cov.shape[2]
            c = abs(cov.reshape((x,y)))
            im = ax.imshow(c,norm=LogNorm())
            ax.set_yticks([i for i in range(x)])
            ax.set_yticklabels([f'{i%cov.shape[0]},{i//cov.shape[0]}' for i in range(x)])        
    
            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax)

            fig, ax = plt.subplots(1,1,dpi=200)
            plt.imshow(abs(cov[:,:,-1]),norm=LogNorm())
            


        return    
    
    #============================= PLOT METHODS =============================#
    
    def store_results(self,time,x,y,y_hat,):
        # res = utils.results_wrapper({'t':time,
        #                              'x':x,
        #                              'y':y,
        #                              'yhat':y_hat,
        #                              'xhat':x_hat,
        #                              'A_d_hat':A_d_hat,
        #                              'thetahat':thetahat,
        #                              'thetahat_guess':thetahat_guess,
        #                              'sigma2':sigma2,
        #                              'V':V,
        #                              'J':J,
        #                              'P':P,
        #                              'K':K,
        #                              'R':R,
        #                              'eps':eps,
        #                              'ML_est':ML_est,
        #                              'V':V,
        #                              })
        
        return

    def simulate(self,A,B,C,D,x0,u,t1,t2,dt):
        time = np.arange(t1,t2+dt,dt)

        # creating indices
        n_y = len(time)
        n = self.A.shape[0]
        m = self.C.shape[0]

        # Initialize arrays
        x   = np.zeros((n,n_y))
        y   = np.zeros((m,n_y))

        # Creating input function
        # if Sx is None: Sx = lambda: sx_random*((sx,1)[sx is None])*np.random.randn(m) + mu_x
        # if Sy is None: Sy = lambda: sy_random*((sy,1)[sy is None])*np.random.randn(m) + mu_y

        print('',f'Simulating discrete-time system:',sep='\n')
        t1_ = datetime.datetime.now()
        # initial values
        x[:,0] = x0
        y[:,0] = C @ x0 # + Sy() # TODO: Add noise, be aware of operator for u[k]
        for k,t in enumerate(time[:-1]):
            # --------------- Solving x[k+1] ---------------
            # Calculate x[:, k + 1]
            x[:,k+1] = A @ x[:,k] + B @ np.array([u[k]]) #+ Sx() # TODO: Add noise, be aware of operator for u[k]
            y[:,k] = C @ x[:,k] # + Sy() # TODO: Add noise, be aware of operator for u[k]
        y[:,-1] = C @ x[:,-1] # + Sy() # TODO: Add noise, be aware of operator for u[k]
        t2_ = datetime.datetime.now()            
        print(f'Finished in: {(t2_-t1_).total_seconds()} s')

        return x,y
    
    def KalmanFilter_CF(self,A,B,C,D,x0,u,y,R0,R1,R2,t1,t2,dt,optimization_routine:bool=False):
        if optimization_routine and len(A.shape) == 1:
            dim = int(np.sqrt(len(A)))
            A = A.reshape((dim,dim))
        time = np.arange(t1,t2+dt,dt)

        # creating indices
        n_y = len(time)
        n = self.A.shape[0]
        m = self.C.shape[0]

        # Initialize arrays
        eps = np.zeros((m,n_y))
        x_hat_pred   = np.zeros((n,n_y))
        y_hat_pred   = np.zeros((m,n_y))
        P_pred       = np.zeros((n,n,n_y))
        K       = np.zeros((n,n,n_y))
        R       = np.zeros((n,n,n_y))

        # if Sx is None: Sx = lambda: sx_random*((sx,1)[sx is None])*np.random.randn(m) + mu_x
        # if Sy is None: Sy = lambda: sy_random*((sy,1)[sy is None])*np.random.randn(m) + mu_y
        # Creating input function
        # u = lambda t: self.p.V*np.sin(self.p.omega*t+self.p.phi)*(0,1)[t>=0.1]
        
        # initial values
        x_hat_pred[:,0] = x0
        P_pred[:,:,0] = R[:,:,0] = R0
        for k,t in enumerate(time):
            # --------------- Calculating output ---------------
            # Calculating output and estimated output
            y_hat_pred[:,k] = C @ x_hat_pred[:,k] # + D @ u[k]
            # Calculating error / innovation term
            eps[:,k] = y[:,k] - y_hat_pred[:,k]
    
            # --------------- Discrete-time Kalman Filter  ---------------
            # Calculate kalman gain
            R[:, :, k] = C @ P_pred[:, :, k] @ C.T + R2 # R2 = Measurement noise covariance matrix.
            K[:,:,k] = P_pred[:,:,k] @ C.T @ inv(R[:,:,k])
    
            # Time update
            if k < len(time)-1:
                x_hat_pred[:,k+1] = A @ x_hat_pred[:,k] + B @ np.array([u[k]]) + K[:,:,k] @ eps[:,k]
                P_pred[:, :, k+1] = A @ P_pred[:,:,k] @ A.T - K[:,:,k] @ C @ P_pred[:,:,k] @ A.T # + R1# TODO: R1 = B @ R1 @ B.T
                # x_hat_pred[:,k+1] = A @ x_hat_pred[:,k] + B @ np.array([u(t)])
                # P_pred[:, :, k+1] = A @ P_pred[:,:,k] @ A.T + R1# TODO: R1 = B @ R1 @ B.T
        
        return x_hat_pred, y_hat_pred, eps, R        


    def LS(self,x,y,dt,u=None):
        # Input validation        
        if x.shape != y.shape:
            raise ValueError(f'x and y do not have same dimensions: {x.shape} != {y.shape}')

        # If u are provided
        if u is not None:
            # Initialize
            n,m = x.shape
            n += (u.shape[0],1)[len(u.shape)==1]
            thetahat = np.zeros((n,n,m))
            x = np.vstack([x,u])
            y = np.vstack([y,np.zeros_like(u)])
            
            for k in range(m):
                # thetahat[:,:,k] = inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2) @ x[:,k].reshape((n,1)) @ y[:,k].reshape((1,n))  
                thetahat[:,:,k] =  y[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) @ inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2)
                
            
        else:
            # Initialize
            n,m = x.shape
            thetahat = np.zeros((n,n,m))
            for k in range(m):
                thetahat[:,:,k] = inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2) @ x[:,k].reshape((n,1)) @ y[:,k].reshape((1,n))  
        
        
        return thetahat
    

    def KalmanFilter(self,A,B,C,D,x0,u,y,R0,R1,R2,t1,t2,dt,optimization_routine:bool=False):
        if optimization_routine and len(A.shape) == 1:
            dim = int(np.sqrt(len(A)))
            A = A.reshape((dim,dim))
            
        time = np.arange(t1,t2+dt,dt)

        # creating indices
        n_y = len(time)
        n = A.shape[0]
        m = C.shape[0]

        # Initialize arrays
        eps = np.zeros((m,n_y))
        x_hat_filt   = np.zeros((n,n_y))
        x_hat_pred   = np.zeros((n,n_y))
        y_hat_pred   = np.zeros((m,n_y))
        P_filt       = np.zeros((n,n,n_y))
        P_pred       = np.zeros((n,n,n_y))
        K       = np.zeros((n,n,n_y))
        R       = np.zeros((n,n,n_y))
        
        # initial values
        x_hat_pred[:,0] = x0
        P_pred[:,:,0] = R[:,:,0] = R0
        for k,t in enumerate(time):
            # --------------- Calculating output ---------------
            # Calculating output and estimated output
            y_hat_pred[:,k] = C @ x_hat_pred[:,k] # + D @ u[k]

            # Calculating error / innovation term
            eps[:,k] = y[:,k] - y_hat_pred[:,k]
    
            # --------------- Discrete-time Kalman Filter  ---------------
            # Calculate kalman gain
            R[:, :, k] = C @ P_pred[:, :, k] @ C.T + R2 # R2 = Measurement noise covariance matrix.
            K[:,:,k] = P_pred[:,:,k] @ C.T @ inv(R[:,:,k])
    
            # Measurement update
            x_hat_filt[:,k] = x_hat_pred[:,k] + K[:,:,k] @ eps[:,k]
            P_filt[:,:,k] = (np.eye(n) - K[:,:,k] @ C) @ P_pred[:,:,k]

            # Time update
            if k < len(time)-1:
                x_hat_pred[:,k+1] = A @ x_hat_filt[:,k] + B @ np.array([u[k]])
                P_pred[:, :, k+1] = A @ P_filt[:,:,k] @ A.T + R1 # TODO: R1 = B @ R1 @ B.T
        
        return x_hat_pred, y_hat_pred, eps, R

    #============================= ESTIMATION METHODS =============================#    
    
    def dt_to_ct_zoh(self, H_z, Ts, T):
        """
        Transforms a discrete-time system into a continuous-time system using the Zero-Order Hold (ZOH) method.
        
        Parameters
        ----------
        H_z : array_like
            The transfer function of the discrete-time system in z-domain.
        Ts : float
            The sampling period of the discrete-time system.
        T : float
            The duration of the rectangular pulse used in the ZOH method.
            
        Returns
        -------
        H_s : array_like
            The transfer function of the continuous-time system in s-domain.
        """
        w, H_ejw = scipy.signal.freqz(H_z)
        H_r = np.sinc(w / np.pi / Ts)
        H_cjw = H_ejw * H_r
        t, h_ct = scipy.signal.freqz(H_cjw, [1, 0], worN=2**14)
        h_ct = np.real(h_ct)
        H_s = scipy.signal.TransferFunction(h_ct, [1, 0])
        return H_s


    def c2d(self,A,B,C,D,dt):

        if len(A.shape) == 1:
            n = int(np.sqrt(A.shape[0]))
            A = A.reshape((n,n))

        # Calculate the continuous-time state transition matrix using the matrix exponential
        At = np.block([[A, B], [np.zeros((1, A.shape[1])), np.zeros((1, 1))]])
        eAt = scipy.linalg.expm(At * dt)
        Ad = eAt[:A.shape[0], :A.shape[1]]
        Bd = eAt[:A.shape[0], -1:]          

        # assign to class
        self.A_d, self.B_d = Ad, Bd
        
        return Ad, Bd, C, D

    def d2c(self, A_d, B_d, C_d, D_d, Ts, T):
        
        n = A_d.shape[0]
        # Ad = np.block([[A_d, B_d], [np.zeros((1, n)), 1]])
        # Bd = np.block([B_d, 0]).reshape(n + 1, 1)
        # Cd = np.block([C_d, D_d]).reshape(1, n + 1)
        
        H_z = scipy.signal.ss2tf(A_d, B_d, C_d, D_d)[0]
        H_s = self.dt_to_ct_zoh(H_z, Ts, T)
        A_c, B_c, C_c, D_c = scipy.signal.tf2ss(H_s.num, H_s.den)
        return A_c[:, :n], B_c[:n], C_c[:, :n], D_c
    #============================= MODEL CHECKING METHODS =============================#
    
    def runs_test(residuals, alpha=0.05):
        """
        Performs a runs test for a change in signs in the residuals of a linear regression model.
    
        Parameters:
            residuals (array-like): An array of residuals from a linear regression model.
            alpha (float): The desired significance level (default is 0.05).
    
        Returns:
            result (string): A string indicating whether there is evidence of a change in signs in the residuals.
            p_value (float): The p-value associated with the test statistic.
            test_statistic (float): The value of the test statistic.
        """
        n = len(residuals)
        runs = 1
        for i in range(1, n):
            if np.sign(residuals[i]) != np.sign(residuals[i-1]):
                runs += 1
        expected_runs = (2*n - 1) / 3
        variance = (16*n - 29) / 90
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        if p_value < alpha:
            result = "Evidence of a change in signs in the residuals."
        else:
            result = "No evidence of a change in signs in the residuals."
        return result, p_value, z

    def residual_analysis(self,eps):
        single = len(eps.shape)==1
        n = (eps.shape[0],1)[len(eps.shape)==1]
        fig, ax = plt.subplots(n,1,sharex=True,dpi=200)

        for i in range(n):
            if single:
                ax.plot(eps)
            else:
                ax[i].plot(eps[i,:])
                
            

        # TODO: Calculate sign changes
            
        return
    
    #============================= ESTIMATION METHODS =============================#
    import numpy as np
    from scipy import stats
    
    def kernel_density_estimate(self,matrix):
        print('',f'Starting Kernel Density Estimation (KDE)',sep='\n')
        t1_ = datetime.datetime.now()

        n = matrix.shape[0]
        KDE = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                
                w, v = np.histogram(matrix[i,j,:])

                # Get index of most likely        
                idx = list(w).index(max(w))
                
                KDE[i,j] = v[idx:idx+2].mean()

        t2_ = datetime.datetime.now()
        print(f'Finished in: {(t2_-t1_).total_seconds()} s')

        return KDE



    def ML_Wrapper(self,theta,A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt,r_idx,c_idx,opt_in_z):
        # Load model
        A, B, C, D = self.load_model()

        # ------ Choose optimization domain ------
        # Discrete-time
        if opt_in_z:            
            # Discretize continous system
            A_d,B_d,C,D = self.c2d(A,B,C,D,dt)

            # assigning theta to discrete system
            if r_idx is None or c_idx is None:
                A_d = theta
            else:
                A_d[r_idx,c_idx] = theta
            
        # Continous-time
        else:    
            # assigning theta to continous system
            if r_idx is None or c_idx is None:
                A = theta
            else:
                A[r_idx,c_idx] = theta
    
            # Discretize continous system
            A_d,B_d,C,D = self.c2d(A,B,C,D,dt)
  
        # Evaluating cost function
        J = self.ML(A_d,B_d,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt)
        
        return J

    
    def ML(self,A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt, matrix:bool = False):        

        # Get covariance and prediction error
        _,_, eps,R = self.KalmanFilter(A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,optimization_routine=True)

        # Evaluate the cost function from the batch sum
        J = 0
        for k in range(eps.shape[1]):
            J += 1/2*(np.log(det(R[:,:,k])) \
                      + np.log(2*np.pi)\
                      + eps[:,k].T @ inv(R[:,:,k]) @ eps[:,k]) 
                
        return J
        
    def ML_opt(self,A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt, thetahat0,r_idx=None,c_idx=None,e_tol=1e-32,opt_in_z=True):
        
        # If element wise, make sure to collect elementwise thetahat
        if r_idx is None and c_idx is None:
            print('',f'Starting maximum likelihood estimation of dim(A)={A.shape}:',sep='\n')
            t1_ = datetime.datetime.now()

            
        # Minimization
        thetahat = minimize(self.ML_Wrapper,
                            args=(A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt,r_idx,c_idx,opt_in_z),
                            x0=thetahat0,
                            tol=e_tol)

        # Print summary
        if r_idx is None or c_idx is None:
            dim = int(np.sqrt(len(thetahat.x)))
            thetahat.x = thetahat.x.reshape((dim,dim))            
            t2_ = datetime.datetime.now()
            print(f'Finished in: {(t2_-t1_).total_seconds()} s')
            print(f'#========= ESTIMATION SUMMARY OF A[:,:] =========#')
            print('theta:\n',self.A)
            print('thetahat0:\n',thetahat0)
            print('thetahat:\n',thetahat.x)
            print('Deviation:\n',(thetahat.x - self.A))
            print('2-norm:\n',norm(thetahat.x - self.A))
            print('Eigenvalues:\n',eigvals(thetahat.x))
            print('')
        else:
            print(f'#========= ESTIMATION SUMMARY OF A[{r_idx},{c_idx}] =========#')
            print('theta:\t\t',self.A[r_idx,c_idx])
            print('thetahat0:\t',thetahat0)
            print('thetahat:\t',thetahat.x)
            print('Deviation:\t',(thetahat.x - self.A[r_idx,c_idx]))
            # print('1-norm:\n',abs(thetahat.x - self.A))
            print('')
            
        return thetahat, thetahat0

    def ML_opt_elm(self,A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt, thetahat0, opt_in_z=True):               

        ests = np.zeros(self.A.shape)
        devs = np.zeros(self.A.shape)
        thetahat0s = np.zeros(self.A.shape)
        print('',f'Starting maximum likelihood estimation of an individual element:',sep='\n')
        t1_ = datetime.datetime.now()
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):        
                thetahat, thetahat0_ = self.ML_opt(A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,thetahat0[i,j],r_idx=i,c_idx=j,opt_in_z=opt_in_z)
                ests[i,j] = thetahat.x
                devs[i,j] = thetahat.x - A[i,j]
                thetahat0s[i,j] = thetahat0_
        t2_ = datetime.datetime.now()            
        print(f'Finished in: {(t2_-t1_).total_seconds()} s')

        print(f'#========= FINAL ESTIMATION SUMMARY =========#')
        print('System',A,sep='\n')
        print('Estimated',ests,sep='\n')
        print('Initial guess',thetahat0s,sep='\n')
        print('Deviations',devs,sep='\n')
        print('2-norm:\n',np.linalg.norm(devs))
        print('Eigenvalues:\n',eigvals(ests))
        print('')
        return ests,thetahat0s



