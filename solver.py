import numpy as np
from scipy.integrate import solve_ivp
import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import SITOBBDS_utils as utils
from numerical_methods import NumericalMethods
import datetime
import scipy

#
class Solver:
    def __init__(self):
        self.method = NumericalMethods()
        return
    
    def load_model(self,model,
                   discretize:bool=False,
                   dt:float=None,
                   params:dict=None,
                   domain:str = 't'):
        """
        :param model:
        :param discretize:
        :param dt:
        :param params:
        :return:
        """
        # ------ Load parameters------
        R = 1
        L = 1e-3
        C1 = 0.5e-6
        C2 = 0.5e-6
        Rin = 0.05 # BRK = 0.05
        # Source
        self.V = 66e3
        self.f = 50
        self.omega = 2*np.pi*self.f
        self.phi = 0
        # 
        if params is not None:
            for k, v in params.items():
                setattr(self,k,v)

        
        # ------ Load model ------
        if isinstance(model, (list,tuple)):
            # ------ CUSTOM MODEL ------ 
            model='Custom'
            self.n = n = 1
            raise NotImplementedError('')
            
        elif isinstance(model,str) and model == 'c1_s0_o3':
            # ------ Single core, no screen, 3rd order ------ 
            self.n = n = 3
            A = np.array([[-R/L, 1/L, -1/L],
                            [-1/(C1), -1/(Rin * C1), 0],
                            [1/(C2), 0, 0],
                          ])
            
            B = np.array([[0], [1 / (Rin * C1)], [0]])
        elif isinstance(model,str) and model == 'c1_s0_o2':
            # ------ Single core, no screen, 2rd order ------ 
            self.n = n = 2
            A = np.array([[-R/L,-1/(L)],
                          [1/(C2),0]])

            B = np.array([[1/L],[0]])
           
        # Generalized output matrices
        C = np.eye(n)
        D = np.zeros(1)


        # Discretize if needed
        if discretize:
            if dt is None:
                raise ValueError('dt must be defined to discretize the system')
            else:

                # A, B, C, D, dt = scipy.signal.cont2discrete((A, B, C, D),dt)
                A, B, C, D = self.method.c2d(A,B,C,D,dt,domain=domain)

        # Store state and relevant data
        self.A, self.B, self.C, self.D = A, B, C, D
        self.discretize = discretize
        self.dt = dt
        self.condition_number = np.linalg.cond(A)
        self.lambd = np.linalg.eigvals(A)
        self.A_unstable = np.real(self.lambd).any() > (0,1)[discretize]

        # print
        print('',f'Model: {model}','A=',A,'B=',B,'C=',C,'D=',D,'Lambdas=',*list(self.lambd),'',f'Condition number:\t{self.condition_number}',f'A matrix unstable:\t{self.A_unstable}',sep='\n') 
                        
        # The condition number of x is defined as the norm of x times the norm of the inverse of x [1]; the norm can be the usual L2-norm (root-of-sum-of-squares) or one of a number of other matrix norms.
        
        return

    def solve(self,t_start,t_end,u=None,dt=None,x0=None,method='RK45'):
        """

        :param t_start:
        :param t_end:
        :param u:
        :param dt:
        :param x0:
        :param method:
        :return:
        """
        # ========== Input validation ==========
        if self.dt is None and dt is None:
            raise ValueError('Please provide dt if the system is not discretized')
        elif self.dt is not None and dt is not None and self.dt != dt:
            raise Warning(f'System is discretized for {self.dt} but you are trying to solve for {dt}')
        elif self.dt is not None:
            dt = self.dt
        elif dt is not None:
            self.dt = dt
            
        # ========== Initializing ==========
        # Creating initial conditions
        if x0 is None:
            x0 = np.zeros(self.n)
        self.x0 = x0

        # Creating input function
        if u is None:
            u = lambda t: self.V*np.sin(self.omega*t+self.phi)*(0,1)[t>=0.1]
    
        # Creating time series
        time = np.arange(t_start,t_end+dt,dt)
        time = np.round(time,7)        

        # ========== DISCRETE-TIME SOLUTION ==========
        if self.discretize:            
            f = lambda t,x,u,dt: self.A @ x + self.B @ np.array([u(t)])

            # initialize state matrix
            x = np.empty((self.n,len(time)))

            print('',f'Solving discrete-time system:',sep='\n')
            t1 = datetime.datetime.now()

            x[:,0] = x0
            for k,t in enumerate(time[:-1]):
                # dx = x + dt * f(t,x,u)
                x[:,k+1] = self.method.euler_step(x[:,k], u, f, t, dt)

            t2 = datetime.datetime.now()            
            print(f'Finished in: {(t2-t1).total_seconds()}')

            res = utils.results_wrapper({'t':time,
                                         'x':x,
                                         'y':self.C @ x, #  + self.D @ u(time)
                                         })
            
        # ========== CONTINUOUS-TIME SOLUTION ==========
        else:
            # creating solution matrix
            f = lambda t, x: self.A @ x + self.B @ np.array([u(t)])

            print(f'Solving continuous-time system:')
            t1 = datetime.datetime.now()
            res = solve_ivp(f, [time[0], time[-1]], x0, t_eval=time, method=method)  #
            t2 = datetime.datetime.now()
            print(f'Finished in: {(t2-t1).total_seconds()}')

        return res








