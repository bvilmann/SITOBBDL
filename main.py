#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os

#%%
sol = Solver()

sol.load_model('c1_s0_o3',discretize=False,dt=10e-6,domain='z')
# sol.load_model('c1_s0_o3',dt=10e-6)

res = sol.solve(0.095,1.3)

#%%
fig, ax = plt.subplots(3,1,dpi=300,sharex=True)

for i in range(3):
    ax[i].plot(res.t,res.y[i])
    
# ax[-1].set_xlim(0.095,0.13)
   

#%%asdf
asdf

xlabs = ['$I_L$ [A]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']
def plot_sols(sols,pscad:list=None,labs=None,names:list=None,t_pscad=None,clr_pscad='gold'):
    
    # Axis handling
    if pscad is not None:
        
        N = len(pscad)
    else:        
        N = max([r.y.shape[0] for r in res])

    # Axis handling
    fig,ax = plt.subplots(N,1,sharex=True,dpi=300)

    # Axis handling
    for i, r in enumerate(res):
        for j, y in enumerate(r.y):
            if i == 0:
                ax[j].set_ylabel(('',labs[j])[labs is not None])
                ax[i].grid()
            ax[j].plot(r.t,y/1e3,label=('',),color='teal',alpha=0.5,zorder=4)
    
    # plotting pscad
    if pscad is not None:
        for j, y in enumerate(pscad):
            ax[j].plot(t_pscad,y,label='PSCAD',color=clr_pscad,alpha=0.5,zorder=5)
            
    ax[-1].legend()
    
    return

# plot_sols([res_1_1,res_1_2],pscad[iabc1,vabc1,vabc1])


def plot(name=None,xlim=(0.095,0.13)):

    xlabs = ['$I_L$ [A]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']
    
    ys = [iabc2,vabc1,vabc2]
    fig,ax = plt.subplots(len(res_1.y),1,sharex=True,dpi=300)
    cnt = 0
    for i,y in enumerate(res_1.y):
        print(xlabs[i])
        ax[i].plot(res_1.t,y/1e3,label='Python, $\\varrho^3$',color='black',alpha=0.5,zorder=4,lw=3)
        if i in [0,2]:
            # if 'V_{c1}' in xlabs[i]:
            #     ax[i].plot(t,[u(x)/1000 for x in t],label='Python1',color='blue',alpha=0.5)
    
            ax[i].plot(res_2.t,res_2.y[cnt]/1000,label='Python, $\\varrho^2$',color='blue',alpha=0.5)
            cnt += 1

        
        ax[i].plot(t_pscad,ys[i],label='PSCAD',color='gold',ls=('-','--')[i>0],alpha=0.75,zorder=5)
        ax[i].set_ylabel(xlabs[i])
        ax[i].grid()
    ax[-1].set_xlim(*xlim)
    
    ax[-1].legend()
    ax[-1].set_xlabel('Time [s]')
    if name is not None:
        plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img\sim'+f'\\{name}_{xlim[0]}_{xlim[1]}.pdf')
    plt.show()
    plt.close()
    return fig, ax




#%% DATA READER

rpath = f'{os.getcwd()}\\data'
fname = r'cable_1phase.infx'
file_addr = f'{rpath}\\{fname}'
dread = DataReader()
_,iabc1 = dread.get_signal('Iabc_b1',file_addr)
_,iabc2 = dread.get_signal('Iabc_b2',file_addr)

_,vabc1 = dread.get_signal('Vabc_b1',file_addr)
t_pscad,vabc2 = dread.get_signal('Vabc_b2',file_addr)

#%% SIMULATION SETTINGS
time_step = 10e-6
f = 50
phi= 0
omega = 2*np.pi*f
V = 66e3*np.sqrt(2)

# %% Power system - admittance matrix
ps = PS(f = 50, Vbase = 66e3, Sbase = 100e6)

# Adding lines and transformers (bus1, bus2, r, x)
ps.add_line(1, 2, 0.04, 0.52, 'Za')

# Adding generators
ps.add_gen(1,1)

# Adding loads
ps.add_load(2,'load', complex(.1,.2))
# ps.add_load(2,'load1', complex(.3,.2))

Z, Y, B = ps.build()

#%% LUMPED PARAMETER - SINGLE CABLE (CONDUCTOR ONLY)
# ------ PARAMETERS ------ 
R = 1
L = 1e-3
C1 = 0.5e-6
C2 = 0.5e-6
Rin = 0.05 # BRK = 0.05
# Rin = 1 # BRK = 0.05

# ------ INPUT FUNCTIONS ------ 
u = lambda t: V*np.sin(omega*t+phi)*(0,1)[t>=0.1]

#%%
Z1 = np.poly1d([L/C2])
Z2 = np.poly1d([1,R/L,L/C2])
num = list(Z1*Z1)
den = list(Z2*Z2)

Z = ctrl.tf(num,den)


ctrl.bode(Z)

#%%
name = '1ph_c_1'

# ------ MATRICES ------ 
A = np.array([[-R/L, 1/L, -1/L],
                [-1/(C1), -1/(Rin * C1), 0],
                [1/(C2), 0, 0],
              ])

B = np.array([[0], [1 / (Rin * C1)], [0]])


# ------ SOLVING ODE/PDE ------ 
sol_1 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
# res_1 = sol_1.run_dynamic_simulation(u)
plot()

#%% System 
ss = ctrl.ss(A,B,np.diag(np.ones(3)),0)

fn =ss.damp()[0]/(2*np.pi)

# %%
# sols = 

# RLC, with load
# x = [di/dt, dvc2/dt]
name = '1ph_c_1'

# 1) Match no resonant coupling
A = np.array([[-R/L,-1/(L)],
              [1/(C2),0]])

B = np.array([[1/L],[0]])

# plot(name=name,xlim=(0.095,0.13))
# plot(name=name,xlim=(0.0995,0.102))

sol_2 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
res_2 = sol_2.run_dynamic_simulation(u)

#%% Kalman
"""
P0 (ndarray): Initial state covariance matrix, with shape (num_states, num_states)
Q (ndarray): Process noise covariance matrix, with shape (num_states, num_states)
R (ndarray): Measurement noise covariance matrix, with shape (num_outputs, num_outputs)
"""
u_ = np.zeros((1,len(res_2.t)))
u_[0,:] = [u(t) for t in res_2.t]


# second order
P0 = np.diag(np.ones(2))*1
S_p = np.diag(np.ones(2))
S_m = np.diag(np.ones(2))
x_hat2, y_hat2, stats2 = sol_2.kalman_filter(res_2.y.T ,u_.T, P0, S_p, S_m)

# third order
P0 = np.diag(np.ones(3))*1
S_p = np.diag(np.ones(3))
S_m = np.diag(np.ones(3))
x_hat1, y_hat1, stats1 = sol_1.kalman_filter(res_1.y.T ,u_.T, P0, S_p, S_m)

u_ = np.zeros((1,len(t_pscad)))
u_[0,:] = [u(t) for t in t_pscad]
x_hat3, y_hat3, stats3 = sol_1.kalman_filter(np.hstack([iabc2*1000,vabc1*1000,vabc2*1000]) ,u_.T, P0, S_p, S_m)

# PLOTTING estimates and outputs
fig,ax = plt.subplots(3,1,sharex=True,dpi=300)
labs = ['$I_L$ [kA]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']
# labs = ['$I_L$ [A]','$V_{c2}$ [kV]']
cnt = 0
for i in range(3):
    print(i)
    # ax[i].plot(t_pscad,[iabc2,vabc1,vabc2][i],label='$Y_{PSCAD}$')    
    ax[i].plot(t_pscad,y_hat3.T[i],label='$\\hat{Y}_{PSCAD}$',alpha = 0.5)    
    # ax[i].plot(res_1.t,res_1.y[i]/1000,label='$Y_{\\varrho^3}$')    
    # ax[i].plot(res_1.t,y_hat1.T[i]/1000,label='$\\hat{Y}_{\\varrho^3}$',alpha = 0.5)    
    if i in [0,2]:
        # ax[i].plot(res_2.t,res_2.y[cnt]/1000,label='$Y_{\\varrho^2}$')    
        # ax[i].plot(res_2.t,y_hat2.T[cnt]/1000,label='$\\hat{Y}_{\\varrho^2}$',alpha = 0.5)    
        cnt += 1
    ax[i].grid()
    ax[i].set_ylabel(labs[i])
    
    ax[i].legend(loc='lower right',ncols=(2,3)[i!=1])


from matplotlib.colors import LogNorm

#%%

I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02
# t2 = 0.005
# t2 = 0.15
t1 = 0
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
Nx = np.array([1e-7,1e-4,1e-4])*0
r123 = np.array([1,1,1])*1e-4

R0=np.diag(r123)*1e5
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1

#%%



#%% EVAULATE CONDITION NUMBER
m = SITOBBDS()
n = 10
data = {'Rin':[],
        'Rload':[],
        'L':[],
        'R':[],
        'condition':[],
        }
for rin in np.linspace(0.05,100,n):
    for rload in np.linspace(0.2,1e5,n):
        for l in np.linspace(1e-4,1e-2,n):
            for r in np.linspace(0,100,n):
                params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4}
                m = SITOBBDS()
                m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True,silence=True)
            
                data['Rin'].append(rin)                
                data['Rload'].append(rload)                
                data['L'].append(l)                
                data['R'].append(r)                
                data['condition'].append(np.linalg.cond(m.A))                                

df = pd.DataFrame(data,columns=list(data.keys()))

table = pd.pivot(df.groupby(['Rin','Rload','condition']).mean(),columns='Rin',index='Rload',values='condition')

fig, ax=plt.subplots(1,1,dpi=300)
im = ax.imshow(table,norm=LogNorm(),extent=(df.Rin.min(), df.Rin.max(), df.Rload.max(), df.Rload.min()),aspect='auto') # (left, right, bottom, top) 
ax.set(ylabel='Rload',xlabel='Rin')
cbar = fig.colorbar(im)
plt.show()


#%%
import seaborn as sns
g = sns.PairGrid(df)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)

#%%
# fig, ax = plt.subplots(4,4,figsize=(12,12),dpi=200)

# for i,r in enumerate(list(set(df.columns)-set(['condition']))):
#     for j,c in enumerate(list(set(df.columns)-set(['condition']))):
#         if r == c:
#             continue
            
#         df_temp = df[[r,c,'condition']].drop_duplicates().reset_index()[[r,c,'condition']]        
#         table = pd.pivot(df_temp,columns=r,index=c,values='condition')
#         ax[i,j].imshow(table,norm=LogNorm(),extent=(df[r].min(), df[r].max(), df[c].max(), df[c].min()),aspect='auto') # (left, right, bottom, top))        
#         ax[i,0].set(ylabel=r)
#         ax[-1,j].set(xlabel=c)

m = SITOBBDS()
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

# --------------- ESTIMATE --------------- 
# LS - LEAST SQUARES
thetahat = m.LS(xhat,yhat,dt,u=uk)
# thetahat = m.LS(xhat,yhat,dt)

#%%
# KDE - KERNEL DENSITY ESTIMATION
KDE = m.kernel_density_estimate(thetahat)

#%%
# MLE - MAXIMUM LIKELIHOOD ESTIMATION
thetahat_, thetahat0_ = m.ML_opt(Ad,Bd,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= np.zeros((3,3)))
thetahat_mat = thetahat_.x

thetahat_elm, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,np.zeros((3,3)))

#%%
n = m.A_d.shape[0]
fig, ax =plt.subplots(n+1,n,sharex=True,dpi=200,figsize=(12,9))

for i in range(n+1):
    for j in range(n):
        if i ==0:
            ax[i,j].plot(t,yhat[j,:],color='red',label=['$I_L$','$V_{C1}$','$V_{C2}$'][j])
            ax[i,j].axhline(0,color='k',ls='-',lw=0.75)
        else:
            ax[i,j].set(yscale='symlog')
            
            ax[i,j].axhline(0,color='k',ls=':',lw=0.75,label='0')
            ax[i,j].axhline(m.A_d[i-1,j],color='k',ls='--',lw=0.75,label='True',zorder=3)
            ax[i,j].plot(t,thetahat[i-1,j,:],label='LS')
            ax[i,j].axhline(KDE[i-1,j],color='forestgreen',label='KDE',zorder=3)
            ax[i,j].axhline(thetahat_mat[i-1,j],color='red',ls='-',label='MLE_mat')
            ax[i,j].axhline(thetahat_elm[i-1,j],color='darkred',ls='-',lw=2,label='MLE_elm')
            # ax[i,j].axhline(m.Ad_elm[i-1,j],color='k',ls=':',lw=0.75,label='MLE_elm')
ax[-1,-1].legend(loc='upper right')

# # ESTIMATIONS OF ELEMENTS
# thetahat_elm, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)


# --------------- VALIDATE --------------- 
# DISCRETIZE
# Ad_mat,*_ = m.discretize(thetahat_mat, B, C, D, dt)
# Ad_elm,*_ = m.discretize(thetahat_elm, B, C, D, dt)

# # Simulate the system
# _, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
# _, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)



#%%
data = {'Rin':[],
        'Rload':[],
        'u_mode':[],
        'that':[],
        'phi':[],
        'success':[],
        'error_elm':[],
        'error_mat':[],
        'condition':[],
        }

t1 = -0.001
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

for u_mode in ['sin','step','impulse']:
    for rin in [0.05,0.5,1]:
        for rload in [5,43,100,1e6]:
            for that in [False,True]:
                for phi in [0,np.pi/4]:
                    # 'sin_0.5_100_False_0'
                    if that:                    
                        that0 = m.A
                    else:
                        that0 = -I(m.A.shape[0])*0
    
                    # Define parameters    
                    params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':phi}
        
                    # Get model
                    m = SITOBBDS()
                    m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
                    save_name=f'{u_mode}_{rin}_{rload}_{that}_{phi}'        
                    print(save_name,np.linalg.cond(m.A))
        
                    # Create input
                    u, uk = m.create_input(t1, t2, dt,mode=u_mode)        
                    
                    # Get matrices
                    Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
        
                    # --------------- GET GROUND TRUTH --------------- 
                    # Simulate the system
                    x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)
        
                    # Filter the data with the Kalman Filter
                    xhat, yhat,eps,R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)
        
                    try:
                        # --------------- ESTIMATE --------------- 
                        # ESTIMATION OF ENTIRE A
                        thetahat_, thetahat0_ = m.ML_opt(Ad,Bd,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= that0)
                        thetahat_mat = thetahat_.x
                        # ESTIMATIONS OF ELEMENTS
                        thetahat_elm, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)
            
                        # --------------- VALIDATE --------------- 
                        # DISCRETIZE
                        Ad_mat,*_ = m.discretize(thetahat_mat, B, C, D, dt)
                        Ad_elm,*_ = m.discretize(thetahat_elm, B, C, D, dt)
                        
                        # Simulate the system
                        _, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
                        _, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)
                        
                        # --------------- PLOT --------------- 
                        m.plot_simulations(t, [y,yhat,yhat_mat,yhat_elm],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$','$\\hat{y}_{MLE_{ELM}}$'],save=save_name,file_extension='png')
            
                        m.plot_estimation_summary(thetahat_elm, thetahat0,save=save_name + '_elm',file_extension='png')
                        m.plot_estimation_summary(thetahat_mat, thetahat0_,save=save_name + '_mat',file_extension='png')
                        m.plot_eigenvalues([m.A,thetahat_elm,thetahat_mat],['Ground truth','MLE - Element-wise','MLE - Matrix-wise'],save=save_name,file_extension='png')

                        # --------------- STORE VALUES --------------- 
                        data['Rin'].append(rin)                
                        data['Rload'].append(rload)                
                        data['u_mode'].append(u_mode)                
                        data['that'].append(that)                
                        data['phi'].append(phi)                
                        data['error_elm'].append(np.linalg.norm(thetahat_elm - m.A))                
                        data['error_mat'].append(np.linalg.norm(thetahat_mat - m.A))                
                        data['condition'].append(np.linalg.cond(m.A))                                
                        data['success'].append(False)                                

                    except (np.linalg.LinAlgError) as e:
                        # --------------- PLOT --------------- 
                        m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'],save=save_name,file_extension='png')

                        # --------------- STORE VALUES --------------- 
                        data['Rin'].append(rin)                
                        data['Rload'].append(rload)                
                        data['u_mode'].append(u_mode)                
                        data['that'].append(that)                
                        data['phi'].append(phi)                
                        data['error_elm'].append(np.nan)                
                        data['error_mat'].append(np.nan)                
                        data['condition'].append(np.linalg.cond(m.A))                                
                        data['success'].append(False)                                

df = pd.DataFrame(data)
df.to_excel('img/data.xlsx',header=True,index=False)


#%%
m = SITOBBDS()
# Load model
print('Available models:\n',m.models)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

u, uk = m.create_input(t1, t2, dt,mode='sin')

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

# Filter the data with the Kalman Filter
xhat, yhat,eps,R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)



# Plot simulation against the filtered values
m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'])

# asdf 

#%% ESTIMATION OF ENTIRE A
that0 = -I(m.A.shape[0])*0
that0 = m.A
thetahat_, thetahat0_ = m.ML_opt(Ad,Bd,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,V_theta0=1,thetahat0= that0)

#%% ESTIMATIONS OF ELEMENTS
thetahat, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)

#%%
Ad_mat,*_ = m.discretize(thetahat_.x, B, C, D, dt)
Ad_elm,*_ = m.discretize(thetahat, B, C, D, dt)

# Simulate the system
_, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
_, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)

#%%
m.plot_simulations(t, [y,yhat,yhat_mat,yhat_elm],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$','$\\hat{y}_{MLE_{ELM}}$'],save='')
m.plot_simulations(t, [y,yhat,yhat_mat],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$'],save='')

#%% PRINT ESTIMATION SUMMARY
m.plot_estimation_summary(thetahat_elm, np.zeros((3,3)))
m.plot_estimation_summary(thetahat_mat, np.zeros((3,3)))
m.plot_eigenvalues([m.A_d,thetahat_elm,thetahat_mat],['Ground truth','MLE - Element-wise','MLE - Matrix-wise'])

#%%
asdf 

#%% PLOT COVARIANCES IN TIME
m.plot_covariance_in_time(R)
















