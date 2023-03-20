#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import Solver
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


# ax[-1].legend()
# ax[0].set_ylim(-.035,0.035)
ax[0].set_ylim(-110,110)
# ax[1].set_ylim(-100,100)
# ax[2].set_ylim(-100,100)
# ax[-1].set_ylim(-1.05*V,1.05*V)
ax[-1].set_xlim(0.095,0.13)
# ax[-1].set_xlim(0.1285,0.1286)

#%% Stat plots
from matplotlib.colors import LogNorm
stats1[0]

fig,ax = plt.subplots(3,2,dpi=300)

for i in range(3):
    ax[i,0].imshow(eval(f'stats{i+1}')[0], norm=LogNorm())
    ax[i,1].imshow(eval(f'stats{i+1}')[1], norm=LogNorm())

#%% PLOTTING RESIDUALS
fig,ax = plt.subplots(3,1,sharex=True,dpi=300)
labs = ['$I_L$ [kA]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']

# labs = ['$I_L$ [A]','$V_{c2}$ [kV]']
cnt = 0
for i in range(3):
    print(i)
    # ax[i].plot(t_pscad,[iabc2,vabc1,vabc2][i],label='$Y_{PSCAD}$')    
    # ax[i].plot(t_pscad,y_hat3.T[i]/1000,label='$\\hat{Y}_{PSCAD}$',alpha = 0.5)    
    ax[i].plot(res_1.t,res_1.y[i]/1000 - y_hat1.T[i]/1000,label='$\\varepsilon_{\\varrho^3}$')    
    if i in [0,2]:
        ax[i].plot(res_1.t,res_2.y[cnt]/1000 - y_hat2.T[cnt]/1000,label='$\\varepsilon_{\\varrho^2}$')    
        cnt += 1
    ax[i].grid()
    ax[i].set_ylabel(labs[i])
    ax[i].legend()

ax[-1].legend()
ax[-1].set_xlim(0.095,0.12999)
            
            


#%%
# https://scipy-lectures.org/packages/sympy.html
import sympy as sym

def jacobian(ode:list,var:list,mode='numeric'):
    if len(ode) != len(var):
        raise AttributeError('len(ode) != len(var)')
    if mode == 'numeric':
        M = np.zeros((len(ode),len(var)))        
    elif mode == 'symbolic':
        M = np.empty((len(ode),len(var)), dtype=object)
    else:
        raise KeyError('Mode must be "numeric" or "symbolic"')
        
    print(M.shape)
    for i, o in enumerate(ode):
        for j, v in enumerate(var):
            print(i,j,o,v)
            sol = sym.diff(o,v)
            if sol == np.nan:
                raise ValueError('Could not find solution')
            print(str(sol))
            if mode == 'numeric':
                str(sol).replace('')
                M[i,j] = float()
            elif mode == 'symbolic':
                M[i,j] = sol            
    return M

def init_vars(var):
    return [sym.Symbol(v) for v in var]
    
x, y = init_vars(['x','y'])
odes = [1/2*y-x**1,x**2+y]
var = [x,y]
M = jacobian(odes,var,mode='symbolic')
print(M)

#%%
i,v = init_vars(['i','v'])
odes = [y*v,z*i]
var = [i,v]
A = jacobian(odes,var,mode='numerical')
print(A)

# A = 




