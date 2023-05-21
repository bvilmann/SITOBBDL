# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:02:40 2023

@author: bvilm
"""
import sys
sys.path.insert(0, '../')  # add parent folder to system path


import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
# from solver import SITOBBDS
from sysid_app import SITOBBDS
# from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)

I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -1e-4
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

# Initial conditions
x0 = np.zeros(3)

# Covariance
r123 = np.array([1,1,1])*1e-4
R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1


thetahat0 = 1e-4

# Initial parameters
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}

# Optimization options
# opts = {'gradient':'2-point','method':'BFGS','disp':True}
# opts = {'gradient':'2-point','method':'Newton-CG','gradient':H_inv}
opts = {'gradient':'2-point','method':'BFGS'}
opts = {'method':'SLSQP',
        # 'jac':np.diag(H_inv),
        # 'hessp':H_inv
        }


#%%
log=True
epsilons = [float(f'{j}e-{i}') for i in range(5,9) for j in range(1,2)]

epsilons.sort()

print(epsilons)
opt_params  = ['R','Rin','Rload']
opt_params  = ['R','Rin','Rload','L','C']
n = len(opt_params)

errs = np.zeros((n,len(epsilons)))
d1_fd = np.zeros((n,len(epsilons)))
d2_fd = np.zeros((n,n,len(epsilons)))

#%%
for i, eps in enumerate(epsilons):
    print(round((i+1)/len(epsilons),2),eps)

    # opts = {'method':'SLSQP',
    #         'disp':True,
    #         'jac':'2-point',
    #         # 'hess':'2-point',
    #         'epsilon':eps,
    #         'cnstr_lb_factor':0.5,
    #         'cnstr_ub_factor':1.5,
    #         }
    opts = {'jac':'2-point',
            'epsilon':eps,
            'method':'BFGS'}


    m = SITOBBDS(opts=opts)
    
    m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
        
    # Create input
    u, uk = m.create_input(t1, t2, dt,mode='sin')        
    Sx = m.create_noise(t1, t2, dt,amp=.01,dim=3,seed=1234)*1        
    # Sx = None         
    
    # Get matrices
    Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
    
    #%% --------------- GET GROUND TRUTH --------------- 
    # Simulate the system
    x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)
    
    m.plot_simulations(t, [y],labels=['$y$'])
    
    # Identification of the parameter space
    ests, thetahat, res, A_hat = m.ML_opt_param(opt_params,A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=thetahat0,log=True)

    errs[:,i] = ests - np.array([m.p.params[k] for k in opt_params])
    d1_fd[:,i] = thetahat.jac
    d2_fd[:,:,i] = thetahat.hess_inv

#%%

fig, ax = plt.subplots(1,3,dpi=200,figsize=(12,8),sharex=True)
for i in range(3):
    ax[i].set(xscale='log',yscale='log')
# ax[1].set(xscale='log',yscale='log')

# 1st order derivative
for i in range(n):
    ax[0].plot(epsilons,d1_fd[i,:],label='$\\Del_1$'+f'[{i+1}]')

# 2nd order derivative
for i in range(n):
    for j in range(n):
        if i >= j:
            ax[1].plot(epsilons,d2_fd[i,j,:],label='$H^{-1}$'+f'[{i+1},{j+1}]')

#%%
# Resulting
for i in range(n):
    ax[2].plot(epsilons,d2_fd[i,j,:0],label='$H^{-1}$'+f'[{i+1},{j+1}]')

    for j in range(n):
        if i >= j:
            ax[2].plot(epsilons,d2_fd[i,j,:0],label='$H^{-1}$'+f'[{i+1},{j+1}]')


#%%
fig, ax = plt.subplots(n,1,dpi=200,figsize=(12,12))
path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

for i in range(n):
    ax[i].bar([str(e) for e in epsilons],errs[i,:],label=opt_params[i],lw=1)
    # ax[i].axhline(res.loc[list(m.p.params.keys())[i]]['System']-res.loc[list(m.p.params.keys())[i]]['Lower bound'],lw=0.75,color='k')
    ax[i].axhline(0,lw=0.75,color='red')
    # ax[i].axhline(res.loc[list(m.p.params.keys())[i]]['System']-res.loc[list(m.p.params.keys())[i]]['Upper bound'],lw=0.75,color='k')
    ax[i].set(yscale='symlog',ylabel=list(m.p.params.keys())[i])
    # ax[i].set(xscale='log',yscale='log')
# ax.grid()
# ax.legend(ncol=3,loc='upper right',fontsize=5.5)

# plt.savefig(f'{path}\\tol_ind_test_eps_clean_.pdf')

