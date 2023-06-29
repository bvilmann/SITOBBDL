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
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
n = 3
I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -1e-4
t2 = 0.04
t = np.arange(t1,t2+dt,dt)

# Initial conditions
x0 = np.zeros(n)
# x0 = np.array([-0.0104619 , -0.00789252, -0.00771146])

# Covariance
r123 = np.ones(n)*1e-4
R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1

thetahat0 = 1e-12

# Initial parameters

# Optimization options
# opts = {'gradient':'2-point','method':'BFGS','disp':True}
# opts = {'gradient':'2-point','method':'Newton-CG','gradient':H_inv}
opts = {'gradient':'2-point','method':'BFGS'}
opts = {'method':'SLSQP',
        # 'jac':np.diag(H_inv),
        # 'hessp':H_inv
        }

opts = {'method':'Newton-CG',
        'jac':lambda *x: dfFD,
        'hess':lambda x, *args: H_inv
        }

opts = {'method':'SLSQP',
        'disp':True,
        }

opts = {'method':'SLSQP',
        'disp':True,
        # 'jac':'2-point',
        # 'hess':'2-point',
        'epsilon':1e-7,
        'cnstr_lb_factor':0.5,
        'cnstr_ub_factor':1.5,
        }

opts = {'jac':'2-point',
        # 'epsilon':1e-5,
        # 'method':'L-BFGS-B'
        'method':'BFGS'
        # 'method':'BFGS'
        }

# # opts = {'method':'BFGS'}
# opts = {'method':'BFGS',
#         'epsilon':1e-6,
#         # 'jac':'2-point',
#         }



#%%
model = f'c1_s0_o3_load' # f'c1_s0_o3_load'
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4*0}
m = SITOBBDS(opts=opts)
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.001,dim=n,seed=1234)*0
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)*1
# Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
# Sx = None         

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

m.plot_simulations(t, [y],labels=['$y$'])




#%%
# Identification of the parameter space
opt_params  = ['R','Rin']
# opt_params  = ['Rload']
# opt_params  = ['Rin','Rload','R','L','C']
# opt_params  = ['Rin','Rload','R','L','C','G']
# opt_params  = ['R','L','C','G']
# opt_params  = ['R','L']
# opt_params  = ['Rload']
# opt_params  = ['R','Rload','L','C']
# opt_params  = ['L','C']
# opt_params  = ['Rload']
# thetahat0 = [m.p.params[k] for k in opt_params]
thetahat0 = [abs(np.random.randn()*1e-4) for k in opt_params]
# thetahat0 = None

ests, thetahat, res, A_hat = m.ML_opt_param(opt_params,A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=thetahat0,log=True)

name = "_".join(opt_params)
df = pd.DataFrame({name:ests},index=opt_params)

w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
df.to_excel(f'{w_path}\\MLE_1c_all_params_{name}.xlsx',header=True,index=True)

#%%
import seaborn as sns
fig, ax = plt.subplots(1,1,dpi=150)
plt.imshow(thetahat.hess_inv)
# plt.figure(figsize=(10,7))
sns.heatmap(thetahat.hess_inv, annot=True, fmt=".2f", cmap='viridis', cbar=True, annot_kws={"color":'white'}, xticklabels=opt_params, yticklabels=opt_params)

# ax.set(xticklabels=opt_params,yticklabels=opt_params,xticks=[i for i in range(len(opt_params))],yticks=[i for i in range(len(opt_params))])
fig.tight_layout()
# plt.show()

w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
plt.savefig(f'{w_path}\\MLE_COV_{opt_params}.pdf')

#%%
params.update(dict(res['Estimated']*m.p.Zbase))

m = SITOBBDS()

m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# Simulate the system
_, y1 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

m.plot_simulations(t, [y,y1],labels=['$y$','$y1$'])




