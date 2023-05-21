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
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

# Initial conditions
x0 = np.zeros(n)

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
        'epsilon':1e-5,
        'method':'BFGS'}
# # opts = {'method':'BFGS'}
# opts = {'method':'BFGS',
#         'epsilon':1e-6,
#         # 'jac':'2-point',
#         }

#%%
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}
m = SITOBBDS(opts=opts)
m.get_model(f'c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)
# Sx = None         

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)

m.plot_simulations(t, [y],labels=['$y$'])

