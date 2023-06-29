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

dt = 10e-6
# dt = 10e-5
ys = {}
xs = {}
dts = np.arange(10e-7,10e-5,10e-6)

cmap = plt.get_cmap('viridis')  # 'viridis' is one of the predefined colormaps
fig, ax=plt.subplots(3,1,sharex=True,dpi=150)

# for dt in dts:
t1 = -1e-4
t2 = 0.04
t = np.arange(t1,t2+dt,dt)

# Initial conditions
x0 = np.zeros(n)
# x0 = np.array([-0.0104619 , -0.00789252, -0.00771146])

# Covariance
r123 = np.ones(n)*1e-4
R0=np.diag(r123)*1e1
R1=np.diag(r123)*1e1
R2=np.diag(r123)*1e3

thetahat0 = 1e-12

model = f'c1_s0_o3_load' # f'c1_s0_o3_load'
params= {'Rin':0.05,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

opts = {'jac':'2-point',
        # 'epsilon':1e-5,
        # 'method':'L-BFGS-B'
        'method':'BFGS'
        # 'method':'BFGS'
        }

#%% LOAD PSCAD DATA
# model = f'c1_s0_o3_load' # f'c1_s0_o3_load'
# params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4*0}
# m = SITOBBDS(opts=opts)
# m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True)
    
# dr = DataReader()

# t_pscad, v1,i2,v2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_1c\Cable_1phase.infx',t1=t1,t2=t2)

# v1 *= 1000/m.p.Vbase/np.sqrt(2)
# i2 *= 1000/m.p.Ibase/np.sqrt(2)/np.sqrt(3)
# v2 *= 1000/m.p.Vbase/np.sqrt(2)

# t_pscad, V1,I2,V2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_1c\Cable_1phase.infx',series='b',t1=t1,t2=t2)

# V1 *= 1000/m.p.Vbase/np.sqrt(2)
# I2 *= 1000/m.p.Ibase/np.sqrt(2)/np.sqrt(3)
# V2 *= 1000/m.p.Vbase/np.sqrt(2)

# y_pscad_fdcm = np.vstack([v1,i2,v2])
# y_pscad_pi = np.vstack([V1,I2,V2])

#%%
m = SITOBBDS(opts=opts)
m.get_model(model,discretize=True,dt=dt,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.001,dim=n,seed=1234)*0
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)*0
# Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
# Sx = None         

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

for i in range(3):
    ax[i].plot(x[i,:],y[i,:]) # , color=cmap(j / len(dts)


ys[dt]=y
xs[dt]=x


