# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:02:40 2023

@author: bvilm
"""
import sys
sys.path.insert(0, '../')  # add parent folder to system path

#%%
import numpy as np
import matplotlib.pyplot as plt
from datareader import DataReader
# from solver import SITOBBDS, ParamWrapper
from sysid_app import SITOBBDS, ParamWrapper
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime

#%%
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

 
thetahat0 = 1e-12

# Initial parameters
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}


#%% Get normal system
m = SITOBBDS()
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=3,seed=1234)*0         
x0 = np.zeros(3)

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#% --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y1 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)

#%%
m = SITOBBDS()
m.get_model('c1_s0_o2_ZY',discretize=True,dt=10e-6,params=params,pu=True)
x0 = np.zeros(8)

# Create input
u, uk1 = m.create_input(t1, t2, dt,mode='sin')        
u, uk2 = m.create_input(t1, t2, dt,mode='cos')
uk = np.vstack((uk1,uk2*0))
        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=8,seed=1234)*0

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

C = C[[0,2,6],:]

#% --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y2 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)

#%% PLOTTING

# m.plot_simulations(t, [y1,y2],labels=['$y_1$','$y_2$'])
m.plot_simulations(t, [y2],labels=['$y_2$'])

