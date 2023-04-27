# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:02:40 2023

@author: bvilm
"""
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime


I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02

t1 = 0
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
r123 = np.array([1,1,1])

R0=np.diag(r123)*1e-1
R1=np.diag(r123)*1e-1
R2=np.diag(r123)*1e-1

#%%

opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
# m.get_model('Cable_2',discretize=True,dt=10e-6,params=params,pu=True)
# m.get_model('dummy1',discretize=True,dt=10e-6,params=params,pu=True)


# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Create Noise input
Sx = m.create_noise(t1, t2, dt,amp=0.02,dim=3,seed=1234)         
Sy = m.create_noise(t1, t2, dt,amp=0.02,dim=3,seed=1235)        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# --------------- GET GROUND TRUTH --------------- 5
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'])

#%% CHECKING IF SIMULATING WITHOUT NOISE OR ACTUALLY TRACKING

opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Create Noise input
Sx = m.create_noise(t1, t2, dt,amp=.1,dim=3,seed=1234)         
Sy = m.create_noise(t1, t2, dt,amp=0.05,dim=3,seed=1235)        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

# Change the system
params= {'Rin':0.25,'R':100,'V':1,'Vbase':66e3,'Rload': 250,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'])


