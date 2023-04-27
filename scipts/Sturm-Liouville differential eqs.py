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
r123 = np.array([1,1,1])*1e-4

R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1


# 
thetahat0 = 1

#%%
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}

m = SITOBBDS()
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

d = {k:[] for k in m.p.params.keys()}
d['solver'] = []
d['jac'] = []
d['err_norm'] = []
d['dt'] = []

opts = {'gradient':'2-point','method':'SLSQP'}
m = SITOBBDS(opts=opts)
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=3,seed=1234)         

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)

#%%
thetahat, thetahat0, Ahat = m.ML_opt_param(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=thetahat0,opt_in='multi',log=True)





