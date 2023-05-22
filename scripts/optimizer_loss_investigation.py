# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:20:46 2023

@author: bvilm
"""
import sys
sys.path.insert(0, '../')  # add parent folder to system path

import numpy as np
from numpy.linalg import det, inv
import matplotlib.pyplot as plt
from datareader import DataReader
from sysid_app import SITOBBDS, ParamWrapper
from PowerSystem import PS
from scipy.optimize import minimize
import pandas as pd
import os
import datetime
import copy 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def ML_f(theta, x0, u, y, R0, R1, R2, t1, t2, dt, m):
    # Base case
    params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}

    # Add perturbation
    for i, (k,v) in enumerate(m.p.params.items()):
        params.update({k:theta[i]})
        
    # Get model
    p = ParamWrapper(params,'c1_s0_o3_load',pu=True)
    A,B,C,D = m.load_model('c1_s0_o3_load',p=p,log=True)
    Ad, Bd, _, _ = m.c2d(A, B, C, D, dt)

    # Get covariance and prediction error
    _,_, eps,R = m.KalmanFilter(Ad, Bd, C, D,x0, u, y, R0, R1, R2, t1, t2, dt,optimization_routine=True)

    # Evaluate the cost function from the batch sum
    J = 0
    for k in range(eps.shape[1]):
        J += 1/2*(np.log(det(R[:,:,k])) \
                  + np.log(2*np.pi)\
                  + eps[:,k].T @ inv(R[:,:,k]) @ eps[:,k]) 

    # print(J)

    return J

I = lambda n: np.eye(n)


#%%
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
R2=np.diag(r123)*1e2

# Perturbation magnitude for fastFiniteDifferenceDeviation
epsilon = 1e-5

#%%
opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Create Noise input
Sx = m.create_noise(t1, t2, dt,amp=0.01,dim=3,seed=1234)         
Sy = m.create_noise(t1, t2, dt,amp=0.01,dim=3,seed=1235)        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# --------------- GET GROUND TRUTH --------------- 
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

#%%
N = len(m.p.params.values())
M = np.ndarray((6,6),dtype=object)
keys = np.array(list(m.p.params.keys()))
vals = np.array(list(m.p.params.values()))
eps = 0.25
n = 7

for i, (k1,v1) in enumerate(m.p.params.items()):
    for j, (k2,v2) in enumerate(m.p.params.items()):
        print(i,j)
        if k1 == k2:
            mat = np.zeros(n)
            rng1 = np.linspace(v1*(1-eps),v1*(1+eps),n)
            for k,rv1 in enumerate(rng1):
                theta = copy.deepcopy(vals)
                theta[i] = rv1
                mat[k] = ML_f(theta, x0, uk, y, R0, R1, R2, t1, t2, dt, m)
            M[i,j] = mat                        
        else:
            mat = np.zeros((n,n))
            rng1 = np.linspace(v1*(1-eps),v1*(1+eps),n)
            rng2 = np.linspace(v2*(1-eps),v2*(1+eps),n)
            for k,rv1 in enumerate(rng1):
                for l,rv2 in enumerate(rng2):
                    theta = copy.deepcopy(vals)
                    theta[i] = rv1
                    theta[j] = rv2
                    mat[k,l] = ML_f(theta, x0, uk, y, R0, R1, R2, t1, t2, dt, m)
            M[i,j] = mat
            
#%%
fig, ax = plt.subplots(N,N,dpi= 150,figsize=(18,18))

for i, (k1,v1) in enumerate(m.p.params.items()):
    for j, (k2,v2) in enumerate(m.p.params.items()):
        if k1 == k2:
            ax[i,j].plot(np.linspace(vals[i]*(1-eps),vals[i]*(1+eps),n),M[i,j])
        else:
            ax[i,j].imshow(M[i,j],extent=(vals[i]*(1-eps),vals[i]*(1+eps),vals[j]*(1+eps),vals[j]*(1-eps)), aspect=(vals[i]*(1+eps)-vals[i]*(1-eps))/(vals[j]*(1+eps)-vals[j]*(1-eps)))
            ax[i,j].scatter(vals[i],vals[j],color='red',marker='x')
        

        # Formatting
        if j == 0:
            ax[i,j].set_ylabel(keys[i])
        if i == N-1:
            ax[i,j].set_xlabel(keys[j])

#%%
# plt.pcolormesh(np.random.randn(5,5))


