# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:20:46 2023

@author: bvilm
"""
import sys
sys.path.insert(0, '../')  # add parent folder to system path

import numpy as np
import matplotlib.pyplot as plt
from datareader import DataReader
from solver import SITOBBDS, ParamWrapper
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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


#%%
opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Create Noise input
Sx = m.create_noise(t1, t2, dt,amp=.1,dim=3,seed=1234)         
Sy = m.create_noise(t1, t2, dt,amp=0.05,dim=3,seed=1235)        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# --------------- GET GROUND TRUTH --------------- 5
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

# Define thetahat0
thetahat0 = np.array(list(m.p.params.values()))

#%%
from numpy.linalg import det, inv
def FastFiniteDifferenceDerivatives(fun, x, epsilon, *funargs, order=2):
    # TODO: Implement list of epsilons for parameter sensitivity selectibility
    
    # Evaluate function
    f = fun(x, *funargs)
    
    # Dimensions
    nx = x.size
    nf = f.size
    
    ## First order derivatives
    dfFD = np.zeros((nf, nx))
    for j in range(nx):
        # Pertubation
        x[j] = x[j] + epsilon
        
        # Perturbed function evaluation
        fp = fun(x, *funargs)
        
        # Approximation
        dfFD[:, j] = (fp.ravel() - f.ravel()) / epsilon
        
        # Remove pertubation
        x[j] = x[j] - epsilon
    
    ## Second order derivatives
    if order == 2:
        epssq = epsilon**2
        d2fFD = np.zeros((nx, nx, nf))
        for j in range(nx):
            # Pertubation
            x[j] = x[j] + 2*epsilon
                
            # Perturbed function evaluation
            fpp = fun(x, *funargs)
                
            # Pertubation
            x[j] = x[j] - epsilon
    
            # Perturbed function evaluation
            fpz = fun(x, *funargs)
            
            # Approximation
            d2fFD[j, j, :] = (fpp.ravel() - 2*fpz.ravel() + f.ravel()) / epssq
            
            # Reset pertubation
            x[j] = x[j] - epsilon
            
            for k in range(j):
                # Pertubation
                x[j] = x[j] + epsilon
                x[k] = x[k] + epsilon
                
                # Perturbed function evaluation
                fpp = fun(x, *funargs)
                
                # Reset pertubation
                x[k] = x[k] - epsilon
                 
                # Perturbed function evaluation
                fpz = fun(x, *funargs)
                
                # Pertubation
                x[k] = x[k] + epsilon
                x[j] = x[j] - epsilon
                 
                # Perturbed function evaluation
                fzp = fun(x, *funargs)
                
                # Approximation
                d2fFD[k, j, :] = d2fFD[j, k, :] = (fpp.ravel() - fpz.ravel() - fzp.ravel() + f.ravel()) / epssq
                
                # Reset pertubation
                x[k] = x[k] - epsilon
    if order == 2:
        return dfFD, d2fFD
    else:
        return dfFD

def ML_f(x, x0, u, y, R0, R1, R2, t1, t2, dt, thetahat0, m, log):
    # Base case
    params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}

    # Add perturbation
    for i, (k,v) in enumerate(m.p.params.items()):
        params.update({k:x[i]})

    # Get model
    p = ParamWrapper(params,'c1_s0_o3_load',pu=True)
    if log:
        A,B,C,D = m.load_model('c1_s0_o3_load',p=p,log='all')
    else:
        A,B,C,D = m.load_model('c1_s0_o3_load',p=p)
    Ad, Bd, _, _ = m.c2d(A, B, C, D, dt)

    # Get covariance and prediction error
    _,_, eps,R = m.KalmanFilter(Ad,Bd,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,optimization_routine=True)

    # Evaluate the cost function from the batch sum
    J = 0
    for k in range(eps.shape[1]):
        J += 1/2*(np.log(det(R[:,:,k])) \
                  + np.log(2*np.pi)\
                  + eps[:,k].T @ inv(R[:,:,k]) @ eps[:,k]) 

    print(J)

    return J


#%%
# Perturbation magnitude for fastFiniteDifferenceDeviation
epsilon = 1e-4

log=True
funargs = (x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0, m, log)

dfFD, H = FastFiniteDifferenceDerivatives(ML_f, np.log(thetahat0), epsilon, *funargs, order=2) # 

#%%
n = len(dfFD.T)
print(pd.DataFrame(H.reshape((n,n))))

H.reshape((n,n))

#%%
Sigma = H_inv = inv(H.reshape((6,6)))
print(Sigma)

keys = list(m.p.params.keys())

fig, ax = plt.subplots(1,1,dpi=200)
im = ax.imshow(abs(Sigma)) # cmap='rainbow'

cbar = ax.figure.colorbar(im)
ax.set(xticklabels=keys,yticklabels=keys,xticks=range(6),yticks=range(6))

ax.set_title("Fisher's Information Matrix ($\\varepsilon=$"+str(epsilon)+")")



