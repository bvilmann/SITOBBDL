#%% ================ Initialize packages and general settings ================
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
# from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os

#%%

I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02
# t2 = 0.005
# t2 = 0.15
t1 = 0
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
Nx = np.array([1e-7,1e-4,1e-4])*0
r123 = np.array([1,1,1])*1e-4

R0=np.diag(r123)*1e5
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1

# ================================== Modelling
m = SITOBBDS()
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
# m.get_model('dummy1',discretize=True,dt=10e-6,params=params,pu=True)

# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# --------------- GET GROUND TRUTH --------------- 5
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

# --------------- ESTIMATE ---------------
# LS - LEAST SQUARES
thetahat = m.LS(xhat,yhat,dt,u=uk)
# thetahat = m.LS(xhat,yhat,dt)

#%%
# KDE - KERNEL DENSITY ESTIMATION
KDE = m.kernel_density_estimate(thetahat)

#%%
# MLE - MAXIMUM LIKELIHOOD ESTIMATION
thetahat_, thetahat0_,A_hat_m_ = m.ML_opt(A,B,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= 0*np.diag(np.ones(3)),opt_in='m')
# thetahat_mat = thetahat_.x

#%% q
thetahat_elm_q, thetahat0, A_hat_q = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='q')
# thetahat_elm, thetahat0, A_hat_m = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='m')
# thetahat_elm_z, thetahat0, A_hat_z = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='z')
