#%% ================ Initialize packages and general settings ================
import sys
sys.path.insert(0, '../')  # add parent folder to system path
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
# from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
from Plots import grouped_bar

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

# ================================== Modelling ================================== 
opts = {'method':'BFGS',
        'jac':'2-point',
        'epsilon':1e-8,
        }
m = SITOBBDS(opts = opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4*0}
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
thetahat_, thetahat0_,A_hat_m_ = m.ML_opt_param(A,B,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= 1e-12,opt_in='single')

#%% PLOTTING


#%%
# ================= PLOT =================
data = {'True':m.p.params.values(),'Individual':thetahat_}
df = pd.DataFrame(data ,index=list(m.p.params.keys()))

fig, ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
ax = grouped_bar(ax,df,ax_kwargs=dict(yscale='log'),legend_kwargs=dict(loc='upper left'))

w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
df.to_excel(f'{w_path}\\MLE_1c_individual_params.xlsx',header=True,index=True)

