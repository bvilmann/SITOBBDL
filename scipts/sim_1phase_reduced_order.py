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
import Plots
# from solver import SITOBBDS
from sysid_app import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime
from scipy.stats import norm

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']


def df_to_latex(df, filename, **kwargs):
    """
    This function converts a dataframe into a LaTeX table.
    The resulting LaTeX table is saved into a .tex file.

    Parameters:
    - df (pandas.DataFrame): The dataframe to convert.
    - filename (str): The name of the file (without extension) where to save the table.
    - index (bool): Whether to include the dataframe's index into the table.
    - header (bool): Whether to include the dataframe's header into the table.
    """
    
    df.style.to_latex(filename + '.tex',position = 'H',**kwargs)
    return

def merge_df(t,y,header:str,df):
    print(t.shape,y.shape)
    df2 = pd.DataFrame({'t':list(t),header: list(y)}).set_index('t')
    
    df = df.merge(df2, right_index=True, left_index=True, how='outer')   
    
    return df

def obtain_results_1phase(t,y,measurement:str,source:str,df=None):
    # Make sure of rotation of measurements
    
    header = f'{measurement}_{source}'
    if df is None:
        df = pd.DataFrame({'t': t.round(7),header: y}).set_index('t')

    else:
        df = merge_df(t.round(7),y,header,df)
        
    return df


def plot_simulations_1phase(df):
    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)

    for col in df.columns:
        meas, source = col.split('_')
        i = ['v1','i2','v2'].index(meas)
               
        k = ['Simulation (order 3)','Simulation (order 2)','Estimated parameters'].index(source)       
        ax[i].plot(df[col].dropna(),color=clrs[k],zorder=3,label=source)    
            
    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right',fontsize=6)

    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_xlim(df.index.min(),df.index.max())
    
    return fig, ax 

def plot_residuals_1phase_kf(df,sx,sy,ax=None):
    # Fill NA with the previous value in each column
    df.fillna(method='ffill', inplace=True)
    
    # Fill remaining NA (those that were in the first position) with zero
    df.fillna(0, inplace=True)

    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)
        
    for j, meas in enumerate(['v1','i2','v2']):            
        cnt = 0
        for Rx in [1e-3,1e3]:
            for Ry in [1e-3,1e3]:
                eps = df[f'{meas}_Est._{Rx}_{Ry}']-df[f'{meas}_Sim._{sx}_{sy}']
                y, x,_ = ax[j].hist(eps,bins=100,alpha=0.25,label=('','$\\varepsilon_{'+f'(Rx={Rx},Ry={Ry})'+'}$')[j==0],color=clrs[cnt],density=True)
                ax[j].plot(x[1:]-(x[1]-x[0])/2,y,color=clrs[cnt],alpha=0.5)

                # calculate the corresponding probability density function (pdf) values
                y_theory = norm.pdf(x, eps.mean(), eps.std())
                
                # plot the theoretical normal distribution
                ax[j].plot(x, y_theory,color=clrs[cnt],ls='--',zorder=5)

                cnt+=1
            
    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['Error: $V_k$\n[density]','Error: $I_{km}$\n[density]','Error: $V_m$\n[density]'][i])
        if i ==0:
            ax[i].legend(loc='upper right',fontsize=7)

    # ax[-1].set_xlabel('Time [s]')
    # ax[-1].set_xlim(df.index.min(),df.index.max())

    return fig, ax 

def load_3ph_system(i):
    amp = [[1,1,1],[1,1,0],[0.5,1.05,0.75]][i]
    phi = [[0,-2/3*np.pi,2/3*np.pi],[0,-2/3*np.pi,2/3*np.pi],[0,-3/2*np.pi,2/3*np.pi]][i]
    return amp, phi

def initialize(n,r0=1e-3,r1=1e-3,r2=1e-3):
    # Initial conditions
    x0 = np.zeros(n)
    # x0 = np.array([-0.0104619 , -0.00789252, -0.00771146])

    # Covariance
    r123 = np.ones(n)
    R0=np.diag(r123)*r0
    R1=np.diag(r123)*r1
    R2=np.diag(r123)*r2
    
    return x0, R0, R1, R2
    

#%%


pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
n = 3
I = lambda n: np.eye(n)

dt = 10e-6

t1 = -1e-4
t2 = 0.2
t = np.arange(t1,t2+dt,dt)


thetahat0 = 1e-12

params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

opts = {'jac':'2-point',
        # 'epsilon':1e-5,
        # 'method':'L-BFGS-B'
        'method':'BFGS'
        # 'method':'BFGS'
        }


#%%
x0, R0, R1, R2 = initialize(3)

model = f'c1_s0_o3_load' # f'c1_s0_o3_load'
m = SITOBBDS(opts=opts)
m.get_model(model,discretize=True,dt=dt,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
x, y3 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx*0,Sy=Sy*0)      

#%%
x0, R0, R1, R2 = initialize(2)

model = f'c1_s0_o2_load' # f'c1_s0_o3_load'
m = SITOBBDS(opts=opts)
m.get_model(model,discretize=True,dt=dt,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=2,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=2,seed=1235)

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
x, y2 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx*0,Sy=Sy*0)      
#%%
df = obtain_results_1phase(t, y3[0,:], 'v1', f'Simulation (order 3)')
df = obtain_results_1phase(t, y3[1,:], 'i2', f'Simulation (order 3)', df=df)
df = obtain_results_1phase(t, y3[2,:], 'v2', f'Simulation (order 3)', df=df)
df = obtain_results_1phase(t, uk,      'v1', f'Simulation (order 2)', df=df)
df = obtain_results_1phase(t, y2[0,:], 'i2', f'Simulation (order 2)', df=df)
df = obtain_results_1phase(t, y2[1,:], 'v2', f'Simulation (order 2)', df=df)

plot_simulations_1phase(df)

#%% Identification of the parameter space

# opt_params  = ['Rload','R','L','C','G']
# opt_params  = ['R','L']
# # opt_params  = ['C','L']
# # opt_params  = ['R','L','C','G']

# thetahat0 = [1e-4 for k in opt_params]
# # thetahat0 = None

# ests, thetahat, res, A_hat = m.ML_opt_param(opt_params,A,B,C,D,x0, uk, y2, R0, R1, R2, t1, t2, dt, thetahat0=thetahat0,log=True)

# name = "_".join(opt_params)
# df = pd.DataFrame({name:ests},index=opt_params)

# w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
# df.to_excel(f'{w_path}\\MLE_1c_2o_all_params_{name}.xlsx',header=True,index=True)

#%%
# params.update(dict(res['Estimated']*m.p.Zbase))

# m = SITOBBDS()

# m.get_model(model,discretize=True,dt=dt,params=params,pu=True)
    
# # Create input
# u, uk = m.create_input(t1, t2, dt,mode='sin')        

# # Get matrices
# Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# # Simulate the system
# _, y_restored = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

# #%%
# df = obtain_results_1phase(t, uk,               'v1', f'Estimated parameters', df=df)
# df = obtain_results_1phase(t, y_restored[0,:],  'i2', f'Estimated parameters', df=df)
# df = obtain_results_1phase(t, y_restored[1,:],  'v2', f'Estimated parameters', df=df)

# plot_simulations_1phase(df)


