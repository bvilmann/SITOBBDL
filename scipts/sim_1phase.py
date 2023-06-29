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
        
        k = ['Sim.','PSCAD FDCM','Est.','PSCAD PI','PSCAD'].index(source)
        
        ax[i].plot(df[col].dropna(),color=clrs[k],zorder=3,label=source)    

    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right')


    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_xlim(df.index.min(),df.index.max())

    return fig, ax 

def plot_residuals_1phase(df,source1,source2,ax=None):
    # Fill NA with the previous value in each column
    df.fillna(method='ffill', inplace=True)
    
    # Fill remaining NA (those that were in the first position) with zero
    df.fillna(0, inplace=True)

    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)
        
    for j, meas in enumerate(['v1','i2','v2']):            
        y1 = df[f'{meas}_{source1}']
        y2 = df[f'{meas}_{source2}']
        ax[j].plot(df[f'{meas}_{source1}']-df[f'{meas}_{source2}'],label='$\\varepsilon$')
            
    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right')

    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_xlim(df.index.min(),df.index.max())

    return fig, ax 

def load_3ph_system(i):
    amp = [[1,1,1],[1,1,0],[0.5,1.05,0.75]][i]
    phi = [[0,-2/3*np.pi,2/3*np.pi],[0,-2/3*np.pi,2/3*np.pi],[0,-3/2*np.pi,2/3*np.pi]][i]
    return amp, phi

#%%


pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
n = 3
I = lambda n: np.eye(n)

dt = 10e-6

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
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

opts = {'jac':'2-point',
        # 'epsilon':1e-5,
        # 'method':'L-BFGS-B'
        'method':'BFGS'
        # 'method':'BFGS'
        }


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

# m.plot_simulations(t[3:], [y[:,3:]],labels=['$y$'])
# m.plot_simulations(t[1:], [y[:,1:],y_pscad_fdcm,y_pscad_pi],labels=['$y$','$y_{PSCAD FDCM}$','$y_{PSCAD PI}$'])


#%% LOAD PSCAD DATA

dr = DataReader()

t_pscad, v1,i2,v2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_1c\Cable_1phase.infx',t1=t1,t2=t2)

v1 *= 1000/m.p.Vbase/np.sqrt(2)
i2 *= 1000/m.p.Ibase/np.sqrt(2)/np.sqrt(3)
v2 *= 1000/m.p.Vbase/np.sqrt(2)

t_pscad, V1,I2,V2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_1c\Cable_1phase.infx',series='b',t1=t1,t2=t2)

V1 *= 1000/m.p.Vbase/np.sqrt(2)
I2 *= 1000/m.p.Ibase/np.sqrt(2)/np.sqrt(3)
V2 *= 1000/m.p.Vbase/np.sqrt(2)

y_pscad_fdcm = np.vstack([v1,i2,v2])
y_pscad_pi = np.vstack([V1,I2,V2])

#%%

w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

df = obtain_results_1phase(t, y[0,:], 'v1', 'Sim.')
df = obtain_results_1phase(t, y[1,:], 'i2', 'Sim.', df=df)
df = obtain_results_1phase(t, y[2,:], 'v2', 'Sim.', df=df)
df = obtain_results_1phase(t_pscad, v1[0,:], 'v1', 'PSCAD FDCM', df=df)
df = obtain_results_1phase(t_pscad, i2[0,:], 'i2', 'PSCAD FDCM', df=df)
df = obtain_results_1phase(t_pscad, v2[0,:], 'v2', 'PSCAD FDCM', df=df)
df = obtain_results_1phase(t_pscad, V1[0,:], 'v1', 'PSCAD PI', df=df)
df = obtain_results_1phase(t_pscad, I2[0,:], 'i2', 'PSCAD PI', df=df)
df = obtain_results_1phase(t_pscad, V2[0,:], 'v2', 'PSCAD PI', df=df)

plot_simulations_1phase(df)

plt.savefig(f'{w_path}\\1phase_results_tds_primitive_{t2}.pdf')

plot_residuals_1phase(df,source1='Sim.',source2='PSCAD PI')
plt.savefig(f'{w_path}\\1phase_results_tds_primitive_res_{t2}.pdf')


#%%

x_hat_pred, y_hat_pred, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0, uk, y, R0, R1, R2, t1, t2, dt)
# m.plot_simulations(t, [y,y_hat_pred],labels=['$y$','$\\hat{y}$'])



