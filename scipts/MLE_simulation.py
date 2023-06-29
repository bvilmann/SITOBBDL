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

def rms(x:np.ndarray,axis=0):
        
    rms = np.zeros(x.shape[axis])

    for i in range(x.shape[axis]):
        rms[i] = np.sqrt((x[i,:]**2)/len(x[i,:]))

    return rms

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
    df2 = pd.DataFrame({'t': t,header: y,}).set_index('t')
    
    df = df.merge(df2, right_index=True, left_index=True, how='outer')   
    
    return df

def obtain_results(t,y,measurement:str,source:str,df=None):
    # Make sure of rotation of measurements
    if y.shape[0] < y.shape[1]:
        y = y.T
    
    for i in range(y.shape[1]):
        header = f'{measurement}_{["A","B","C"][i]}_{source}'
        if df is None and i == 0:
            df = pd.DataFrame({'t': t.round(7),header: y[:,i]}).set_index('t')

        else:
            df = merge_df(t.round(7),y[:,i],header,df)
        
    return df


def plot_simulations_3phase(df):
    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)

    for col in df.columns:
        meas, phase, source = col.split('_')
        i = ['v1','i2','v2'].index(meas)
        j = ['A','B','C'].index(phase)
        k = ['Sim.','PSCAD FDCM','Est.','PSCAD PI','PSCAD'].index(source)
        
        ax[i].plot(df[col].dropna(),alpha=[1,0.25,0.25][j],color=clrs[k],zorder=3,label=('',source)[phase=='A'])    

    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right')


    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_xlim(df.index.min(),df.index.max())

    return fig, ax 

def plot_residuals_3phase(df,source1,source2,ax=None):
    # Fill NA with the previous value in each column
    df.fillna(method='ffill', inplace=True)
    
    # Fill remaining NA (those that were in the first position) with zero
    df.fillna(0, inplace=True)

    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)
        
    for i, phase in enumerate(['A','B','C']):
        for j, meas in enumerate(['v1','i2','v2']):            
            y1 = df[f'{meas}_{phase}_{source1}']
            y2 = df[f'{meas}_{phase}_{source2}']
            ax[j].plot(df[f'{meas}_{phase}_{source1}']-df[f'{meas}_{phase}_{source2}'],label='$\\varepsilon_'+phase+'$')
            
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
    
def plotA(A):
    
    fig, ax = plt.subplots(1,1,dpi=150)
    
    ax.imshow(np.where(abs(A)>0,abs(A),np.nan))

    return fig, ax

I = lambda n: np.eye(n)



#%% ========== Settings ========== 
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200}

dt = 10e-6
t1 = -1e-4
t2 = 0.04


t = np.arange(t1,t2+dt,dt)

# get input
system = 0
amp, phi = load_3ph_system(system)
model = f'Cable_3c' # f'c1_s0_o3_load'
m = SITOBBDS()
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
n = m.n
x0 = np.zeros(n)
u, uk = m.create_input(t1, t2, dt, mode='abc',amp=amp,phi=phi)        

read_pscad = True

# Initial conditions

# --------- Covariance ---------
r123 = np.ones(n)*1e-4
R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1
thetahat0 = 1e-12

# --------- Solver options ---------
opts = {
        'jac':'2-point',
        # 'epsilon':1e-5,
        'method':'BFGS'
        # 'method':'BFGS'
        }


w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
l_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

#%% 
import re
def filter_matrix(s,reshape=None):

    # Remove brackets
    s = re.sub(r'\[|\]', '', s)
    
    # Split string by spaces
    list_string = s.split()
    
    # Convert each element in the list to a float
    list_float = [float(i) for i in list_string]

    if reshape is not None:
        if isinstance(reshape,int):
            m = np.array(list_float).reshape(reshape,reshape)
        elif isinstance(reshape,tuple):
            m = np.array(list_float).reshape(*reshape)
    else:
        m = np.array(list_float)

    return m

df = pd.read_excel(f'{w_path}\\MLE_assesment.xlsx',header=0)


fig, ax = plt.subplots(3,1,dpi=150)

for i,row in df.iterrows():
    

    pass

#%%
df = pd.read_excel(f'{w_path}\\MLE_assesment_load.xlsx',header=0)

uniq_params = df.params.unique()
fig, ax = plt.subplots(4,1,dpi=150,sharex=True)
for i, idx in enumerate(uniq_params):
    df_temp = df[df.params == idx]
    
    true_val = eval(df_temp.true_params.values[0])[0]
    
    y = [eval(v)[0] for v in df_temp.ests.values]

    x = df_temp.Rload
    
    ax[i].plot(x,y)
    ax[i].set(xscale='log',yscale='log',ylabel='$\\widehat{'+idx[2]+['}$ $[\\Omega]','}$ $[H]','}$ $[S]','}$ $[F]'][i]+'$')
    if i == 3:
        ax[i].set(xlabel='$R_{load}$ [$\\Omega$]')
    ax[i].grid()
    ax[i].axhline(true_val,color='gold',label='True value',lw=2)
    ax[i].axhline(1,color='red',alpha=0.5,ls=':',label='1 (helper line)')
    ax[i].axvline(43.56,color='k',label='$100\%$ loading')
    ax[i].axvline(435.6,color='k',label='$10\%$ loading',alpha=0.5)
    # ax[i].axvline(4356.6,color='k',label='$1\%$ loading',alpha=0.25)

ax[-1].invert_xaxis()
ax[1].legend(loc='upper left',fontsize=8,ncols=2)

fig.tight_layout()
plt.savefig(f'{l_path}\\MLE_rload_sensitivity.pdf')

#%%

df = pd.read_excel(f'{w_path}\\MLE_assesment_load.xlsx',header=0)

uniq_params = df.params.unique()
fig, ax = plt.subplots(4,1,dpi=150,sharex=True)
for i, idx in enumerate(uniq_params):
    df_temp = df[df.params == idx]
    
    true_val = eval(df_temp['thetahat.hess_inv'].values[0])[0][0]
    
    y = [eval(v)[0][0] for v in df_temp['thetahat.hess_inv'].values]

    x = df_temp.Rload
    
    ax[i].plot(x,y)
    ax[i].set(xscale='log',yscale='log',ylabel='$H^{-1}_{\\widehat{'+idx[2]+'}}$')
    if i == 3:
        ax[i].set(xlabel='$R_{load}$ [$\\Omega$]')
    ax[i].grid()
    # ax[i].axhline(true_val,color='gold',label='True value',lw=2)
    ax[i].axhline(1,color='red',alpha=0.5,ls=':',label='1 (helper line)')
    ax[i].axvline(43.56,color='k',label='$100\%$ loading')
    ax[i].axvline(435.6,color='k',label='$10\%$ loading',alpha=0.5)
    # ax[i].axvline(4356.6,color='k',label='$1\%$ loading',alpha=0.25)

ax[-1].invert_xaxis()
ax[1].legend(loc='upper left',fontsize=8,ncols=3)

fig.tight_layout()
plt.savefig(f'{l_path}\\MLE_rload_sensitivity_hess_inv.pdf')

#%%
df = pd.read_excel(f'{w_path}\\MLE_assesment_load.xlsx',header=0)

uniq_params = df.params.unique()
fig, ax = plt.subplots(4,1,dpi=150,sharex=True)
for i, idx in enumerate(uniq_params):
    df_temp = df[df.params == idx]
    
    true_val = eval(df_temp.true_params.values[0])[0]
    
    y = [eval(v)[0] for v in df_temp.ests.values]

    x = df_temp.Rload
    
    ax[i].plot(x,y)
    ax[i].set(xscale='log',yscale='log',ylabel='$\\widehat{'+idx[2]+['}$ $[\\Omega]','}$ $[H]','}$ $[S]','}$ $[F]'][i]+'$')
    if i == 3:
        ax[i].set(xlabel='$R_{load}$ [$\\Omega$]')
    ax[i].grid()
    ax[i].axhline(true_val,color='gold',label='True value',lw=2)
    ax[i].axhline(1,color='red',alpha=0.5,ls=':',label='1 (helper line)')
    ax[i].axvline(43.56,color='k',label='$100\%$ loading')
    ax[i].axvline(435.6,color='k',label='$10\%$ loading',alpha=0.5)
    # ax[i].axvline(4356.6,color='k',label='$1\%$ loading',alpha=0.25)

ax[-1].invert_xaxis()
ax[1].legend(loc='upper left',fontsize=8,ncols=2)

fig.tight_layout()
plt.savefig(f'{l_path}\\MLE_rload_sensitivity.pdf')

#%%

df = pd.read_excel(f'{w_path}\\MLE_assesment_load.xlsx',header=0)

uniq_params = df.params.unique()
fig, ax = plt.subplots(4,1,dpi=150,sharex=True)
for i, idx in enumerate(uniq_params):
    df_temp = df[df.params == idx]
    
    true_val = eval(df_temp['thetahat.hess_inv'].values[0])[0][0]
    
    y = [eval(v)[0][0] for v in df_temp['thetahat.hess_inv'].values]

    x = df_temp.Rload
    
    ax[i].plot(x,y)
    ax[i].set(xscale='log',yscale='log',ylabel='$H^{-1}_{\\widehat{'+idx[2]+'}}$')
    if i == 3:
        ax[i].set(xlabel='$R_{load}$ [$\\Omega$]')
    ax[i].grid()
    # ax[i].axhline(true_val,color='gold',label='True value',lw=2)
    ax[i].axhline(1,color='red',alpha=0.5,ls=':',label='1 (helper line)')
    ax[i].axvline(43.56,color='k',label='$100\%$ loading')
    ax[i].axvline(435.6,color='k',label='$10\%$ loading',alpha=0.5)
    # ax[i].axvline(4356.6,color='k',label='$1\%$ loading',alpha=0.25)

ax[-1].invert_xaxis()
ax[1].legend(loc='upper left',fontsize=8,ncols=3)

fig.tight_layout()
plt.savefig(f'{l_path}\\MLE_rload_sensitivity_hess_inv.pdf')


