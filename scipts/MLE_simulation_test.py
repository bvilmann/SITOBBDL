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
import Plots
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

def plot_simulations_1phase(df):
    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)

    for col in df.columns:
        print(col)
        meas, source = col.split('_')
        i = ['v1','i2','v2'].index(meas)
       
        
        if source == 'Sim.':
            ax[i].plot(df[col].dropna(),color='k',zorder=5,label='Ground truth',lw=0.75)    
        else:
            ax[i].plot(df[col].dropna(),zorder=4,alpha=0.5,label=f'${source}$')    
            
    for i in range(3):
        ax[i].set_ylim(df[f"{['v1','i2','v2'][i]}_Sim."].min()*1.12,df[f"{['v1','i2','v2'][i]}_Sim."].max()*1.12)
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right',fontsize=7.5,ncols=max(1,df.shape[1]//(4*3)))
            # ax[i].legend(loc='upper right',fontsize=7.5)


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
def obtain_results_1phase(t,y,measurement:str,source:str,df=None):
    # Make sure of rotation of measurements
    
    header = f'{measurement}_{source}'
    if df is None:
        df = pd.DataFrame({'t': t.round(7),header: y}).set_index('t')

    else:
        df = merge_df(t.round(7),y,header,df)
        
    return df

I = lambda n: np.eye(n)
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


#%% ========== Settings ========== 
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

dt = 10e-6
t1 = -1e-4
t2 = 0.04

t = np.arange(t1,t2+dt,dt)

# get input
# --------- Covariance ---------
r123 = np.ones(n)
R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e-3
thetahat0 = 1e-12




l_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'

#%% single-phase TEST 
model = 'c1_s0_o3_load_test'
# --------------- GET GROUND TRUTH --------------- 
m = SITOBBDS()
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
n = m.n
x0 = np.zeros(n)
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

df = obtain_results_1phase(t, y[0,:], 'v1', f'Sim.')
df = obtain_results_1phase(t, y[1,:], 'i2', f'Sim.', df=df)
df = obtain_results_1phase(t, y[2,:], 'v2', f'Sim.', df=df)

# --------------- GET GROUND TRUTH --------------- 
mle = pd.read_excel(f'{w_path}\\MLE_assesment.xlsx',header=0,index_col=0).set_index('key')

files = [f for f in os.listdir(w_path) if model in f]

for file in files:
    # Fetch data
    mle_params = pd.read_excel(f'{w_path}\\{file}',header=0,index_col=0)
    mle_params['Normalized error'] = (mle_params.Estimated-mle_params.System)/mle_params.System*100
    opt_params = list(mle_params.index)
    name = ', '.join(list(mle_params.index))
    key = 'c1_s0_o3_load_test-' + file.split('pdenan_')[1][:-5] + '-nan-m'    
    print(opt_params,file)

    # Get gradients and hessian    
    try:
        jac = filter_matrix(mle.loc[key,'thetahat.jac'])
        hess_inv = filter_matrix(mle.loc[key,'thetahat.hess_inv'],reshape=len(jac))
    except KeyError as e:
        print(e)
        continue
    
    # evaluate
    zero = np.any(jac == 0)
    nan = np.any(np.isnan(jac) )
    inf = np.any((jac == np.inf))

    # ========= simulate =========
    print(nan,inf)
    if not nan and not inf:
        # Update parameters
        params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}
        for i, row in mle_params.iterrows():
            params[i] = row['Estimated']
        # Simulate
        m = SITOBBDS()
        m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
        Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

        x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)
        df = obtain_results_1phase(t, y[0,:], 'v1', name, df=df)
        df = obtain_results_1phase(t, y[1,:], 'i2', name, df=df)
        df = obtain_results_1phase(t, y[2,:], 'v2', name, df=df)       

plot_simulations_1phase(df)    
plt.savefig(f'{l_path}\\MLE_{model}_sim.pdf')

#%% single-phase
model = 'c1_s0_o3_load'
# --------------- GET GROUND TRUTH --------------- 
m = SITOBBDS()
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
n = m.n
x0 = np.zeros(n)
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

df = obtain_results_1phase(t, y[0,:], 'v1', f'Sim.')
df = obtain_results_1phase(t, y[1,:], 'i2', f'Sim.', df=df)
df = obtain_results_1phase(t, y[2,:], 'v2', f'Sim.', df=df)

# --------------- GET GROUND TRUTH --------------- 
files = [f for f in os.listdir(w_path) if model in f and 'test' not in f]

for file in files:
    # Fetch data
    mle_params = pd.read_excel(f'{w_path}\\{file}',header=0,index_col=0)
    mle_params['Normalized error'] = (mle_params.Estimated-mle_params.System)/mle_params.System*100
    opt_params = list(mle_params.index)
    name = ', '.join(list(mle_params.index))
    key = f'{model}-' + file.split('pdenan_')[1][:-5] + '--m'    
    print(opt_params,file)

    # Get gradients and hessian    
    try:
        jac = filter_matrix(mle.loc[key,'thetahat.jac'])
        hess_inv = filter_matrix(mle.loc[key,'thetahat.hess_inv'],reshape=len(jac))
    except KeyError as e:
        print(e)
        continue
    
    # evaluate
    zero = np.any(jac == 0)
    nan = np.any(np.isnan(jac) )
    inf1 = np.any(jac == np.inf)
    inf2 = np.any(mle_params.Estimated.values == np.inf)
    inf = (inf1 or inf2)

    # ========= simulate =========
    print(nan,inf)
    if not nan and not inf:
        # Update parameters
        params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}
        for i, row in mle_params.iterrows():
            params[i] = row['Estimated']
        # Simulate
        m = SITOBBDS()
        m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
        Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

        x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)
        df = obtain_results_1phase(t, y[0,:], 'v1', name, df=df)
        df = obtain_results_1phase(t, y[1,:], 'i2', name, df=df)
        df = obtain_results_1phase(t, y[2,:], 'v2', name, df=df)       

plot_simulations_1phase(df)    
plt.savefig(f'{l_path}\\MLE_{model}_sim.pdf')





