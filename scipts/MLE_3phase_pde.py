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
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

dt = 10e-6
t1 = -1e-4
t2 = 0.2


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

read_pscad = False

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
   
#%% --------------- READ PSCAD SOLUTIONS --------------- 
if read_pscad:
    dr = DataReader()
    
    t_pscad, v1,i2,v2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_3c\Cable_3c.infx',t1=t1,t2=t2)
    
    v1 *= 1000/m.p.Vbase
    i2 *= 1000/m.p.Ibase/np.sqrt(3)
    v2 *= 1000/m.p.Vbase
    
    t_pscad, V1,I2,V2 = dr.get_system(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\cable_3c\Cable_3c.infx',series='b',t1=t1,t2=t2)
    
    V1 *= 1000/m.p.Vbase
    I2 *= 1000/m.p.Ibase/np.sqrt(3)
    V2 *= 1000/m.p.Vbase



#%% --------------- GET GROUND TRUTH --------------- 
N_pi = 1
m = SITOBBDS(opts=opts,N_pi = N_pi)
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
x0, R0, R1, R2 = initialize(m.n,r0=1e3,r1=1e3,r2=1e-3)
m.check_observability(m.A,m.C)

# Create input
# u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)
# Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

x, y = m.simulate(Ad,Bd,C,D,x0,uk.real,t1,t2,dt,Sx=None)


#%%
df = obtain_results(t, y[:3,:], 'v1', 'Sim.')
df = obtain_results(t, y[-6:-3,:], 'i2', 'Sim.', df=df)
df = obtain_results(t, y[-3:,:], 'v2', 'Sim.', df=df)
df = obtain_results(t_pscad, v1, 'v1', 'PSCAD FDCM', df=df)
df = obtain_results(t_pscad, i2, 'i2', 'PSCAD FDCM', df=df)
df = obtain_results(t_pscad, v2, 'v2', 'PSCAD FDCM', df=df)
df = obtain_results(t_pscad, V1, 'v1', 'PSCAD PI', df=df)
df = obtain_results(t_pscad, I2, 'i2', 'PSCAD PI', df=df)
df = obtain_results(t_pscad, V2, 'v2', 'PSCAD PI', df=df)

plot_simulations_3phase(df)

w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
# fig.tight_layout()
plt.savefig(f'{w_path}\\3phase_results_tds_primitive_{t2}.pdf')


plot_residuals_3phase(df,'Sim.','PSCAD PI')
# # fig.tight_layout()
plt.savefig(f'{w_path}\\3phase_results_tds_primitive_res_{t2}.pdf')



# plt.plot((y[ - i2).T)


#%% EIGENVALUE ANALYSIS

# m.eigenvalues_analysis(m.A)
# w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\tabs'
# df_to_latex(pd.DataFrame(m.p.Y,index=[1,2,3],columns=[1,2,3]), f'{w_path}\\Y_matrix',caption='Y matrix of 3-phase cable system (conductors only)')
# df_to_latex(pd.DataFrame(m.p.Z,index=[1,2,3],columns=[1,2,3]), f'{w_path}\\Z_matrix',caption='Z matrix of 3-phase cable system (conductors only)')

#%%
# for i in range(1):
#     amp, phi = load_3ph_system(i)
#     u, uk = m.create_input(t1, t2, dt,mode='abc',amp=amp,phi=phi)        
#     uk_seq = m.abc2seq(uk)
    
#     fig, ax = plt.subplots(2,1,dpi=150,sharex=True)
#     # 
#     for i in range(3):
#         ax[0].plot(t*1000,uk.T[:,i].real,label=['A','B','C'][i]+'$='+str(round(amp[i],2))+'\\angle'+str(round(phi[i]*180/np.pi,2))+'^\\circ$')
#         ax[1].plot(t*1000,uk_seq.T[:,i],ls=['-','--',':'][i],label=[0,1,2][i])
        
#     for i in range(2):
#         ax[i].legend(fontsize=8,title=['Phase','Sequence'][i],title_fontsize=8,loc='center right')
#         ax[i].grid()
#         ax[i].set_ylabel('Voltage [p.u.]',fontsize=8)
#     ax[1].set_xlabel('Time [ms]',fontsize=8)
        
#     w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
#     plt.show()
#     plt.savefig(f'{w_path}\\seq_comp_{["balanced","unbalanced1","unbalanced2"][i]}.pdf')


#%% Identification of the parameter space
# opt_params  = [f'{ABC}{i}{j}' for ABC in ['R','L','G','C'] for i in range(3) for j in range(3) if i == j] + ['Rin','Rload']

# data = {
#         'model':[],
#         'params':[],
#         'ests':[],
#         'N_pi':[],
#         'errorNorm':[],
#         'thetahat.fun':[],
#         'thetahat.hess_inv':[],
#         'thetahat.jac':[],
#         'thetahat.message':[],
#         'thetahat.nfev':[],
#         'thetahat.nit':[],
#         'thetahat.njev':[],
#         'thetahat.status':[],
#         'thetahat.success':[],
#         'thetahat.x':[],
#         }

# for model in ['c1_s0_o3_load','c1_s0_o2_load','cable_3c']:
#     for rload in [True,False]:
#         for r_ in ['','R']:
#             for l_ in ['','L']:
#                 for g_ in ['','G']:
#                     for c_ in ['','C']:
#                         # Control flow
#                         abc_string = f'{r_}{l_}{g_}{c_}'
#                         print(abc_string)
#                         if model == 'cable_3c':
#                             N_phi = 3
#                             opt_params  = [f'{ABC}0{j}' for ABC in ['R','L','G','C'] for j in range(2) if ABC in abc_string]
#                             N_pi = 5
#                         else:
#                             opt_params  = [f'{ABC}' for ABC in ['R','L','G','C'] if ABC in abc_string]
#                             N_phi = 1
#                             N_pi = np.nan
                        
#                         if rload:
#                             opt_params += ['Rload']
                        
#                         # ------------- simulate ------------- 
#                         m = SITOBBDS(opts=opts,N_pi = N_pi)
#                         m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
#                         x0, R0, R1, R2 = initialize(m.n,r0=1e-3,r1=1e-3,r2=1e-3)
#                         m.check_observability(m.A,m.C)
        
#                         # Create input
#                         # u, uk = m.create_input(t1, t2, dt,mode='sin')        
#                         Sx = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)
#                         # Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
        
#                         # Get matrices
#                         Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
        
#                         x, y = m.simulate(Ad,Bd,C,D,x0,uk.real,t1,t2,dt,Sx=None)
        
        
#                         # ------------- estimate ------------- 
#                         thetahat0 = [abs(np.random.randn()*1e-4) for k in opt_params]
                        
#                         ests, thetahat, res, A_hat = m.ML_opt_param(opt_params,A,B,C,D,x0, uk.real, y, R0, R1, R2, t1, t2, dt, thetahat0=thetahat0,log=True)
                        
#                         name = "_".join(opt_params)
#                         df = pd.DataFrame({name:ests},index=opt_params)
                        
#                         w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
#                         df.to_excel(f'{w_path}\\MLE_{N_phi}c_pde{N_pi}_{name}.xlsx',header=True,index=True)
                        
#                         # ------------- estimate ------------- 
#                         data['model'].append(model)
#                         data['params'].append(opt_params)
#                         data['ests'].append(ests)
#                         data['N_pi'].append(N_pi)
#                         data['errorNorm'].append(np.linalg.norm(res['Deviations']))
#                         for k in list(thetahat.keys()):
#                             if f'thetahat.{k}' in list(data.keys()):
#                                 data[f'thetahat.{k}'].append(thetahat[k])
                        
