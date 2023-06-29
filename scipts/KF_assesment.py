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

    axin,_ = Plots.insert_axis(ax[1],(0.00475,0.00525,0.18,0.3),(0.5,0.3,0.45,.6),arrow_pos=(0.0055,0.02,0.225,0.05))
    axin2,_ = Plots.insert_axis(ax[2],(0.00475,0.00525,0.8,1.4),(0.5,0.3,0.45,.6),arrow_pos=(0.0055,0.02,1.1,0.05+.7))

    for col in df.columns:
        print(col)
        meas, source, sx, sy = col.split('_')
        i = ['v1','i2','v2'].index(meas)
       
        
        k = ['Sim.','Est.','Det. Sim.'].index(source)       
        if k == 0:
            ax[i].plot(df[col].dropna(),color='k',zorder=3,label=source)    
            if 'v2' in col:
                axin2.plot(df[col].dropna(),color='k',zorder=3)    
            if 'i2' in col:
                axin.plot(df[col].dropna(),color='k',zorder=3)    
        elif k == 1:
            if float(sx) == 1e3 and float(sy) == 1e3:
                cnt = 0
            elif float(sx) == 1e-3 and float(sy) == 1e-3:
                cnt = 1
            elif float(sx) == 1e3 and float(sy) == 1e-3:
                cnt = 2
            elif float(sx) == 1e-3 and float(sy) == 1e3:
                cnt = 3

            ax[i].plot(df[col].dropna(),color=clrs[cnt],zorder=4,label=f'{source} (Rx={sx},Ry={sy})',alpha=0.5,ls=['-','--','-.',':'][cnt])    
            if 'i2' in col:
                axin.plot(df[col].dropna(),color=clrs[cnt],zorder=4,alpha=0.5,ls=['-','--','-.',':'][cnt])    
            if 'v2' in col:
                axin2.plot(df[col].dropna(),color=clrs[cnt],zorder=4,alpha=0.5,ls=['-','--','-.',':'][cnt])    
        elif k == 2:

            ax[i].plot(df[col].dropna(),color='black',zorder=4,label=f'{source}',alpha=0.5)    
            if 'i2' in col:
                axin.plot(df[col].dropna(),color='black',zorder=4,label=f'{source}',alpha=0.5)       
            if 'v2' in col:
                axin2.plot(df[col].dropna(),color='black',zorder=4,label=f'{source}',alpha=0.5)       
            
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
r123 = np.ones(n)
R0=np.diag(r123)*1/10
R1=np.diag(r123)*1/10
R2=np.diag(r123)*10

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
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

#%% --------------- GET GROUND TRUTH --------------- 
x, ydet = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx*0,Sy=Sy*0)      

data = {'meas':[],
        's':[],
        'sx':[],
        'sy':[],
        'Rx':[],
        'Ry':[],
        'abs_mean':[],
        'mean':[],
        'std':[],
        }
l_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
w_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'

for sx in [0,1]:
    for sy in [0,1]:
        # Simulate the system
        x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx*sx,Sy=Sy*sy)      

                
        
        df = obtain_results_1phase(t, y[0,:], 'v1', f'Sim._{sx}_{sy}')
        df = obtain_results_1phase(t, y[1,:], 'i2', f'Sim._{sx}_{sy}', df=df)
        df = obtain_results_1phase(t, y[2,:], 'v2', f'Sim._{sx}_{sy}', df=df)
                
        if sx+sy >= 1:
            df = obtain_results_1phase(t, ydet[0,:], 'v1', f'Det. Sim._{sx}_{sy}', df=df)
            df = obtain_results_1phase(t, ydet[1,:], 'i2', f'Det. Sim._{sx}_{sy}', df=df)
            df = obtain_results_1phase(t, ydet[2,:], 'v2', f'Det. Sim._{sx}_{sy}', df=df)
        
        #%%
        cnt = 0

        for Rx in [1e-3,1e3]:
            for Ry in [1e-3,1e3]:
                x_hat_pred, y_hat_pred, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0, uk, y, R0*Rx, R1*Rx, R2*Ry, t1, t2, dt)
                
                df = obtain_results_1phase(t, y_hat_pred[0,:], 'v1', f'Est._{Rx}_{Ry}', df=df)
                df = obtain_results_1phase(t, y_hat_pred[1,:], 'i2', f'Est._{Rx}_{Ry}', df=df)
                df = obtain_results_1phase(t, y_hat_pred[2,:], 'v2', f'Est._{Rx}_{Ry}', df=df)

                for meas in ['v1','i2','v2']:
        
                    err=(df[f'{meas}_Est._{Rx}_{Ry}'] -  df[f'{meas}_Sim._{sx}_{sy}'])
                        
                    data['meas'].append(meas)
                    data['s'].append(sx+sy)
                    data['sx'].append(sx)
                    data['sy'].append(sy)
                    data['Rx'].append(Rx)
                    data['Ry'].append(Ry)
                    data['abs_mean'].append(abs(err.mean()))
                    data['mean'].append(err.mean())
                    data['std'].append(err.std())
        
        plot_simulations_1phase(df)
        plt.savefig(f'{l_path}\\kf_{sx}_{sy}.pdf')

        plot_residuals_1phase_kf(df,sx,sy)
        plt.savefig(f'{l_path}\\kf_res_{sx}_{sy}.pdf')

data = pd.DataFrame(data)
data.to_excel(f'{w_path}\\KF_assesment.xlsx',header=True)

# data.style.to_latex(filename + '.tex',position = 'H',**kwargs)

#%% With confidential interval
Rx = 1e3
Ry = 1e-3
x_hat_pred, y_hat_pred, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0, uk, y, R0*Rx, R1*Rx, R2*Ry, t1, t2, dt)

x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)      

    

df = obtain_results_1phase(t, y[0,:], 'v1', f'Sim._{sx}_{sy}')
df = obtain_results_1phase(t, y[1,:], 'i2', f'Sim._{sx}_{sy}', df=df)
df = obtain_results_1phase(t, y[2,:], 'v2', f'Sim._{sx}_{sy}', df=df)



















