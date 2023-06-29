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
def filter_matrix(s,reshape=None,square_matrix=False):

    # Remove brackets
    if 'nan' in str(s):
        return np.nan
    else:
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
    elif square_matrix is True:
        reshape = int(np.sqrt(len(list_float)))
        m = np.array(list_float).reshape(reshape,reshape)
            
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

#%% Filter data

mle = pd.read_excel(f'{w_path}\\MLE_assesment.xlsx',header=0,sheet_name='Sheet1').set_index('key')

for col in ['true_params','ests','thetahat.jac','thetahat.x']:
    mle[col] = mle[col].apply(lambda x: filter_matrix(x))

mle['params'] = mle['params'].apply(lambda x: eval(x))
    
mle['N_params'] = mle['true_params'].apply(lambda x: len(x))
    
mle['thetahat.hess_inv'] = mle['thetahat.hess_inv'].apply(lambda x: filter_matrix(x,square_matrix=True))
mle['hess_inv'] = mle['thetahat.hess_inv']
mle['R'] = mle['hess_inv'].apply(lambda x: np.diag(x))

mle['jac'] = mle['thetahat.jac']
    

mle['zero'] = mle['jac'].apply(lambda x: np.any(x == 0))
mle['nan'] = mle['jac'].apply(lambda x: np.any(np.isnan(x)))
mle['inf1'] = mle['jac'].apply(lambda x: np.any(x==np.inf))
mle['inf2'] = mle['ests'].apply(lambda x: np.any(x==np.inf))
mle['inf'] = (mle['inf1']) | (mle['inf2'])

mle['names'] = mle['params'].apply(lambda x: ', '.join(x))

#%% RETRIEVE DATA
models = [m for m in mle.model.unique() if isinstance(m,str)]
data = {}
for model in models:
    data[model] = {}

for i, r in mle.iterrows():
    for j, p in enumerate(r['params']):
        data[mle.loc[i,'model']][p] = []
        data[mle.loc[i,'model']][p+'_name'] = []
        data[mle.loc[i,'model']][p+'_var'] = []
        data[mle.loc[i,'model']][p+'_print'] = []
        data[mle.loc[i,'model']][p+'_val'] = []
        data[mle.loc[i,'model']][p+'_trueval'] = []
        
for i, r in mle.iterrows():
    for j, p in enumerate(r['params']):
        data[mle.loc[i,'model']][p].append(r['ests'][j]/r['true_params'][j])
        names = list(set([re.sub(r'\d+', '', n) for n in r['params']]))
        data[mle.loc[i,'model']][p+'_name'].append(', '.join(r['params']))
        data[mle.loc[i,'model']][p+'_print'].append(', '.join(names))
        data[mle.loc[i,'model']][p+'_var'].append(r['R'][r['params'].index(p)]) 
        data[mle.loc[i,'model']][p+'_val'].append(r['ests'][r['params'].index(p)])
        data[mle.loc[i,'model']][p+'_trueval'].append(r['true_params'][r['params'].index(p)])
        # mle.loc[i,p] = r['ests'][j]/r['true_params'][j]        
        # print(mle.loc[i,'model'],p,mle.loc[i,p],r['ests'][j],r['true_params'][j]     )

#%% asd

for model in models:
    d = data[model]
    
    params = [k for k in list(d.keys()) if not '_name' in k and  not '_var' in k and not '_print' in k and not '_val' in k and not '_trueval' in k]    
    
    for p in params:
        fig, ax = plt.subplots(1,1,figsize=(6,3),dpi=150)

        names = d[p+'_print']
        y = d[p]
        yerr = d[p+'_var']
        vals = d[p+'_val']
        true_val = d[p+'_trueval'][0]
        
        ax.bar(names,y)

        ax.errorbar([i for i in range(len(names))], y, yerr, fmt='x', linewidth=1, capsize=4,color='grey',)
        # ax.bar(names,y,)
        # ax.set(yscale='log',ylim(1e))
        ax.set(yscale='log',ylim=(.5,2))
        latex_name  ='{'+p[0] + ('}','}_{'+p[1:]+'}')[len(p)>1]
        ylabel='$\\widehat'+latex_name+'/'+latex_name+'$'
        print(ylabel)
        ax.set_ylabel(ylabel,fontsize=9)
        ax.axhline(1,color='k')
        ax.set_xticklabels(names,rotation=90)

        for i in range(len(names)):
            if vals[i] == 0 or vals[i] == np.inf:
                ax.annotate(vals[i],(i-0.01,0.55),rotation=90,fontsize=4,ha='right',va='bottom')                
            else:
                ax.annotate(vals[i],(i,0.55),rotation=90,fontsize=4,ha='center',va='bottom')
        ax.annotate('${'+p[0] + ('}','}_{'+p[1:]+'}')[len(p)>1]+f'$={true_val}',(-1,1.9),fontsize=8,ha='left',va='top')        
        # ax.grid(ls=':',alpha=0.5)
        fig.tight_layout()
        plt.savefig(f'{l_path}\\MLE_bar_{model}_{p}.pdf')
        # plt.show()
        # plt.close()
        # asdf
        
#%% asd

models = [m for m in mle.model.unique() if isinstance(m,str)]

for model in models:
    df = mle[mle.model == model]
    if 'test' in model:
        params = ['R','L','C','Rload']
    elif model != 'cable_3c':
        params = ['R','L','G','C','Rload']
    else:
        continue
        
    for p in params:
        fig, ax = plt.subplots(1,1,dpi=150)
        df_temp = df.dropna(subset=[p])
        # df_temp = df[p].replace(0, np.nan)  
        # df_temp = df[p].replace(np.inf, np.nan)  

        # df_temp = df_temp.sort_values(by=p)
        
        names = df_temp['names'].values
        y = df_temp[p].values
        yerr = df_temp[p]
        
        # ax.bar(names,y,yerr=yerr)
        ax.bar(names,y)
        
        asdf
    
    

































