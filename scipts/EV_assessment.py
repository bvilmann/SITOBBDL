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

def obtain_results(t,y,measurement:str,source:str,df=None,N_pi=None):
    # Make sure of rotation of measurements
    if y.shape[0] < y.shape[1]:
        y = y.T
    
    for i in range(y.shape[1]):
        if 'PSCAD' in source:
            header = f'{measurement}_{["A","B","C"][i]}_{source}_None'
        else: 
            header = f'{measurement}_{["A","B","C"][i]}_{source}_{N_pi}'
            
        if df is None and i == 0:
            df = pd.DataFrame({'t': t.round(7),header: y[:,i]}).set_index('t')

        else:
            df = merge_df(t.round(7),y[:,i],header,df)
        
    return df


def plot_simulations_3phase(df):
    fig, ax = plt.subplots(3,1,dpi=150,sharex=True)

    for col in df.columns:
        meas, phase, source, N_pi = col.split('_')
        i = ['v1','i2','v2'].index(meas)
        j = ['A','B','C'].index(phase)
        k = ['Sim.','PSCAD FDCM','Est.','PSCAD PI','PSCAD'].index(source)

        if N_pi != 'None':
            if N_pi == '50':
                ax[i].plot(df[col].dropna(),alpha=[0.5,0.125,0.125][j],ls='--',color=clrs[k],zorder=4,label=('',f'{source} ($N_\\pi = ' + N_pi + '$)')[phase=='A'])
            elif N_pi == '100':
                ax[i].plot(df[col].dropna(),alpha=[1,0.125,0.125][j],ls=':',color=clrs[k],zorder=5,label=('',f'{source} ($N_\\pi = ' + N_pi + '$)')[phase=='A'])
            
        else:
            if k == 0:
                ax[i].plot(df[col].dropna(),alpha=[0.25,0.125,0.125][j],color=clrs[k],zorder=3,label=('',f'{source} ($N_\\pi = 1$)')[phase=='A'])
            else:
                ax[i].plot(df[col].dropna(),alpha=[1,0.25,0.25][j],color=clrs[k],zorder=3,label=('',source)[phase=='A'])

    for i in range(3):
        ax[i].grid()
        ax[i].set_ylabel(['$V_k$ [pu]','$I_{km}$ [pu]','$V_m$ [pu]'][i])
        if i ==0:
            ax[i].legend(loc='upper right',fontsize=6)


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
    
    plt.colorbar()

    return fig, ax

I = lambda n: np.eye(n)



#%% ========== Settings ========== 
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
w_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}

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
        'method':'L-BFGS-B'
        # 'method':'BFGS'
        }
   

#%% EIGENVALUE ANALYSIS - N_pi
cmap = plt.get_cmap('viridis')
fig, ax =plt.subplots(1,1,dpi=150)
# plt.gca().set_aspect('equal', adjustable='box')    

r = 1
theta = np.linspace(0, 2*np.pi, 100)  # angles
x = r * np.cos(theta)
y = r * np.sin(theta)

ax.grid()
# Plot the circle
# ax.plot(x, y,color='k',zorder=3)

N_pis = np.arange(5,25,5)
for i, N_pi in enumerate([1,10,25,40,50]):
    m = SITOBBDS(opts=opts,N_pi = N_pi)
    m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
    
    lambs = np.linalg.eigvals(m.A)
    df = pd.DataFrame({'real':lambs.real,'imag':lambs.imag})

    ax.scatter(lambs.real,lambs.imag,color=cmap(i / len(N_pis)),alpha=0.25,label='$N_{\\pi}='+str(N_pi)+'$',zorder=1/(i+1)+2)

    xy_idx1 = df['real'].idxmin()
    x1,y1 = df.loc[xy_idx1,'real'],df.loc[xy_idx1,'imag']
    
    xy_idx2 = df['imag'].idxmin()
    x2,y2 = df.loc[xy_idx2,'real'],df.loc[xy_idx2,'imag']

    ax.plot([x1,x2],[y1,y2],color='k',alpha=0.1,ls='--',label=('','Helper line')[i==4])
    ax.plot([x1,x2],[y1,-y2],color='k',alpha=0.1,ls='--')

# plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()

ax.set(xlabel='$\\sigma$',ylabel='$j\\omega$')
ax.legend(loc='upper left',ncols=2,fontsize=8)
plt.savefig(f'{w_path}\\pde_eigenvalue_analysis.pdf')


#%% EIGENVALUE ANALYSIS - Rload
cmap = plt.get_cmap('viridis')
fig, ax =plt.subplots(1,1,dpi=150)
# plt.gca().set_aspect('equal', adjustable='box')    

r = 1
theta = np.linspace(0, 2*np.pi, 100)  # angles
x = r * np.cos(theta)
y = r * np.sin(theta)

ax.grid()
# Plot the circle
# ax.plot(x, y,color='k',zorder=3)

rloads = 43.56*np.array([0.5,1,1.5,2,5,10,100])
rloads = 43.56*np.arange(0.1,100,0.2)
for i, rload in enumerate(rloads):
    params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4*0}
    m = SITOBBDS(opts=opts,N_pi = 1)
    m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
    
    lambs = np.linalg.eigvals(m.A)
    df = pd.DataFrame({'real':lambs.real,'imag':lambs.imag})

    # ax.scatter(lambs.real,lambs.imag,color=cmap(i / len(rloads)),alpha=0.5,label='$R_{load}='+str(round(rload,2))+'$',zorder=1/(i+1)+2,s=(rload,43.56*10)[rload>43.56*10])
    ax.scatter(lambs.real,lambs.imag,color=cmap(i / len(rloads)),alpha=0.5,label='$R_{load}='+str(round(rload,2))+'$',zorder=1/(i+1)+2)

# # Create a normalized colormap
# # Create a normalized colormap
# import matplotlib.colors as mcolors
# norm = mcolors.Normalize(vmin=min(rloads), vmax=max(rloads))

# # Create a colorbar
# cbar = plt.colorbar(mcolors.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical'))

# # plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
# plt.colorbar()
# # plt.show()
ax.set(xlabel='$\\sigma$',ylabel='$j\\omega$')
# ax.legend(loc='upper left',ncols=2,fontsize=8)
plt.savefig(f'{w_path}\\pde_eigenvalue_analysis_rload.pdf')

#%%
import matplotlib.colors as mcolors
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)

model =  f'c1_s0_o3_load'
model =  f'cable_3c'
rloads = 43.56*np.array([0.001,1,1e6])
for i, rload in enumerate(rloads):
    params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4*0}

    m = SITOBBDS()
    m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True,seq=False)
        
    # Get eigenvalues, right eigenvector, and left eigenvector.
    lambs, Phi = np.linalg.eig(m.A)
    Psi = np.linalg.inv(Phi)
    
    # Get participation factors
    P = Phi * Psi.T

    fig, ax =plt.subplots()
    P = np.where(abs(P)>0.05,abs(P),np.nan) 
    im = ax.imshow(P, clim=(0, 1))
    # cbar = plt.colorbar(im, cmap=cmap, 
    norm=norm, orientation='vertical')
    cbar = plt.colorbar(im, cmap=cmap, orientation='vertical',extend='max')

    xlabs = ['$\\lambda_{'+str(i_)+'}$' for i_ in range(1,P.shape[0]+1)]
    ylabs = []
    for state in ['$V_{k,','$I_{km,','$V_{m,']:
        ylabs += [state+str(i_)+'}$' for i_ in range(1,4)]

    ax.set(xticks=[i for i in range(P.shape[0])],yticks=[i for i in range(P.shape[0])])
    ax.set(xticklabels=xlabs,yticklabels=ylabs)
    
    for j,_ in enumerate(range((P.shape[0]))):
        for k,_ in enumerate(range((P.shape[0]))):
            ax.annotate(round(abs(P)[k,j],1),(j,k),color=('white','black')[P[k,j]>0.75],va='center',ha='center',fontsize=10)

    ax.set(title='$R_{load}='+str(round(rload/43.56,1)) + ' \\mathrm{ [pu]},\\quad\\mathrm{rank}\\left(P\\right)=$'+str(np.linalg.matrix_rank(Phi * Psi.T)))
    
    fig.tight_layout()
    # plt.savefig(r'C:\Users\bvilm\Dropbox\DTU\10_params_est\pres\participation_matrix_'+str(i)+'.pdf')
    plt.show()
    
    
