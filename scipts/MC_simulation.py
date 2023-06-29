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
def load_3ph_system(i):
    amp = [[1,1,1],[1,1,0],[0.5,1.05,0.75]][i]
    phi = [[0,-2/3*np.pi,2/3*np.pi],[0,-2/3*np.pi,2/3*np.pi],[0,-3/2*np.pi,2/3*np.pi]][i]
    return np.array(amp), np.array(phi)

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)
n = 3
I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -1e-4
t2 = 0.2
t = np.arange(t1,t2+dt,dt)

# Initial conditions
x0 = np.zeros(n)

# Covariance
r123 = np.ones(n)*1e-4
R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1

#%%
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}
m = SITOBBDS()
m.get_model(f'c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)
m.plot_simulations(t, [y],labels=['$y$'])


#%% --------------- PERFORM MONTE CARLO SIMULATION --------------- 
# n_iters = 1e3
# opt_params  = ['R','Rin','Rload','L','C']

# df = m.MC_param(opt_params,n_iters, A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt)

#%% --------------- PERFORM CREATE SENSITIVITY GRID --------------- 
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0,'SCR':5,'XR':5}
opt_params  = ['R','L','C','G','Rload']

model = f'c1_s0_o3_load'
opt_params  = ['R','L','C','G']
model = f'c1_s0_o3_load_test'
opt_params  = ['R','L','C']
model = f'Cable_3c_grid'
opt_params  = ['SCR','XR']
N = 10
delta = 0.5
m = SITOBBDS()
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
if model == 'Cable_3c_grid':
    amp, phi = load_3ph_system(0)

    u, uk = m.create_input(t1, t2, dt, mode='abc',amp=1.05*amp,phi=5*np.pi/180+phi)        
    u, ug = m.create_input(t1, t2, dt, mode='abc',amp=amp,phi=phi)        
    uk = np.vstack([uk,ug]).real

else:      
    u, uk = m.create_input(t1, t2, dt,mode='sin')        



Sx = m.create_noise(t1, t2, dt,amp=.005,dim=n,seed=1234)
Sy = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1235)
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=None,Sy=None)
m.plot_simulations(t, [y],labels=['$y$'])

#%%
df = m.ML_sensitivity(opt_params,N, A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,delta=delta)

#%%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

N = len(opt_params)
# Scatter plot of each parameter vs evaluation
fig, ax = plt.subplots(N, N, figsize=(8, 8),dpi=200)

# Get the parameters for the minimum cost function
min_J_index = df['J'].idxmin()
min_J_parameters = df.loc[min_J_index, opt_params]

cmap = plt.cm.viridis
norm = colors.Normalize(vmin=df['J'].min(), vmax=df['J'].max())  # Here, the colorbar range is fixed from -1 to 1


for i, i_param in enumerate(opt_params):
    for j, j_param in enumerate(opt_params):
        if i == j:            

            ax[i,j].axvline(m.p.params[j_param],color='black',lw=1,label='True value')
            ax[i,j].scatter(df[i_param], df['J'],alpha=0.5,label='Data')
            # ax[i,j].set_title(f'J vs {i_param}')
            ax[i,j].set_xlabel(j_param)
            ax[i,j].set_ylabel('J')
            ax[i,j].scatter([min_J_parameters[i_param]], [df.loc[min_J_index,'J']],color='red',marker='x',label='min arg $J$')
            if i == 0:
                ax[i,j].legend(loc='upper left',fontsize=7)
            
        # elif i>j:
        #     sc = ax[i,j].scatter(df[i_param], df[j_param],c=df['J'], cmap=cmap, norm=norm,alpha=0.5)
        #     # ax[i,j].set_title(f'{j_param} vs {i_param}')
        #     ax[i,j].set_xlabel(j_param)
        #     ax[i,j].set_ylabel(i_param)
        #     if i == 1 and j == 0:
        #         # Create an inset axes for the colorbar
        #         cax = inset_axes(ax[i,j],
        #                          width="90%",  # width = 5% of parent_bbox width
        #                          height="5%",  # height : 50%
        #                          loc='lower left',
        #                          bbox_to_anchor=(0.05, 0.9, 1, 1),
        #                          bbox_transform=ax[i,j].transAxes,
        #                          borderpad=0,
        #                          )
    
        #         cbar = plt.colorbar(sc, cax=cax, label='J',orientation='horizontal')
        #         cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
            
        else:
            # get
            # get unique sorted values
            N_i = len(df[i_param].unique())
            N_j = len(df[j_param].unique())
            df_temp = df[[i_param,j_param,'J']].drop_duplicates(subset=[i_param,j_param])
            J_2d = df_temp['J'].values.reshape((N_i, N_j))
            x_unique = np.sort(df_temp[i_param].unique())
            y_unique = np.sort(df_temp[j_param].unique())
            if i > j:
                contour = ax[i,j].imshow(J_2d, extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()], cmap=cmap,aspect= (x_unique.max()-x_unique.min())/(y_unique.max()-y_unique.min()))
            else:
                contour = ax[i,j].imshow(J_2d.T, extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()], cmap=cmap,aspect= (x_unique.max()-x_unique.min())/(y_unique.max()-y_unique.min()))

            ax[i,j].set_xlabel(i_param)
            ax[i,j].set_ylabel(j_param)

            # plot true and optimized value
            ax[i,j].scatter([m.p.params[i_param]],[m.p.params[j_param]],color='blue',marker='o',label='True value')
            ax[i,j].scatter([min_J_parameters[i_param]],[min_J_parameters[j_param]],color='red',marker='x',label='min arg $J$')
            if i == 0 and j == 1:
                ax[i,j].legend(loc='upper left',fontsize=7)
            else:
                # Create an inset axes for the colorbar
                cax = inset_axes(ax[i,j],
                                 width="90%",  # width = 5% of parent_bbox width
                                 height="5%",  # height : 50%
                                 loc='lower left',
                                 bbox_to_anchor=(0.05, 0.9, 1, 1),
                                 bbox_transform=ax[i,j].transAxes,
                                 borderpad=0,
                                 )
                
                cbar = plt.colorbar(contour, cax=cax, label='J',orientation='horizontal')
                cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)

fig.tight_layout()

l_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
plt.savefig(f'{l_path}\\MLE_grid_sensitivity_{model}_{opt_params}.pdf')
plt.show()


#%% Compare results
n = 3
J_opt = df.loc[min_J_index, opt_params]*m.p.Zbase
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 200,'phi':np.pi/4*0}
params.update(J_opt)
m = SITOBBDS()
m.get_model(model,discretize=True,dt=10e-6,params=params,pu=True)
    
# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        
Sx = m.create_noise(t1, t2, dt,amp=.01,dim=n,seed=1234)
# Sx = None         

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# Simulate the system
_, y1 = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx*0)

m.plot_simulations(t, [y,y1],labels=['$y$','$y1$'])





