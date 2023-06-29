# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 07:31:00 2023

@author: bvilm
"""

import sys
sys.path.insert(0, '../')  # add parent folder to system path

import numpy as np
import matplotlib.pyplot as plt
from datareader import DataReader
# from solver import SITOBBDS, ParamWrapper
from sysid_app import SITOBBDS, ParamWrapper
from PowerSystem import PS
import control as ctrl
from Plots import insert_axis, update_spine_and_ticks
import pandas as pd
import os
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

I = lambda n: np.eye(n)

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02

t1 = 0
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
r123 = np.array([1,1,1])*1e-4
r1 = 1e1
r2 = 1e3
R0=np.diag(r123)*1e3
R1=np.diag(r123)*r1
R2=np.diag(r123)*r2


#%%
n = 10
rins = np.linspace(1e-5,1,n)
rloads = np.linspace(1,1e4,n)
Cn = np.zeros((n,n))

for i, rin in enumerate(rins):
    print(i)
    for j, rload in enumerate(rloads):    
        m = SITOBBDS()
        params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4}
        m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
        
        Cn[i,j] = np.linalg.cond(m.A)


n2 = 20
x1,x2 = 0.45,0.55
y1,y2 = 1,1e3
rins2 = np.linspace(x1,x2,n2)
rloads2 = np.linspace(y1,y2,n2)
Cn2 = np.zeros((n2,n2))

for i, rin in enumerate(rins2):
    print(i)
    for j, rload in enumerate(rloads2):    
        m = SITOBBDS()
        params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4}
        m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
        
        Cn2[i,j] = np.linalg.cond(m.A)
#%%
x,y = 0,-1
rin = rins[x]
rload = rloads[y]

m = SITOBBDS()
params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

print(x,',',y,'\n',rin,rload)
print(np.linalg.cond(m.A))
print(Cn[x,y])



#%%
import matplotlib.colors as colors
path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'
fig, ax = plt.subplots(1,1,dpi=200)

X, Y = np.meshgrid(rins, rloads)
pcm = ax.pcolormesh(X,Y,Cn.T,
              norm=colors.LogNorm(vmin=Cn.min(), vmax=Cn.max()),
              # cmap='PuBu_r', 
              shading='auto')
ax.set(xlabel='$R_{in}$ [$\\Omega$]',ylabel='$R_{load}$ [$\\Omega$]')
fig.colorbar(pcm, ax=ax)

ax.set(xlim=(0,1),ylim=(0,1e4))

axin, ax = insert_axis(ax,(x1,x2,0,y2),(0.25,0.4,0.5,0.5),arrow_pos = (0.5,0.5,1100,3000),grid=False)

X, Y = np.meshgrid(rins2, rloads2)
pcm = axin.pcolormesh(X,Y,Cn2.T,
              norm=colors.LogNorm(vmin=Cn.min(), vmax=Cn.max()),
              # cmap='PuBu_r', 
              shading='auto')

ax.scatter([0.5],[200],color='red',marker='x')
axin.scatter([0.5],[200],color='red',marker='x')

axin = update_spine_and_ticks(axin,color='white',tick_params=dict(colors='white'))
fig.tight_layout()
plt.savefig(f'{path}\\Condition_number.pdf')






















