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
# from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime
import linecache

from numpy.linalg import inv

def load_cable_data(path, file,n_conductors):
    n = n_conductors
    fpath = f'{path}\\{file}.out'

    conv = {0: lambda x: str(x)}

    # df = pd.read_csv(fpath, skiprows=59,nrows=n,converters=conv)
    with open(fpath,'r') as f:
        for i, line in enumerate(f):
            cnt = 0
            if 'SERIES IMPEDANCE MATRIX (Z)' in line:
                # print(f'line: {i}')
                zline = i + 1 + 1
                # z = np.loadtxt(fpath,skiprows=i+1,max_rows=7,converters=conv,delimiter=',')
            elif 'SHUNT ADMITTANCE MATRIX (Y)' in line:
                yline = i + 1 + 1
                # y = np.genfromtxt(fpath,skip_header=i+1,max_rows=7,autostrip=True)            
    f.close()

    Z = np.zeros((n,n),dtype=np.complex128)
    Y = np.zeros((n,n),dtype=np.complex128)
    for i in range(n):
        z_i = linecache.getline(fpath, zline).strip().split(' '*3)
        y_i = linecache.getline(fpath, yline).strip().split(' '*3)
        
        Z[i,:] = [complex(float(x.split(',')[0]),float(x.split(',')[1])) for x in z_i]
        Y[i,:] = [complex(float(x.split(',')[0]),float(x.split(',')[1])) for x in y_i]

    R = Z.real
    X = Z.imag
    G = Y.real
    B = Y.imag
    return (Z, Y), (R, X, G, B)

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 50)

I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

# Initial parameters
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}


#%%
m = SITOBBDS()

m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

#%%
# Settings
N = 1 # number of Pi
f = np.linspace(0,10e3,1000) # frequencies of interest
fnom = 50
path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\LCP'
file = r'Cable_2'
n_conductors = 7

# helper functions
def zs(r,l,f):
    if isinstance(r,float) or isinstance(r,int):
        zs = r + 1j*2*np.pi*f*l
    elif isinstance(r,np.ndarray):
        if isinstance(f,float) or isinstance(f,int):
            zs = np.zeros((r.shape[0],r.shape[1]),np.complex128)
        else:
            zs = np.zeros((*r.shape,len(f)),np.complex128)

        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if isinstance(f,float) or isinstance(f,int):
                    zs[i,j] = r[i,j] + 1j*2*np.pi*f*l[i,j]
                else:
                    zs[i,j,:] = r[i,j] + 1j*2*np.pi*f*l[i,j]
                    
    return zs
        

def yp(g,c,f):
    if isinstance(g,float) or isinstance(g,int):
        yp = g + 1j*2*np.pi*f*c
    elif isinstance(g,np.ndarray):
        if isinstance(f,float) or isinstance(f,int):
            yp = np.zeros((*g.shape),np.complex128)
        else:
            yp = np.zeros((*g.shape,len(f)),np.complex128)

        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if isinstance(f,float) or isinstance(f,int):
                    yp[i,j] = g[i,j] + 1j*2*np.pi*f*c[i,j]
                else:
                    yp[i,j,:] = g[i,j] + 1j*2*np.pi*f*c[i,j]

    return yp

        
zs = lambda r,l,f: r + 1j*2*np.pi*f*l
yp = lambda g,c,f: g + 1j*2*np.pi*f*c
z0 = lambda z,y: np.sqrt(z/y)
gamma = lambda z, y: np.sqrt(z*y)

# Getting values
R = m.p.R
L = m.p.L
C = m.p.C1

# Getting per-unit-length
d = 90e3 # distance [m]
r, l = 0.189787342E-03,0.121654383E-02/(2*np.pi*fnom)
g, c = 0.000000000E+00,0.533520750E-07/(2*np.pi*fnom)

# SERIES IMPEDANCE MATRIX (Z) [ohms/m]: 
d = 90e3 # distance [m]
r, l = 0.689202756E-04,0.677844477E-03/(2*np.pi*fnom)

# # SHUNT ADMITTANCE MATRIX (Y) [mhos/m]: 
g, c = 0.000000000E+00,0.631300096E-07/(2*np.pi*fnom)

# # LONG-LINE CORRECTED SERIES IMPEDANCE MATRIX [ohms]: 
# d = 90 # distance [m]
# r,x = np.array([0.550451976E+01,0.575768561E+02])
# l = x / (2*np.pi * f)
# # LONG-LINE CORRECTED SHUNT ADMITTANCE MATRIX [mhos]: 
# g, b = np.array([0.179068548E-04,0.585164562E-02])
# c = b / (2*np.pi * f)

# _, (r,x,g,b) = load_cable_data(path, file, n_conductors)
# l = x / (2*np.pi*fnom)
# c = b / (2*np.pi*fnom)

# Get impedance as a function of frequency
rs = 0.550451976E+01
rp = 1/0.179068548E-04
z = zs(r,l,f)
y = yp(g,c,f)
zn = zs(r,l,fnom)
yn = zs(g,c,fnom)
z0 = np.sqrt(zn/yn)

# calculate 
Vs, Is = 1*m.p.Vbase,(m.p.Zbase/(200+abs(rs)))*m.p.Ibase
# Vs, Is = 1*m.p.Vbase,.3*m.p.Ibase
Vr = Vs*np.cosh(d*gamma(z,y)) - z0*Is*np.sinh(d*gamma(z,y))
Ir = Is*np.cosh(d*gamma(z,y)) - 1/z0*Vs*np.sinh(d*gamma(z,y))
Z = Vr/Ir
Y = Z**(-1)
    

# # LONG-LINE CORRECTED SERIES IMPEDANCE MATRIX [ohms]: 
# d = 1 # distance [m]
# r, l = 0.550451976E+01,0.575768561E+02 / (2*np.pi * f)
# # LONG-LINE CORRECTED SHUNT ADMITTANCE MATRIX [mhos]: 
# g, c = 0.179068548E-04,0.585164562E-02 / (2*np.pi * f)

# # Get impedance as a function of frequency
# z = zs(r,l,f)
# y = yp(g,c,f)
# zn = zs(r,l,fnom)
# yn = zs(g,c,fnom)
# z0 = np.sqrt(zn/yn)

# # calculate 
# Vs, Is = 1*m.p.Vbase,(m.p.Zbase/200)*m.p.Ibase
# Vr = Vs*np.cosh(d*gamma(z,y)) - z0*Is*np.sinh(d*gamma(z,y))
# Ir = Is*np.cosh(d*gamma(z,y)) - 1/z0*Vs*np.sinh(d*gamma(z,y))
# Z1 = Vr/Ir
# Y1 = Z**(-1)

#%% Plotting
plot_path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

# https://stackoverflow.com/questions/64405959/how-to-zoom-inside-a-plot

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# getting frequency responses
f_pi = np.genfromtxt(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\freq\Harm_1c_pi.out',skip_header=1)
f_fdcm = np.genfromtxt(r'C:\Users\bvilm\PycharmProjects\SITOBB\data\freq\Harm_1c_fdcm.out',skip_header=1)

fig, ax = plt.subplots(2,1,dpi=200,sharex=True)
# Plot magnitudes

axin1 = ax[0].inset_axes([0.3625, 0.55, 0.15, 0.4])

for ax_ in [axin1,ax[0]]:    
    ax_.plot(f,abs(Z),label=f'Calculated from FDCM',zorder=3)
    # ax_.plot(f,abs(Z1),label=f'Calculated from FDCM',zorder=3)
    ax_.plot(f_pi[:,0],f_pi[:,1],label=f'Frequency resp. $\\pi$-equivalent',zorder=3)
    ax_.plot(f_fdcm[:,0],f_fdcm[:,1],label=f'Frequency resp. (FDCM)',zorder=3,ls='--')
    ax_.axvline(50,color='k',lw=0.75)

x1, x2= 45,60
y1, y2= 100,170

axin1.set_xlim(x1,x2)
axin1.set_ylim(y1,y2)
axin1.grid()

ax[0].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],color='grey',lw=0.75,ls='-')

ax[0].annotate('', xy=(55, 140),
             xycoords='data',
             xytext=(600, 280),
             textcoords='data',
             arrowprops=dict(arrowstyle= '->',
                             color='grey',
                             alpha=0.75,
                             # lw=3.5,
                             ls='-')
           )

# Plot magnitudes
ax[1].plot(f,-np.angle(Z,deg=True),label=f'Z',zorder=3)
# ax[1].plot(f,-np.angle(Z1,deg=True),label=f'Z',zorder=3)
ax[1].plot(f_pi[:,0],f_pi[:,2],label=f'Z',zorder=3)
ax[1].plot(f_fdcm[:,0],f_fdcm[:,2],label=f'Z',zorder=3,ls='--')

for i in range(2):
    ax[i].axvline(50,color='k',lw=0.75)
    ax[i].grid()
    ax[i].set_xlim(0,2e3)
ax[0].legend(fontsize=8,loc='upper right')
ax[0].set_ylim((0,max(abs(Z))*1.1))
ax[1].set_ylim((-90,90))
ax[1].set_yticks(np.arange(-90,90+30,30))

ax[0].set_ylabel('Magnitude [$\\Omega$]')
ax[1].set_ylabel('Phase [deg]')
ax[1].set_xlabel('$f$ [Hz]')

plt.rcParams['text.usetex'] = True
# plt.savefig(f'{plot_path}\\frequency_response_comp.pdf')
#%%

df = pd.DataFrame({'f':f,
                   'mag':abs(Z),
                   'rad':np.angle(Z),
                   'real':Z.real,
                   'imag':Z.imag,
                   }).set_index('f')

df.to_csv(f'C:\\Users\\bvilm\\PycharmProjects\\SITOBB\\data\\freq\\cable_1C_freq_calc.txt',header=True,index=True)

# ax.set(yscale='log')

