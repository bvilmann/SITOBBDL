# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:14:27 2023

@author: BENVI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
def rect_form(m,p,deg=True):
    if deg: s = np.pi/180
    
    z = m*(np.cos(p*s)+1j*np.sin(p*s))
    
    return z

fig, ax = plt.subplots(2,1,dpi=200,sharex=True)


#%% CALCULATED FREQUENCY RESPONSE 
f_calc = pd.read_csv(f'data\\freq\\cable_1C_freq_calc.txt',header=0,index_col=0)
f = f_calc
f = rect_form(f_calc.real,f_calc.imag)
s = f_calc.index.values*1j*2*np.pi

ax[0].plot(s.imag,abs(f))
ax[1].plot(s.imag,np.angle(f,deg=True))


#%%
f_fdcm = np.genfromtxt(r'data\freq\Harm_1c_fdcm.out',skip_header=1)
f = rect_form(f_fdcm[:,1],f_fdcm[:,2])
s = f_fdcm[:,0]*1j*2*np.pi
# Get fit, poles, residues, d, and h scalars

ax[0].plot(s.imag,abs(f))
ax[1].plot(s.imag,np.angle(f,deg=True))


#%%
f_pi = np.genfromtxt(r'data\freq\Harm_1c_pi.out',skip_header=1)
f = rect_form(f_pi[:,1],f_pi[:,2])

ax[0].plot(s.imag,abs(f),ls=':')
ax[1].plot(s.imag,np.angle(f,deg=True),ls=':')

#%%
f_pscad_m = np.genfromtxt(r'data\cable_1c\Cable_1C_ym.out',skip_header=1)
f_pscad_p = np.genfromtxt(r'data\cable_1c\Cable_1C_yp.out',skip_header=1)

f = rect_form(f_pscad_m[:,2],f_pscad_p[:,2])
s = f_pscad_m[:,1]*1j*2*np.pi

# plt.plot(s.imag,abs(f))
# ax[0].plot(s.imag,abs(f))
# ax[1].plot(s.imag,np.angle(f,deg=True))

ax[0].axvline(50*np.pi,color='k',zorder=-3)

ax[0].set_xlim(0,1e4)



