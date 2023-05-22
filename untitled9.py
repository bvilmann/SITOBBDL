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

#%%
f_fdcm = np.genfromtxt(r'data\freq\Harm_Y.out',skip_header=1)
f = rect_form(f_fdcm[:,1],f_fdcm[:,2])
s = f_fdcm[:,0]*1j*2*np.pi
# Get fit, poles, residues, d, and h scalars

ax[0].plot(s.imag/(2*np.pi),abs(f))
ax[1].plot(s.imag/(2*np.pi),np.angle(f,deg=True))


#%%
ax[0].axvline(50,color='k',zorder=-3)
# 
# ax[0].set(xlim=(0,1e4),yscale='log')



