# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:27:29 2023

@author: bvilm
"""

import matplotlib.pyplot as plt
import numpy as np
z = 0.05 + 0.5j
y = 0.001+ 0.02j
gamma = np.sqrt(z*y)

x = np.linspace(-np.pi,np.pi,1000)
cosh = np.cosh(gamma*l)
sinh = np.sinh(gamma*l)

fig, ax = plt.subplots(1,1,dpi=150)
ax.grid()
ax.set(xticks=[np.pi*i for i in np.arange(-1,1+0.5,0.5)])
ax.set(xticklabels=[f'{i}$\\pi$' for i in np.arange(-1,1+0.5,0.5)])
ax.axhline(0,color='k',lw=0.75)
ax.axvline(0,color='k',lw=0.75)
ax.plot(x,np.cosh(x),label='cosh')
ax.plot(x,np.sinh(x),label='sinh')
ax.legend()

fig, ax = plt.subplots(1,1,dpi=150)
x = np.linspace(0,3,1000)

ax.grid()
# ax.set(xticks=[np.pi*i for i in np.arange(-1,1+0.5,0.5)])
# ax.set(xticklabels=[f'{i}$\\pi$' for i in np.arange(-1,1+0.5,0.5)])
ax.axhline(0,color='k',lw=0.75)
ax.axvline(0,color='k',lw=0.75)
ax.plot(x,np.cosh(x),label='cosh')
ax.plot(x,np.sinh(x),label='sinh')
ax.legend(loc='lower right')




