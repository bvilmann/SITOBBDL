# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:24:27 2023

@author: bvilm
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0    # Length of transmission line
n = 100    # Number of points on the line
length = np.linspace(0,L,n) 
dx = length[1] - length[0]

T = 1.0    # Duration of simulation
m = 10   # Number of time steps
time = np.linspace(0,T,m)
dt = time[1] - time[0]
# dx = L/(n-1)
# dt = T/(m-1)


# Electrical properties
R_, L_ = 0.189787342E-03,0.121654383E-02/(2*np.pi*50)
G_, C_ = 0.000000000E+00,0.533520750E-07/(2*np.pi*50)

# Finite difference coefficients
v = 1/np.sqrt(L_*C_) # velocity
z0 = np.sqrt(L_/C_) # characteristic impedance

# Initial conditions
V = np.zeros((n,m))
I = np.zeros((n,m))

def f(a):
    if a >= -dt and a <= dt:
        f = 1
    else:
        f = 0
    return f


# Main loop
for i, x in enumerate(length):
    for j, t in enumerate(time):
        I[i,j] = f(x/dx - v*t) + f(x/dx+v*t)
        V[i,j] = z0*(f(x/dx - v*t) - f(x/dx+v*t))

# # Apply boundary and initial conditions 
# V[0] = 1.0    # Voltage pulse at t=0
# V[0] = 1.0
# V[n-1] = 0.0
# I[0] = 0.0
# I[n-1] = 0.0
# for j in range(m-1):
#     # Update V and I at interior points
#     for i in range(1,n-1):
#         V[i] = V[i] - a*(I[i+1] - I[i])
#         I[i] = I[i] - b*(V[i+1] - V[i-1])
    
#%% Plotting
fig, ax = plt.subplots(1,1,dpi=200)
for i in range(m):
    
    ax.plot(length, I[:,i]-i, label='Voltage')
    # ax.plot(length, I[:,0], label='Current')
    # ax.legend()
ax.set_xlabel('Distance')
ax.set_ylabel('Amplitude')
plt.show()

