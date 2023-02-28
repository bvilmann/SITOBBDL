# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:53:50 2023

@author: BENVI
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
from control import tf

def vector_to_tf(vector, num_poles, num_zeros):
    print()
    poles = np.array(vector[:num_poles]) + 1j * np.array(vector[num_poles:2*num_poles])
    zeros = np.array(vector[2*num_poles:2*num_poles+num_zeros]) + 1j * (vector[2*num_poles+num_zeros:])
    return tf(np.real(np.poly(zeros)), np.real(np.poly(poles)))

def residual_function(freq, vector, num_poles, num_zeros):
    tf_fit = vector_to_tf(vector, num_poles, num_zeros)
    return np.abs([tf_fit(2j*np.pi*f) for f in freq])

def fit_mvf(freq, data, num_poles, num_zeros, initial_guess):
    def residual_function(*x):
        tf_fit = vector_to_tf(x, num_poles, num_zeros)
        return np.abs([tf_fit(2j*np.pi*f) for f in freq])
    
    vector = np.concatenate((np.real(initial_guess[:num_poles]), np.imag(initial_guess[:num_poles]), 
                             np.real(initial_guess[num_poles:]), np.imag(initial_guess[num_poles:])))
    
    
    # popt, pcov = curve_fit(residual_function, np.array([]), data, p0=vector, method='lm', maxfev=100000)
    popt, pcov = curve_fit(residual_function, np.array([]), data, p0=vector, method='lm', maxfev=100000)

    poles = popt[:num_poles] + 1j * popt[num_poles:2*num_poles]
    
    zeros = popt[2*num_poles:2*num_poles+num_zeros] + 1j * popt[2*num_poles+num_zeros:]
    
    return np.concatenate((np.real(zeros), np.real(poles)))


def mvf(freq, data, poles, zeros):
    num_poles = len(poles)
    num_zeros = len(zeros)
    initial_guess = np.concatenate((np.real(poles), np.real(zeros), np.imag(poles), np.imag(zeros)))
    vector = fit_mvf(freq, data, num_poles, num_zeros, initial_guess)
    return vector_to_tf(vector, num_poles, num_zeros)

zeros_real = np.poly1d([1])
zeros_init = np.poly1d([1])
poles_real = np.poly1d([1,1,1,1,1])
poles_init = np.poly1d([1.1,0.9,0.95,1,1])

G = ctrl.tf(np.array(zeros_real),np.array(poles_real))

freq = np.linspace(0,1e3,num=100)
data = np.array([abs(G(s)) for s in freq])

f = data
s = 1j*freq


residues,d,h,poles_new = mvf(freq,data,poles_init.roots,zeros_init.roots)

print(poles_real.roots)
print(poles_init.roots)
print(poles_new)





