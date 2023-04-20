# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:45:35 2023

@author: bvilm
"""

import numpy as np
from numpy.linalg import eigvals
from scipy import linalg


def c2d(A,B,dt=10e-6):
    At = np.block([[A, B], [np.zeros((1, A.shape[1])), np.zeros((1, 1))]])½
    eAt = linalg.expm(At * dt)
    Ad = eAt[:A.shape[0], :A.shape[1]]
    Bd = eAt[:A.shape[0], -1:]         
    return Ad, Bd


A = abs(np.random.randn(3,3)) + np.random.randn(3,3)*1j*np.where(np.random.randn(3,3)>0,1,0) - 5*np.diag(np.ones(3))

print(A)

A_ = np.vstack((np.hstack((A.real, -A.imag)),
               np.hstack((np.imag(A), np.real(A)))))

# _0 = np.zeros(A.shape)

print(A_)

A_eig = eigvals(A)
# A__eig = eigvals(A_)

print(A_eig)
# print(A__eig)


Ad, Bd = c2d(A,B)


A__ = np.concatenate((np.real(A), np.imag(A)))

print(A__)
