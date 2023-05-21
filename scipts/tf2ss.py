# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:57:56 2023

@author: bvilm
"""

import control as ctrl
import numpy as np

tfs = []

rms = np.random.randn(4)
lms = 1e-3*np.random.randn(4)

for i, (rm,lm) in enumerate(zip(rms,lms)):
    if i == 0:
        tf = ctrl.tf([rm*lm],[rm,lm])
    else:
        tf = tf * ctrl.tf([rm*lm],[rm,lm])
    tfs.append(tf)    

SER = ctrl.tf2ss(tf)

print(SER)


