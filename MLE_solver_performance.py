# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:02:40 2023

@author: bvilm
"""
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
import datetime


I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02

t1 = 0
t2 = 0.025
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
r123 = np.array([1,1,1])*1e-4

R0=np.diag(r123)*1e3
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1e1


#%%
methods = ['Nelder-Mead' ,'Powell' ,'CG' ,'BFGS' ,'Newton-CG' ,'L-BFGS-B' ,'TNC' ,'COBYLA' ,'SLSQP' ,'trust-constr','dogleg' ,'trust-ncg' ,'trust-exact' ,'trust-krylov']
jacs = ['2-point','3-point','cs',None]

data = {}
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}

m = SITOBBDS()
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

d = {k:[] for k in m.p.params.keys()}
d['solver'] = []
d['jac'] = []
d['err_norm'] = []
d['dt'] = []

for jac in jacs:
    for method in methods:
        print(method,jac)
        opts = {'gradient':jac,'method':method}
        m = SITOBBDS(opts=opts)
        m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
    
        # Create input
        u, uk = m.create_input(t1, t2, dt,mode='sin')        
        Sx = m.create_noise(t1, t2, dt,amp=.01,dim=3,seed=1234)         
        
        # Get matrices
        Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
        
        # --------------- GET GROUND TRUTH --------------- 
        # Simulate the system
        x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx)
    
        # 
        dt1_ = datetime.datetime.now()
        theta = np.array(list(m.p.params.values()))
        nans = [np.nan for t in theta]
        try:
            thetahat, thetahat0, Ahat = m.ML_opt_param(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=0,opt_in='single',log=True)
        except Exception as e:
            print(method, e)
            data[method] = {'thetahat':nans,'Ahat':np.nan,'dt' : np.nan,'err':nans,'err_norm':np.nan}
            continue
                    
        dt2_ = datetime.datetime.now()
        dt_ = (dt2_-dt1_).total_seconds()
        err = theta-thetahat
        err_norm = np.linalg.norm(err)
        data[method] = {'thetahat':thetahat,'Ahat':Ahat,'dt' : (dt2_-dt1_).total_seconds(),'err':err,'err_norm':np.linalg.norm(err)}

        # Append data
        for i, k in enumerate(m.p.params.keys()):
            d[k].append(thetahat[i])
        d['solver'].append(method)
        d['jac'].append(jac)
        d['err_norm'].append(err_norm)
        d['dt'].append(dt_)

    # # ================= PLOT =================
    # fig, ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
        
    # sets = ("True",method)
    # d = {}
    # d['True'] = m.p.params.values()
    # d[method] = thetahat
    # # data['MLE_all'] = thetahat_param_m
    
    # labs = list(m.p.params.keys())
    # y = list(m.p.params.values())
    
    # keys = list(m.p.params.keys())
    # x = np.arange(len(keys))  # the label locations
    # width = 0.25  # the width of the bars
    # multiplier = 0
    
    # for i, (k,v) in enumerate(d.items()):
    #     offset = width * multiplier + 0.125
    #     rects = ax.bar(x + offset, v, width, label=k,zorder=3)
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    
    # ax.set_xticks(x + width, keys)
    # ax.legend(loc='upper left', ncols=1)
    
    # ax.set(yscale='log')
    # ax.grid()
    
#%%
df = pd.DataFrame(d)
df.to_excel('data/solver/solver_data_noise.xlsx',header=True,index=False)

tab1 = pd.pivot(df,index='solver',columns='jac',values='err_norm')

tab2 = pd.pivot(df,index='solver',columns='jac',values='dt')

#%% DATA PLOT
dt = []
err = []
for method in methods:
    try:
        dt.append(data[method]['dt'])
        
        err.append(data[method]['err_norm'])
    except:
        err.append(np.nan)
        continue

df = pd.DataFrame({'err':err,'dt':dt},index=methods)


fig, ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
    
sets = ("True",method)
d = {}
d['True'] = m.p.params.values()
d[method] = thetahat
# data['MLE_all'] = thetahat_param_m

labs = list(m.p.params.keys())
y = list(m.p.params.values())

keys = list(m.p.params.keys())
x = np.arange(len(keys))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for i, (k,v) in enumerate(d.items()):
    offset = width * multiplier + 0.125
    rects = ax.bar(x + offset, v, width, label=k,zorder=3)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_xticks(x + width, keys)
ax.legend(loc='upper left', ncols=1)

ax.set(yscale='log')
ax.grid()


