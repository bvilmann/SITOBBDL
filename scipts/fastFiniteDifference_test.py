# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:04:55 2023

@author: bvilm
"""

import numpy as np
import copy
def FastFiniteDifferenceDerivatives(fun, x, epsilon, *funargs, order=2):
    # TODO: Implement list of epsilons for parameter sensitivity selectibility
    
    x = copy.deepcopy(x)
    
    # Evaluate function
    f = fun(x, *funargs)
    
    print(f)
    
    # Dimensions
    nx = x.size
    nf = f.size
    
    ## First order derivatives
    dfFD = np.zeros((nf, nx))
    for j in range(nx):
        # Pertubation
        x[j] = x[j] + epsilon
        
        # Perturbed function evaluation
        fp = fun(x, *funargs)
        
        # Approximation
        # dfFD[:, j] = (fp.ravel(order='F') - f.ravel(order='F')) / epsilon
        # dfFD[:, j] = (fp.ravel(order='F') - f.ravel(order='F')) / epsilon
        dfFD[:, j] = (fp.ravel(order='F') - f.ravel(order='F')) / epsilon
        
        # Remove pertubation
        x[j] = x[j] - epsilon
    
    ## Second order derivatives
    if order == 2:
        epssq = epsilon**2
        d2fFD = np.zeros((nx, nx, nf))
        for j in range(nx):
            # Pertubation
            x[j] = x[j] + 2*epsilon
                
            # Perturbed function evaluation
            fpp = fun(x, *funargs)
                
            # Pertubation
            x[j] = x[j] - epsilon
    
            # Perturbed function evaluation
            fpz = fun(x, *funargs)
            
            # Approximation
            d2fFD[j, j, :] = (fpp.ravel(order='F') - 2*fpz.ravel(order='F') + f.ravel(order='F')) / epssq
            
            # Reset pertubation
            x[j] = x[j] - epsilon
            
            for k in range(j):
                # Pertubation
                x[j] = x[j] + epsilon
                x[k] = x[k] + epsilon
                
                # Perturbed function evaluation
                fpp = fun(x, *funargs)
                
                # Reset pertubation
                x[k] = x[k] - epsilon
                 
                # Perturbed function evaluation
                fpz = fun(x, *funargs)
                
                # Pertubation
                x[k] = x[k] + epsilon
                x[j] = x[j] - epsilon
                 
                # Perturbed function evaluation
                fzp = fun(x, *funargs)
                
                # Approximation
                d2fFD[k, j, :] = d2fFD[j, k, :] = (fpp.ravel(order='F') - fpz.ravel(order='F') - fzp.ravel(order='F') + f.ravel(order='F')) / epssq
                
                # Reset pertubation
                x[k] = x[k] - epsilon
    if order == 2:
        return f, dfFD, d2fFD
    else:
        return f, dfFD

def FiniteDifferenceTestFunction(x):
    # Unpack argument
    x1, x2 = x[0], x[1]

    # Evaluate function
    f = np.array([[np.sin(x1)*np.sin(x2), np.cos(x1)*np.sin(x2)],
                  [x1**2*x2, x1*np.log(x2)]])

    # Evaluate first order derivatives
    df = np.array([[np.cos(x1)*np.sin(x2), np.sin(x1)*np.cos(x2)],
                   [2*x1*x2, x1**2],
                   [-np.sin(x1)*np.sin(x2), np.cos(x1)*np.cos(x2)],
                   [np.log(x2), x1/x2]])

    # Evaluate second order derivatives
    d2f = np.zeros((2, 2, 4))
    d2f[:, :, 0] = np.array([[-np.sin(x1)*np.sin(x2), np.cos(x1)*np.cos(x2)],
                              [np.cos(x1)*np.cos(x2), -np.sin(x1)*np.sin(x2)]])
    d2f[:, :, 1] = np.array([[2*x2, 2*x1],
                              [2*x1, 0]])
    d2f[:, :, 2] = np.array([[-np.cos(x1)*np.sin(x2), -np.sin(x1)*np.cos(x2)],
                              [-np.sin(x1)*np.cos(x2), -np.cos(x1)*np.sin(x2)]])
    d2f[:, :, 3] = np.array([[0, 1/x2],
                              [1/x2, -x1/x2**2]])

    return f, df, d2f

#%% make function callable
def fun(x):
    x1,x2 = x[0], x[1]
    evalf = np.array([[np.sin(x1)*np.sin(x2), np.cos(x1)*np.sin(x2)],
                  [x1**2*x2, x1*np.log(x2)]])
    return evalf

x = np.array([np.pi,1])

epsilons = [float(f'{j}e{i}') for i in range(-9,2) for j in range(1,11)]
epsilons.sort()
print(epsilons)
fs = np.zeros((4,1,len(epsilons)))
dfs = np.zeros((4,2,len(epsilons)))
d2fs = np.zeros((4,4,len(epsilons)))

errs = np.zeros((8,len(epsilons)))
errs2 = np.zeros((16,len(epsilons)))

#%%
f,df,d2f = FiniteDifferenceTestFunction(x)

print(f,df,d2f,sep='\n')
for i, eps in enumerate(epsilons):
    print(round((i+1)/len(epsilons),2),eps)
    
    F, dfFD, d2fFD = FastFiniteDifferenceDerivatives(fun, x, eps, order=2)
    errs[:,i] = dfFD.ravel(order='C') - df.ravel(order='C')
    errs2[:,i] = d2fFD.ravel(order='C') - d2f.ravel(order='C')

#%%


#%%
F, dfFD, d2fFD = FastFiniteDifferenceDerivatives(fun, x, 1e-5, order=2)
for i in range(4):
    print('',f'{i}: analytic, ffd',d2f[:,:,i],d2fFD[:,:,i],sep='\n')

#%%
for i in range(2):
    print('',f'{i}: analytic, ffd',df[:,i],dfFD[:,i],sep='\n')

#%%
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,3,dpi=200,figsize=(12,4))
n = 4 
ax[2].scatter(epsilons,errs.max(axis=0),color='b',label='$1^{st}$ derivative',marker='o',zorder=3)
ax[2].scatter(epsilons,errs2.max(axis=0),color='r',label='$2^{nd}$ derivative',marker='o',zorder=3)

for i in range(16):
    ax[1].scatter(epsilons,abs(errs2[i,:]),label=f'$\\varepsilon$[{i}]',lw=1,marker='x',zorder=3)
for i in range(n*2):
    ax[0].scatter(epsilons,abs(errs[i,:]),label=f'$\\varepsilon$[{i}]',lw=1,marker='x',zorder=3)

# ax[0].set_yticks([-2*1e-4,-1*1e-4,0,1*1e-4,2*1e-4])
# ax[1].set_yticks([-1e6,-10e3,0,10e3,1e6])

ax[0].set(ylabel='$|$Error$|$')
for i in range(3):
    ax[i].set(xscale='log',yscale='log',xlabel='$\\varepsilon$')
    # ax[i].set(xscale='log')
    ax[i].grid()
    ax[i].legend(ncol=4,loc='upper left',fontsize=5.5)
    if i <= 1:
        ax[i].axhline(np.finfo(np.float64).eps,color='k',ls='--')
        ax[i].text(epsilons[0],np.finfo(np.float64).eps,'64-bit machine precision',fontsize=6,ha='left',va='bottom',color=(0.2,0.2,0.2))


ax[0].set_title('Individual $1^{st}$ order errors',fontsize=10)
ax[1].set_title('Individual $2^{nd}$ order errors',fontsize=10)
ax[2].set_title('Maximum errors',fontsize=10)

# ax[0].set_xlim(1e-7,1e-6)
path = r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img'

# plt.savefig(f'{path}\\tol_ind_test_.pdf')



