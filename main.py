#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import Solver
from PowerSystem import PS
import control as ctrl
import pandas as pd

#%% SIMULATION SETTINGS
time_step = 10e-6
f = 50
phi= 0
omega = 2*np.pi*f
V = 66e3*np.sqrt(2)

# %% Power system - admittance matrix
ps = PS(f = 50, Vbase = 66e3, Sbase = 100e6)

# Adding lines and transformers (bus1, bus2, r, x)
ps.add_line(1, 2, 0.04, 0.52, 'Za')

# Adding generators
ps.add_gen(1,1)

# Adding loads
ps.add_load(2,'load', complex(.1,.2))
# ps.add_load(2,'load1', complex(.3,.2))

Z, Y, B = ps.build()

#%% LUMPED PARAMETER - SINGLE CABLE (CONDUCTOR ONLY)
# ------ PARAMETERS ------ 
R = 1
L = 1e-3
G = 0.0
C1 = 0.5e-6
C2 = 0.5e-6
Rin = 0.05 # BRK = 0.05
# ------ INPUT FUNCTIONS ------ 
u = lambda t: V*np.sin(omega*t+phi)*(0,1)[t>=0.1]

# ------ MATRICES ------ 
A = np.array([[-R/L, 1/L, -1/L],
                [-1/(C1), -1/(Rin * C1), 0],
                [1/(C2), 0, 0],
              ])

B = np.array([[0], [1 / (Rin * C1)], [0]])


A = np.array([[1, 0, 0],
              [-(1/(L*C1)+1/(L*C2)), R/L, -1/(L*C1*Rin)],
                [-1/(C1), 0, -1/(Rin * C1)],
              ])

B = np.array([[0],[1/(L*C1*Rin)], [1 / (Rin * C1)]])

C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])


# ------ SOLVING ODE/PDE ------ 
sol = Solver(A,B,C=C,t0=0,t_end=0.13,dt = time_step)
res, ss = sol.run_dynamic_simulation(t,u)




#%% DATA READER
rpath = r'C:\Users\BENVI\Documents\validation\PSCAD\DTU projects\HCA\cable_1phase.if15_x86'
fname = r'cable_1phase.infx'
file_addr = f'{rpath}\\{fname}'
dread = DataReader()
_,iabc1 = dread.get_signal('Iabc_b1',file_addr)
_,vabc1 = dread.get_signal('Vabc_b1',file_addr)
t_pscad,vabc2 = dread.get_signal('Vabc_b2',file_addr)



#%% PLOT
xlabs = ['$I_L$ [A]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']

ys = [iabc1,vabc1,vabc2]
fig,ax = plt.subplots(len(res.y),1,sharex=True,dpi=300)
cnt = 0
for i,y in enumerate(res.y):
    print(xlabs[i])
    ax[i].plot(res.t,y/1e3,label='Python',color='teal',alpha=0.5,zorder=4)
    # if i in [0,2]:
    #     ax[i].plot(res1.t,res1.y[cnt]/1000,label='Python1',color='green',alpha=0.5)
    #     cnt += 1
    # if 'V_{c1}' in xlabs[i]:
    #     ax[i].plot(t,[u(x) for x in t])
    ax[i].plot(t_pscad,ys[i],label='PSCAD',color='gold',ls=('-','--')[i>0],alpha=0.75)
    ax[i].set_ylabel(xlabs[i])
    ax[i].grid()
ax[-1].set_xlim(0.0995,0.102)
ax[-1].set_xlim(0.095,0.13)

ax[-1].legend()


# %%
# RLC, with load
A = np.array([[-R/L,-1/(L)],[1/C,1/L]])
B = np.array([[1/L],[0]])
sol1 = Solver(A,B,t0=0,t_end=0.15,dt = time_step)
res1, ss = sol1.run_dynamic_simulation(t,u)
# RLC, with load
# A = np.array([[-R/L,-1/(L)],
#               [1/(2*C2),-1/L]])
# B = np.array([[1/(L)],
#               [0]])

# A = np.array([[1,0],
#               [-1/(L*C1)-1/(L*C2),-1/L]])
# B = np.array([[0],
#               [1/(L*C*Rin)]])

# sol1 = Solver(A,B,t0=0,t_end=0.15,dt = time_step)
# res1, ss = sol1.run_dynamic_simulation(t,u)

#%%
# https://scipy-lectures.org/packages/sympy.html
import sympy as sym

def jacobian(ode:list,var:list,mode='numeric'):
    if len(ode) != len(var):
        raise AttributeError('len(ode) != len(var)')
    if mode == 'numeric':
        M = np.zeros((len(ode),len(var)))        
    elif mode == 'symbolic':
        M = np.empty((len(ode),len(var)), dtype=object)
    else:
        raise KeyError('Mode must be "numeric" or "symbolic"')
        
    print(M.shape)
    for i, o in enumerate(ode):
        for j, v in enumerate(var):
            print(i,j,o,v)
            sol = sym.diff(o,v)
            if sol == np.nan:
                raise ValueError('Could not find solution')
            print(str(sol))
            if mode == 'numeric':
                str(sol).replace('')
                M[i,j] = float()
            elif mode == 'symbolic':
                M[i,j] = sol            
    return M

def init_vars(var):
    return [sym.Symbol(v) for v in var]
    
x, y = init_vars(['x','y'])
odes = [1/2*y-x**1,x**2+y]
var = [x,y]
M = jacobian(odes,var,mode='symbolic')
print(M)

#%%
i,v = init_vars(['i','v'])
odes = [y*v,z*i]
var = [i,v]
A = jacobian(odes,var,mode='numerical')
print(A)

# A = 




