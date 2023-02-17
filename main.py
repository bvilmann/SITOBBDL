#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import Solver
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os

#%% Functions
# d = {
#      '':{'t':res.y,
#          'y':res.t,
#          '':,
#          },
#      '':{'':,
#          '':,
#          },
#      }

xlabs = ['$I_L$ [A]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']
def plot_sols(sols,pscad:list=None,labs=None,names:list=None,t_pscad=None,clr_pscad='gold'):
    
    # Axis handling
    if pscad is not None:
        
        N = len(pscad)
    else:        
        N = max([r.y.shape[0] for r in res])

    # Axis handling
    fig,ax = plt.subplots(N,1,sharex=True,dpi=300)

    # Axis handling
    for i, r in enumerate(res):
        for j, y in enumerate(r.y):
            if i == 0:
                ax[j].set_ylabel(('',labs[j])[labs is not None])
                ax[i].grid()
            ax[j].plot(r.t,y/1e3,label=('',),color='teal',alpha=0.5,zorder=4)
    
    # plotting pscad
    if pscad is not None:
        for j, y in enumerate(pscad):
            ax[j].plot(t_pscad,y,label='PSCAD',color=clr_pscad,alpha=0.5,zorder=5)
            
    ax[-1].legend()
    
    return

plot_sols([res_1_1,res_1_2],pscad[iabc1,vabc1,vabc1])


def plot(name=None):

    xlabs = ['$I_L$ [A]','$V_{c1}$ [kV]','$V_{c2}$ [kV]']
    
    ys = [iabc1,vabc1,vabc2]
    fig,ax = plt.subplots(len(res.y),1,sharex=True,dpi=300)
    cnt = 0
    for i,y in enumerate(res.y):
        print(xlabs[i])
        ax[i].plot(res.t,y/1e3,label='Python',color='teal',alpha=0.5,zorder=4,lw=2)
        if i in [0,2]:
            # if 'V_{c1}' in xlabs[i]:
            #     ax[i].plot(t,[u(x)/1000 for x in t],label='Python1',color='blue',alpha=0.5)
    
            ax[i].plot(res_2_1.t,res_2_1.y[cnt]/1000,label='Python1',color='blue',alpha=0.5)
            cnt += 1

        
        ax[i].plot(t_pscad,ys[i],label='PSCAD',color='gold',ls=('-','--')[i>0],alpha=0.75,zorder=5)
        ax[i].set_ylabel(xlabs[i])
        ax[i].grid()
    ax[-1].set_xlim(0.0995,0.102)
    ax[-1].set_xlim(0.095,0.13)
    
    ax[-1].legend()
    ax[-1].set_xlabel('Time [s]')
    if name is not None:
        plt.savefig(r'C:\Users\bvilm\Dropbox\Apps\Overleaf\Special course - System identification of black-box dynamical systems\img\sim'+f'\\{name}.pdf')
    plt.show()
    plt.close()
    return fig, ax


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
C1 = 0.5e-6
C2 = 0.5e-6
Rin = 0.05 # BRK = 0.05

# ------ INPUT FUNCTIONS ------ 
u = lambda t: V*np.sin(omega*t+phi)*(0,1)[t>=0.1]

name = '1ph_c_1'

# ------ MATRICES ------ 
# 0.1) Current half the magnitude
A = np.array([[-R/L, 1/L, -1/L],
                [-1/(C1), -1/(Rin * C1), 0],
                [1/(C2), 0, 0],
              ])

B = np.array([[0], [1 / (Rin * C1)], [0]])

sol_1_1 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
res_1_1, _ = sol_1_1.run_dynamic_simulation(u)

plot()
# 0.2) Double capacitance

A = np.array([[-R/L, 1/L, -1/L],
                [-1/(C1), -1/(Rin * C1), 0],
                [1/(C1+C2), 0, 0],
              ])

B = np.array([[0], [1 / (Rin * C1)], [0]])

sol_1_2 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
res_1_2, _ = sol_1_2.run_dynamic_simulation(u)


# 2) Oversize transient but matching current, Vc2 is off.
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 0],
#                 [1/(1*C2), 0, -1/L],
#               ])

# B = np.array([[1/L], [1 / (Rin * C1)], [L/(R*C2)]])

# 3) Perfect current, Vc2 is off
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), (C2)/(R*L)],
#                 [1/(C2), (C1)/(R*L), -1/L],
#               ])

# B = np.array([[1/L], [1 / (Rin * C1)], [L/(R*C2)]])

# # 4) Handling coupling during the way.
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 1/(R*C2)],
#                 [1/(C2), (1)/(R*C1), -1/(R*C2)],
#               ])

# B = np.array([[R/L], [1 / (Rin * C1)], [1/((R+Rin)*C2)]])

# # 5) Correct coupling
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 1/(L*R*C2)],
#                 [1/(C2), 1/(L*R*C1), -R/(L*C2)],
#               ])

# B = np.array([[1/L], [1 / (Rin * C1)], [R/(L*C2)]])

# # 1.1) Added additional input to current
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 0],
#                 [1/(C2), 0, 0],
#               ])

# B = np.array([[1/L], [1 / (Rin * C1)], [0]])

# # 1.2) Nre approach for solving eq 3. No current
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 0],
#                 # [1/(C2), 0, -1/(Rin * C1)],
#                 [0, (1-1/(C1*Rin)), -1],
#               ])

# B = np.array([[0], [1 / (Rin * C1)], [1/(C1*Rin)]])

# # 1.3) Current half the magnitude
# A = np.array([[-R/L, 1/L, -1/L],
#                 [-1/(C1), -1/(Rin * C1), 0],
#                 # [1/(C2), 0, -1/(Rin * C1)],
#                 # [L/(R*C1), (1-1/(C1*Rin)), -1],
#                 [L/(R*C1), 0, 0],
#               ])

# B = np.array([[0], [1 / (Rin * C1)], [0]])


# 0) Coupled ODEs of I_L
# A = np.array([[0,1,0],
#               [0,0,1],
#               [1/(L*C1**2*R), -1/(L*C1)-1/(L*C2), -R/L],
#               ])

# B = np.array([[0], [0], [-1/(L*C1**2*Rin**2)+1/(L*C1*Rin)]])


# ------ SOLVING ODE/PDE ------ 

sol = Solver(A,B,C=C,t0=0.05,t_end=0.13,dt = time_step)
res, ss = sol.run_dynamic_simulation(u)



#%% DATA READER

rpath = f'{os.getcwd()}\\data'
fname = r'cable_1phase.infx'
file_addr = f'{rpath}\\{fname}'
dread = DataReader()
_,iabc1 = dread.get_signal('Iabc_b1',file_addr)
_,vabc1 = dread.get_signal('Vabc_b1',file_addr)
t_pscad,vabc2 = dread.get_signal('Vabc_b2',file_addr)



# %%
# sols = 

# RLC, with load
# x = [di/dt, dvc2/dt]
name = '1ph_c_1'

# 1) Match no resonant coupling
A = np.array([[-R/L,-1/(L)],
              [1/(2*C2),0]])

B = np.array([[1/L],[0]])

sol_2_1 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
res_2_1, _ = sol_2_1.run_dynamic_simulation(u)

# # 2) Match resonant coupling
A = np.array([[-R/L,-1/(L)],
              [1/(2*C2),-1/L]])
B = np.array([[1/L],[L/(R*2*C2)]])


sol_2_2 = Solver(A,B,t0=0,t_end=0.13,dt = time_step)
res_2_2, _ = sol_2_2.run_dynamic_simulation(u)

# # # 3) Match
# A = np.array([[-R/L,-1/(L)],
#               [1/(2*C2),0]])

# B = np.array([[1/L],[L/(R*2*C2)]])


# plot_sols()
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




