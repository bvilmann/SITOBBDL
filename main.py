#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
from dynamic_simulation import Solver
from PowerSystem import PS
import control as ctrl
import pandas as pd

ps = PS(f = 50, Vbase = 66e3, Sbase = 100e6)

# Adding lines and transformers (bus1, bus2, r, x)
ps.add_line(1, 2, 0.04, 0.52, 'Za')

# Adding generators
ps.add_gen(1,1)

# Adding loads
ps.add_load(2,'load', complex(.1,.2))
# ps.add_load(2,'load1', complex(.3,.2))

Z, Y, B = ps.build()

R1, R2, R3, C1, C2, L = 1,10,1,1,1.3,1
Rshnt = 1
# A = np.array([[-1/R1, 0, 0, 0],
#           [0,1,0,0],
#           [0,1/(L*(C1-C2)),R2/(C1-C2),0],
#           [0,0,0,R3/C2]
#           ])
#
# B = np.array([1/C1,0,0,0])

# RLC, no load
A = np.array([[-R1/L,-1/(L)],[1/C1,0]])

# RLC, with load
A = np.array([[-R1/L,-1/(L)],[1/C1,-1/R2]])

B = np.array([[1/L],[0]])

# RLC, load
A = np.array([[-R1/L,1/(L),-1/L],
              [1/C1,-1/Rshnt,0],
              [1/C2,0,-1/R2],
              ])

B = np.array([[0],[1],[0]])

x0 = [0 for i in B]

xlabs = ['$I_L$','$V_s$','$V_r$']
sol = Solver(A,B,xlabels=xlabs)
t,x = sol.run_dynamic_simulation(x0,0,20)

df = pd.DataFrame(index=t,columns=xlabs)

fig,ax = plt.subplots(len(x0),1,sharex=True)
for i,col in enumerate(x0):
    print(xlabs[i])
    ax[i].plot(t,x[i,:])
    ax[i].set_ylabel(xlabs[i])
    ax[i].grid()
    df[xlabs[i]] = x[i,:]
    # df[col].plot()
    # plt.close()

# ss.damp()
# G = ctrl.ss2tf(ss)
# G.step()

