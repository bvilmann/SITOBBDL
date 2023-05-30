# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:38:22 2022

@author: bvilm
"""

import PowerFlow_46710 as pf # import Power Flow functions
import LoadNetworkData
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import conj
import networkx as nx

def cmplx(mag, ang, degrees:bool = True):
    phi = ang*(1,np.pi/180)[degrees]
    return mag * (np.cos(phi) + 1j*np.sin(phi))

#%%    

# settings
max_iter = 30 # Iteration settings
err_tol = 1e-4
n_pos = {1:(0,0),
         2:(1,1),
         3:(4,1),
         4:(5,0),
         5:(4,-1),
         6:(1,-1),
         7:(2.5,0),
         }
load_from_estimation = False
# Load the Network data ...
lnd = LoadNetworkData.LoadNetworkData()

# Adding lines and transformers (bus1, bus2, r, x, g, b)
# if load_from_estimation:
#     for i, (node1, node2, d) in enumerate(G.edges.data()):
#         lnd.add_line(node1,node2,r[i],x[i],b=b[i],name=f'Z{i}')

# else:
#     lnd.add_line(1,2,0.04,0.52,name='Za',b=0.02)
#     lnd.add_line(2,3,0.03,0.47,name='Zb',b=0.02)
#     lnd.add_line(3,4,0.05,0.42,name='Zps',b=0.02)
#     lnd.add_line(1,7,0.04,0.43,name='Zc',b=0.02)
#     lnd.add_line(4,7,0.01,0.5,name='Zd',b=0.02)
#     lnd.add_line(1,6,0.04,0.46,name='Ze',b=0.02)
#     lnd.add_line(5,6,0.03,0.45,name='Zf',b=0.05)
#     lnd.add_line(4,5,0.04,0.45,name='Zg',b=0.05)

lnd.add_line(1,2,0.04,0.52,name='Za',b=0.02)
# lnd.add_transformer(3,4,0,0.05,n=1,alpha=phi,name='Zps')

# Adding loads and generators
lnd.add_gen(1,0,0,v=1,theta=0)
lnd.add_load(2,0.42,0.05)
# lnd.add_gen(4,0.9,0,v=1)
# lnd.add_load(3,0,0)
# lnd.add_load(5,0.43,0.06)
# lnd.add_load(6,0.48,0.05)
# lnd.add_load(7,0.41,0.04)

lnd.build()
lnd.draw()

V,success,n, J, F = pf.PowerFlowNewton(lnd.Ybus,lnd.Sbus,lnd.V0,lnd.pv_index,lnd.pq_index,max_iter,err_tol)

# Display results if the power flow analysis converged
if success:
    bus, branch = pf.DisplayResults(V,lnd)
    system = pf.getSystem(V,lnd)

    
    
    G, fig, ax = pf.plot_pf(bus,branch,n_pos)
    fig.tight_layout()
    plt.show()
    # plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A1\\img\\pf_graph_{phi}.pdf')

plt.close()


losses = abs(abs(branch['From Bus Inj.','P']) - abs(branch['To Bus Inj.','P']))


#%%
# Get undirected graph
G = G.to_undirected()

# Populate graph with needed data

for i,row in bus.iterrows():
    G.nodes[i]['V'] = cmplx(row['Voltage','Magn.'],row['Voltage','Angle'])
    # print(i,row['Voltage','Magn.'],row['Voltage','Angle'],G.nodes[i]['V'])
    G.nodes[i]['Vr'] = G.nodes[i]['V'].real
    G.nodes[i]['Vi'] = G.nodes[i]['V'].imag
    G.nodes[i]['id'] = i
    
for i, row in branch.iterrows():
    line = G.edges[row['From','Bus'],row['To','Bus']]
    line['k'] = int(row['From','Bus'])
    line['m'] = int(row['To','Bus'])
    
    # Get power flow
    line['Sk'] = row['From Bus Inj.','P'] + 1j*row['From Bus Inj.','Q']
    line['Sm'] = row['To Bus Inj.','P'] + 1j*row['To Bus Inj.','Q']
    # line['Sk'] = line['S_in']
    # line['Sm'] = line['S_out']

    # Calculate currents    
    line['Ik'] = conj(line['Sk']/G.nodes[line['k']]['V'])
    line['Im'] = conj(line['Sm']/G.nodes[line['m']]['V'])
    # line['Ik'] = (line['Sk']/G.nodes[line['k']]['V'])
    # line['Im'] = (line['Sm']/G.nodes[line['m']]['V'])
    line['Ik,re'] = line['Ik'].real
    line['Im,re'] = line['Im'].real
    line['Ik,im'] = line['Ik'].imag
    line['Im,im'] = line['Im'].imag

    # Get voltages
    line['Vk']      = G.nodes[line['k']]['V']
    line['Vk,re']   = G.nodes[line['k']]['Vr']
    line['Vk,im']   = G.nodes[line['k']]['Vi']
    line['Vm']      = G.nodes[line['m']]['V']
    line['Vm,re']   = G.nodes[line['m']]['Vr']
    line['Vm,im']   = G.nodes[line['m']]['Vi']

    line['dV']      = line['Vk'] - line['Vm']
    line['dVr']      = line['Vk,re'] - line['Vm,re']
    line['dVi']      = line['Vk,im'] - line['Vm,im']

    # Get impedances
    line['Z'] = lnd.G.edges[line['k'],line['m']]['z']
    line['Y'] = lnd.G.edges[line['k'],line['m']]['yc']
    # print(line)
    
    
#%%
def get_measurements(G):
    N = len(G.nodes)
    M = len(G.edges)
    z = np.zeros((N*2+M*4,1))
    i = 0
    
    for (node, d) in (G.nodes.data()):
        n_idx = list(G.nodes).index(node)
        z[d['id'] -1] = d['Vr']
        z[d['id'] + n -1] = d['Vi']
        
        i += 1

    i = 2*n
    for (node1, node2, d) in (G.edges.data()):
        z[i+M*0] = d['Ik,re']
        z[i+M*1] = d['Ik,im']
        z[i+M*2] = d['Im,re']
        z[i+M*3] = d['Im,im']
    
        i += 1

    return z

def get_initial_conditions(G,r1=1e0,r2=1e3):
    N = len(G.nodes)
    M = len(G.edges)

    n_states = N*2+M*4
    H = np.zeros((n_states,2*N))

    x0 = np.concatenate([np.ones(N),np.zeros(N )]) # + 4*M
    R0 = np.eye(n_states)
    
    # 
    R1 = np.eye(2*N)*r1
    R2 = np.eye(n_states)*r2
    
    return x0, R0, R1, R2

def initialize_state_vector(G: nx.Graph):
    # Create a list for storing voltage angles and magnitudes
    voltage_angles = []
    voltage_magnitudes = []

    # For each bus (node) in the graph
    for _, node_data in G.nodes(data=True):
        # Get the complex voltage
        V = node_data.get('V', None)
        if V is not None:
            # Append the angle and magnitude to the respective lists
            voltage_angles.append(np.angle(V))
            voltage_magnitudes.append(np.abs(V))

    # Combine the angle and magnitude lists to form the state vector
    state_vector = np.concatenate([voltage_angles[1:], voltage_magnitudes])  # angle of reference bus (bus 1) is excluded

    return state_vector

def measurement_model():

    return

def jacobian(x, h):
    # Calculate the number of state variables and measurements
    n_states = len(x)
    n_meas = len(h)

    # Initialize an empty Jacobian matrix
    H = np.zeros((n_meas, n_states))

    # Fill the Jacobian matrix
    for i in range(n_meas):
        for j in range(n_states):
            # Placeholder for the partial derivative calculation
            # You would replace this with the actual calculation for your system
            H[i, j] = 0.0  # Replace with actual partial derivative calculation

    return H

def cost_function(z, h_x, W):
    # Calculate the difference between the actual and predicted measurements
    diff = z - h_x

    # Calculate the cost function
    J = diff.T @ W @ diff

    return J

x = initialize_state_vector(G)

#%%
import numpy as np

def update_state(x, H, W, z_hx, tol=1e-6, max_iter=100):
    """
    This function implements the state update process, including state correction and convergence check.

    Parameters:
    x: Current state vector
    H: Jacobian matrix
    W: Weight matrix
    z_hx: Difference between actual and predicted measurements
    tol: Convergence tolerance
    max_iter: Maximum number of iterations

    Returns:
    x: Updated state vector
    converged: Whether the solution has converged
    """
    # Initialize convergence flag
    converged = False

    for _ in range(max_iter):
        # 
        
        # Calculate the correction vector
        delta_x = np.linalg.inv(H.T @ W @ H) @ (H.T @ W @ z_hx)

        # Update the state estimate
        x += delta_x

        # Check for convergence
        if np.linalg.norm(delta_x) < tol:
            converged = True
            break

    return x, converged



