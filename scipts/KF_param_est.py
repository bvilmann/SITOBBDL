# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:47:41 2023

@author: bvilm
"""
import sys
sys.path.insert(0, '../')  # add parent folder to system path

import numpy as np
from numpy.linalg import inv
from sysid_app import SITOBBDS
# from scipy.linalg import inv
from scipy.linalg import expm
import datetime
import matplotlib.pyplot as plt

def kalman_filter(z):
    # Initialize variables
    x = np.array([[0.0], [0.0]])
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[1.0]])
    I = np.identity(2)

    # Iterate over measurements
    for i in range(len(z)):
        # Prediction step
        x = F.dot(x)
        P = F.dot(P).dot(F.T)

        # Update step
        y = z[i] - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (I - K.dot(H)).dot(P)

    return x


def estimate_params_kalman(data, ss_system, initial_state, initial_cov, process_noise_cov, measurement_noise_cov):
    # Extract the relevant matrices from the state space system
    A = ss_system.A
    C = ss_system.C
    Q = ss_system.Q
    R = ss_system.R
    
    # Initialize the state estimate and covariance matrix
    state_estimate = initial_state
    cov_estimate = initial_cov
    
    # Initialize the parameter estimate
    theta = np.zeros(C.shape[1])
    
    # Iterate through the data and update the state and parameter estimates
    for i in range(len(data)):
        # Prediction step
        state_estimate = np.dot(A, state_estimate)
        cov_estimate = np.dot(np.dot(A, cov_estimate), A.T) + process_noise_cov
        
        # Update step
        y = data[i]
        kalman_gain = np.dot(np.dot(cov_estimate, C.T), inv(np.dot(np.dot(C, cov_estimate), C.T) + measurement_noise_cov))
        state_estimate = state_estimate + np.dot(kalman_gain, y - np.dot(C, state_estimate))
        cov_estimate = np.dot(np.eye(A.shape[0]) - np.dot(kalman_gain, C), cov_estimate)
        
        # Estimate parameters using the current state estimate and measurement
        theta += np.dot(np.dot(inv(np.dot(C, cov_estimate)), C), state_estimate - np.dot(A, state_estimate))
        
    # Normalize the parameter estimate by the number of data points
    theta /= len(data)
    
    return theta


def kalman_parameter_estimation(X, Y, A, C, Q, R, P):
    """
    Estimates the parameters of a linear Gaussian state space model using the Kalman filter.
    
    Args:
    - X (ndarray): Array of shape (n_samples, n_features) containing the input data.
    - Y (ndarray): Array of shape (n_samples, n_outputs) containing the output data.
    - A (ndarray): Array of shape (n_features, n_features) containing the state transition matrix.
    - C (ndarray): Array of shape (n_outputs, n_features) containing the observation matrix.
    - Q (ndarray): Array of shape (n_features, n_features) containing the process noise covariance matrix.
    - R (ndarray): Array of shape (n_outputs, n_outputs) containing the measurement noise covariance matrix.
    - P (ndarray): Array of shape (n_features, n_features) containing the initial state covariance matrix.
    
    Returns:
    - theta_hat (ndarray): Array of shape (n_outputs, n_features) containing the estimated parameters.
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    
    # Initialize the state vector and the state covariance matrix
    x_hat = np.zeros(n_features)
    P_hat = P
    
    # Initialize the Kalman filter
    theta_hat = np.zeros((n_outputs, n_features))
    K = np.zeros((n_features, n_outputs))
    
    for t in range(n_samples):
        # Prediction step
        x_hat = A @ x_hat
        P_hat = A @ P_hat @ A.T + Q
        
        # Update step
        y_t = Y[t]
        C_t = C @ P_hat @ C.T + R
        K_t = P_hat @ C.T @ np.linalg.inv(C_t)
        x_hat = x_hat + K_t @ (y_t - C @ x_hat)
        P_hat = (np.eye(n_features) - K_t @ C) @ P_hat
        
        # Store the estimated parameters
        theta_hat += np.outer(y_t - C @ x_hat, x_hat)
        K += np.outer(C_t @ x_hat, x_hat)
    
    # Normalize the estimated parameters
    theta_hat /= n_samples
    K /= n_samples
    
    # Calculate the optimal parameters
    theta_opt = np.linalg.inv(K) @ theta_hat
    
    return theta_opt.T


def c2d(A,B,C,D,Ts=1):

    # Calculate the continuous-time state transition matrix using the matrix exponential
    At = np.block([[A, B], [np.zeros((B.shape[1], A.shape[1])), np.zeros((B.shape[1], B.shape[1]))]])
    eAt = expm(At * Ts)
    Ad = eAt[:A.shape[0], :A.shape[1]]
    Bd = eAt[:A.shape[0], -1:]          
    
    return Ad, Bd, C, D

def simulate_state_space(A,B,C,D,u,t1,t2,dt,x0=None,Sx=None,Sy=None):


    time = np.arange(t1,t2+dt,dt)

    # creating indices
    n_y = len(time)
    n = A.shape[0]
    m = C.shape[0]

    # Initialize arrays
    x   = np.zeros((n,n_y))
    y   = np.zeros((m,n_y))

    # Handling noise input
    if x0 is None: x0 = np.zeros(n).reshape(-1, 1)  
    if Sx is None: Sx = np.zeros((n,len(time)))
    if Sy is None: Sy = np.zeros((m,len(time)))

    print('',f'Simulating discrete-time system:',sep='\n')
    t1_ = datetime.datetime.now()
    # initial values
    x[:,0] = x0[:,0]
    y[:,0] = C @ x0 # + Sy() # TODO: Add noise, be aware of operator for u[k]
    for k,t in enumerate(time[:-1]):
        # --------------- Solving x[k+1] ---------------
        # Calculate x[:, k + 1]
        x[:,k+1] = A @ x[:,k] + B @ np.array([u[k]]) + Sx[:,k] #+ Sx() # TODO: Add noise, be aware of operator for u[k]
        y[:,k] = C @ x[:,k] + Sy[:,k] # + Sy() # TODO: Add noise, be aware of operator for u[k]
    y[:,-1] = C @ x[:,-1] + Sy[:,-1] # TODO: Add noise, be aware of operator for u[k]
    t2_ = datetime.datetime.now()            
    print(f'Finished in: {(t2_-t1_).total_seconds()} s')

    return x,y

def KF_estimation(A,B,C,D,x0,u,y,R0,R1,R2,t1,t2,dt):

    time = np.arange(t1,t2+dt,dt)

    # creating indices
    n_y = len(time)
    n = A.shape[0]
    m = C.shape[0]

    # Initialize arrays
    eps = np.zeros((m,n_y))
    x_hat_filt   = np.zeros((n,n_y))
    x_hat_pred   = np.zeros((n,n_y))
    y_hat_pred   = np.zeros((m,n_y))
    P_filt       = np.zeros((n,n,n_y))
    P_pred       = np.zeros((n,n,n_y))
    K       = np.zeros((n,n,n_y))
    R       = np.zeros((n,n,n_y))
    
    # 
    u = np.matrix(u)

    # Parameter estimation        
    hx1     = np.zeros((n,n_y))
    hx2     = np.zeros((m,n_y))
    varphi  = np.zeros((2*n,n_y))
    theta = np.zeros((2*n,n_y))

    
    # initial values
    x_hat_pred[:,0] = x0
    P_pred[:,:,0] = R[:,:,0] = R0
    for k,t in enumerate(time):
        # --------------- Parameter estimation part ---------------
        if k >= n:
            phi = x_hat_pred[:,k-1:k-n]
            psi = u[:,k-1:k-n]
            varphi[:,k] = np.vstack([phi,psi])
        
        # --------------- Calculating output ---------------
        # Calculating output and estimated output
        y_hat_pred[:,k] = C @ x_hat_pred[:,k] # + D @ u[k]

        # Calculating error / innovation term
        eps[:,k] = y[:,k] - y_hat_pred[:,k]

        # --------------- Discrete-time Kalman Filter  ---------------
        # Calculate kalman gain
        R[:, :, k] = C @ P_pred[:, :, k] @ C.T + R2 # R2 = Measurement noise covariance matrix.
        K[:,:,k] = P_pred[:,:,k] @ C.T @ inv(R[:,:,k])

        # Measurement update
        x_hat_filt[:,k] = x_hat_pred[:,k] + K[:,:,k] @ eps[:,k]
        P_filt[:,:,k] = (np.eye(n) - K[:,:,k] @ C) @ P_pred[:,:,k]

        # Time update
        if k < len(time)-1:
            # print(A)
            x_hat_pred[:,k+1] = A @ x_hat_filt[:,k] + B @ np.array([u[k]])
            P_pred[:, :, k+1] = A @ P_filt[:,:,k] @ A.T + R1 # TODO: R1 = B @ R1 @ B.T, B is not the input matrix in this context.
    
    return x_hat_pred, y_hat_pred, eps, R


I = lambda n: np.eye(n)
params= {'Rin':100,'V':1,'Vbase':66e3,'Rload': 1e6,'phi':np.pi/4}

dt = 10e-6

t1 = -0.00005
t2 = 0.0010

t1 = -0.00005
t2 = 0.02

t1 = 0
t2 = 0.02
t = np.arange(t1,t2+dt,dt)

x0 = np.zeros(3)

# Noise
r123 = np.array([1,1,1])

R0=np.diag(r123)*1e-1
R1=np.diag(r123)*1e-1
R2=np.diag(r123)*1e-1

#%%

opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)

# Create input

# Create Noise input
# Sx = m.create_noise(t1, t2, dt,amp=0.02,dim=3,seed=1234)         
# Sy = m.create_noise(t1, t2, dt,amp=0.02,dim=3,seed=1235)        

# # Get matrices
# Ad, Bd, A, B, C, D = m.A_d, m.B_d, m.A, m.B, m.C, m.D

# --------------- GET GROUND TRUTH --------------- 5
# Simulate the system
# x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt,Sx=Sx,Sy=Sy)

# Filter the data with the Kalman Filter


# Define the state space system
A = np.array([[0.8, 1], [-0.4, 0]])
B = np.array([1.68, 2.32]).reshape(-1, 1)
C = np.array([1,0]).reshape(1, -1)
D = np.array([0.0])
dt = 1
t1, t2 = 0,500
t = np.arange(t1,t2+dt,dt)
u, uk = m.create_input(t1, t2, dt,mode='step')        
x0 =np.zeros(2)

r123 = np.diag(np.ones(2*A.shape[0]))

R0=np.diag(r123)*1e6
R1=np.diag(r123)*1e6
R2=np.diag(r123)*1e6


x, y = m.simulate(A,B,C,D,x0,uk,t1,t2,dt)

# xhat, yhat, eps, R = m.KalmanFilter(A, B, C, D, x0, uk, y, R0, R1, R2, t1, t2, dt)

plt.plot(t,y.T)

# theta_hat, P_hat = KF_estimation(Ad,Bd,C,D,x0,u,y,R0,R1,R2,t1,t2,dt)
#%%
u, lambd = np.linalg.eig(A)

print(u)
print(lambd)

#%%





#%%


u = uk.reshape(1,-1)
# Function
time = np.arange(t1,t2+dt,dt)

# creating indices
n_y = len(time)
n = A.shape[0]
m = C.shape[0]

# Initialize arrays
eps = np.zeros((m,n_y))
x_hat_filt   = np.zeros((n,n_y))
x_hat_pred   = np.zeros((n,n_y))
y_hat_pred   = np.zeros((m,n_y))
P_filt       = np.zeros((2*n,2*n,n_y))
P_pred       = np.zeros((2*n,2*n,n_y))
K       = np.zeros((n,n,n_y))
R       = np.zeros((2*n,2*n,n_y))

# 
# u = np.matrix(u)

# Parameter estimation        
hx1     = np.zeros((n,n_y))
hx2     = np.zeros((m,n_y))
varphi  = np.zeros((2*n,n_y))
theta = np.zeros((2*n,n_y))


# initial values
x_hat_pred[:,0] = x0
P_pred[:,:,0] = R[:,:,0] = R0
for k,t in enumerate(time):    
    # --------------- Calculating output ---------------
    # Calculating output and estimated output
    y_hat_pred[:,k] = C @ x_hat_pred[:,k] # + D @ u[k]

    # Calculating error / innovation term
    eps[:,k] = y[:,k] - y_hat_pred[:,k]

    # --------------- Discrete-time Kalman Filter  ---------------
    # Calculate kalman gain
    R[:, :, k] = C @ P_pred[:, :, k] @ C.T + R2 # R2 = Measurement noise covariance matrix.
    # K[:,:,k] = P_pred[:,:,k] @ C.T @ inv(R[:,:,k])
    K[:,:,k] = P_pred[:,:,k] @ (C @ inv(R[:,:,k])).T

    # Measurement update
    x_hat_filt[:,k] = x_hat_pred[:,k] + K[:,:,k] @ eps[:,k]
    P_filt[:,:,k] = (np.eye(n) - K[:,:,k] @ C) @ P_pred[:,:,k]

    # Time update
    if k < len(time)-1:
        # print(A)
        x_hat_pred[:,k+1] = A @ x_hat_filt[:,k] + B @ u[:,k]
        P_pred[:, :, k+1] = A @ P_filt[:,:,k] @ A.T + R1 # TODO: R1 = B @ R1 @ B.T, B is not the input matrix in this context.

    # --------------- ES-RLS - Parameter estimation ---------------
    if k > n:
        phi = x_hat_pred[0,k-n:k][::-1]
        # phi = x_hat_pred[:,k-][::-1]
        psi = u[:,k-n:k][::-1,:]
        # psi = u[:,k][::-1]
        varphi = np.vstack([phi,psi,np.zeros((n-u.shape[0],n))])
        
        theta[:,k+1] = theta[:,k] + K[:,:,k] @ (y[:,k] - varphi.T @ theta[:,k])

# Print the estimated parameter coefficients
print('Estimated parameter coefficients:')
print(theta)


