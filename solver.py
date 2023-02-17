import numpy as np
from scipy.integrate import solve_ivp
import control as ctrl
import matplotlib.pyplot as plt

class Solver:

    def __init__(self,
                 A,
                 B,
                 C=None,
                 D=None,
                 t0 = 0,
                 t_end = 0.5,
                 dt = 1e-5,
                 solver = 'dopri5', # dopri5 -> Runge-Kutta, 'RK45'
                 ):
        # Creating time series
        self.t0 = t0
        self.t_end = t_end
        self.dt = dt
        self.t = np.arange(t0, t_end + dt, dt)

        # ODE solver
        self.solver = solver

        # Defining formats/shapes
        self.n = n = A.shape[0] # num_states

        # Defining matrices
        self.A = A
        self.B = B
        if C is None:
            self.C = np.ones(n)
        else:
            self.C = C
        if D is None:
            self.D = 0
        else:
            self.D = D

        # Defining formats/shapes
        # self.n = n = A.shape[0] # num_measurements
        self.m = m = self.C.shape[0] # num_outputs

        return

    def kalman_filter(self,y, u, x0, P0, Q, R):
        """
        Implements the Kalman filter for state estimation given state space matrices and measurements.

        Args:
            A (ndarray): State transition matrix
            B (ndarray): Input matrix
            C (ndarray): Output matrix
            D (ndarray): Feedthrough matrix
            y (ndarray): Array of measured outputs, with shape (num_measurements, num_outputs)
            u (ndarray): Array of inputs, with shape (num_measurements, num_inputs)
            x0 (ndarray): Initial state estimate, with shape (num_states, 1)
            P0 (ndarray): Initial state covariance matrix, with shape (num_states, num_states)
            Q (ndarray): Process noise covariance matrix, with shape (num_states, num_states)
            R (ndarray): Measurement noise covariance matrix, with shape (num_outputs, num_outputs)

        Returns:
            x_hat (ndarray): Array of estimated states, with shape (num_measurements, num_states)
            y_hat (ndarray): Array of estimated outputs, with shape (num_measurements, num_outputs)
        """
        # Initialize arrays
        num_measurements = y.shape[0]
        num_states = self.A.shape[0]
        num_outputs = self.C.shape[0]
        x_hat = np.zeros((num_measurements, num_states))
        y_hat = np.zeros((num_measurements, num_outputs))

        # Initialize state estimates and covariance
        x_hat[0] = x0.reshape(-1)
        P = P0

        for k in range(num_measurements):
            # Prediction step
            x_hat_priori = self.A @ x_hat[k] + self.B @ u[k]
            P_priori = self.A @ P @ self.A.T + Q

            # Update step
            K = P_priori @ self.C.T @ np.linalg.inv(self.C @ P_priori @ self.C.T + R)
            x_hat[k + 1] = x_hat_priori + K @ (y[k] - self.C @ x_hat_priori - self.D @ u[k])
            P = (np.eye(num_states) - K @ self.C) @ P_priori

            # Compute estimated output
            y_hat[k] = self.C @ x_hat[k + 1] + self.D @ u[k]

        return x_hat, y_hat

    def run_dynamic_simulation(self,t,u, x0=None,method = 'RK45'):

        if x0 is None:
            x0 = np.zeros(self.n)

        ss = ctrl.ss(self.A, self.B, self.C, self.D)

        def f(t, x):
            u_ = np.array([u(t)])

            return self.A @ x + self.B @ u_

        sol = solve_ivp(f, [self.t[0], self.t[-1]], x0, t_eval=self.t, method=method)  #

        return sol, ss
