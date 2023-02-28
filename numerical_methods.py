import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.signal import cont2discrete
import scipy

class NumericalMethods:

    def __init__(self):

        return

    def c2d(self,A, B, C, D, Ts,domain='t'):
        """
        Discretize a continuous-time dynamical system represented by state equations (A, B, C, D matrices)
        using the zero-order hold method.

        Parameters:
        - A: numpy array, the state matrix.
        - B: numpy array, the input matrix.
        - C: numpy array, the output matrix.
        - D: numpy array, the feedforward matrix.
        - Ts: float, the sampling period.

        Returns:
        - Ad: numpy array, the discrete-time state matrix.
        - Bd: numpy array, the discrete-time input matrix.
        - Cd: numpy array, the discrete-time output matrix.
        - Dd: numpy array, the discrete-time feedforward matrix.
        """

        # Calculate the continuous-time state transition matrix using the matrix exponential
        At = np.block([[A, B], [np.zeros((1, A.shape[1])), np.zeros((1, 1))]])
        eAt = scipy.linalg.expm(At * Ts)
        Ad = eAt[:A.shape[0], :A.shape[1]]
        Bd = eAt[:A.shape[0], -1:]

        # # Calculate the discrete-time output matrix using the zero-order hold method
        # Cz = np.zeros((C.shape[0], Bd.shape[1]))
        # for i in range(Bd.shape[1]):
        #     Cz[:, i] = C @ np.linalg.matrix_power(Ad, i) @ Bd
        #
        # # Calculate the discrete-time feedforward matrix using the zero-order hold method
        # Dz = np.zeros_like(Cz)
        # for i in range(Bd.shape[1]):
        #     Dz[:, i] = D + C @ np.linalg.matrix_power(Ad, i) @ D @ np.ones((1, Bd.shape[1])) @ np.heaviside(
        #         Ts * np.arange(Bd.shape[1]) - i, 1)

        Ad_, Bd_, Cd_, Dd_, dt = scipy.signal.cont2discrete((A, B, C, D),Ts)

        if domain.lower() == 'z':
            C = Cz
            D = Dz

        return Ad, Bd, C, D

    def euler_step(self, x, u, f, t, dt:float):
        """
        Compute the next state of a dynamical system using the Euler method.

        Parameters:
        x (numpy.ndarray): Current state of size (n, 1)
        u (numpy.ndarray): Input of size (m, 1)
        f (function): Function that computes the derivative of x with respect to time
        t (float): time instance for evaluating input functions
        dt (float): Time step size

        Returns:
        x_next (numpy.ndarray): Next state of size (n, 1)
        """

        # Compute the derivative of x with respect to time
        dxdt = f(t, x, u, dt)
        # print(dxdt)

        # Update the state using the Euler method
        x_next = x + dt * dxdt

        # Using euler forward method
        dxdt = f(t, x, u, dt)
        x_next = x + dt * dxdt

        return x_next

    def kalman_filter(self, A, B, C, D, y, u, P0, Q, R, x0=None):
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
        num_states = A.shape[0]
        num_outputs = C.shape[0]
        x_hat = np.zeros((num_measurements, num_states))
        y_hat = np.zeros((num_measurements, num_outputs))

        if x0 is None:
            x0 = np.zeros(num_states)


        # Initialize state estimates and covariance
        x_hat[0] = x0.reshape(-1)
        P = P0

        for k in range(num_measurements - 1):
            # Prediction step
            x_hat_priori = A @ x_hat[k] + B @ u[k]
            P_priori = A @ P @ A.T + Q

            # Update step
            K = P_priori @ C.T @ np.linalg.inv(C @ P_priori @ C.T + R)
            x_hat[k + 1] = x_hat_priori + K @ (y[k] - C @ x_hat_priori - D @ u[k])
            P = (np.eye(num_states) - K @ C) @ P_priori

            # Compute estimated output
            y_hat[k] = C @ x_hat[k + 1] + D @ u[k]

        return x_hat, y_hat, (K, P)

    def maximum_likelihood_normal(self, data):
        # Calculate the sample mean and standard deviation
        mean = np.mean(data)
        std_dev = np.std(data)

        # Define the likelihood function as a normal distribution
        def likelihood(x):
            return norm.pdf(x, loc=mean, scale=std_dev)

        # Maximize the likelihood function using the Nelder-Mead algorithm
        result = minimize(lambda x: -np.sum(np.log(likelihood(x))), x0=[mean, std_dev])

        # Return the maximum likelihood estimates of the mean and standard deviation
        return result.x[0], result.x[1]

    def maximum_a_posteriori_normal(data, prior_mean, prior_std_dev, prior_weight):
        # Calculate the sample mean and standard deviation
        mean = np.mean(data)
        std_dev = np.std(data)

        # Define the likelihood function as a normal distribution
        def likelihood(x):
            return norm.pdf(data, loc=x[0], scale=x[1]).prod()

        # Define the prior distribution as a normal distribution
        def prior(x):
            return prior_weight * norm.pdf(x[0], loc=prior_mean, scale=prior_std_dev) \
                   * norm.pdf(x[1], loc=prior_mean, scale=prior_std_dev)

        # Define the log-posterior function
        def log_posterior(x):
            return np.log(likelihood(x)) + np.log(prior(x))

        # Maximize the log-posterior function using the Nelder-Mead algorithm
        from scipy.optimize import minimize
        result = minimize(lambda x: -log_posterior(x), x0=[mean, std_dev])

        # Return the maximum a posteriori estimates of the mean and standard deviation
        return result.x[0], result.x[1]
