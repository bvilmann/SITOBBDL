import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.signal import cont2discrete
import scipy

class NumericalMethods:

    def __init__(self):

        return

    def c2d(self,A, B, C, D, Ts):

        # Calculate the continuous-time state transition matrix using the matrix exponential
        At = np.block([[A, B], [np.zeros((1, A.shape[1])), np.zeros((1, 1))]])
        eAt = scipy.linalg.expm(At * Ts)
        Ad = eAt[:A.shape[0], :A.shape[1]]
        Bd = eAt[:A.shape[0], -1:]

        # Calculate the discrete-time output matrix using the zero-order hold method
        Cz = np.zeros((C.shape[0], Bd.shape[1]))
        for i in range(Bd.shape[1]):
            Cz[:, i] = C @ np.linalg.matrix_power(Ad, i) @ Bd.reshape(Bd.shape[0])

        # Calculate the discrete-time feedforward matrix using the zero-order hold method
        # Dz = np.zeros_like(Cz)
        # for i in range(Bd.shape[1]):
        #     Dz[:, i] = D + C @ np.linalg.matrix_power(Ad, i) @ D @ np.ones((1, Bd.shape[1])) @ np.heaviside(Ts * np.arange(Bd.shape[1]) - i, 1)

        Ad_, Bd_, Cd_, Dd_, dt = scipy.signal.cont2discrete((A, B, C, D),Ts)

        if domain.lower() == 'z':
            C = np.diag(Cz)
            # D = Dz
            D = 0

        print('Delta Ad:',Ad_-Ad,sep='\n')
        print('Delta Bd:',Bd_-Bd,sep='\n')

        return Ad, Bd, C, D

    def init_kalman_filter(self, A, B, C, D, y, u, R0, R1, R2, x0=None):
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
            P0 = R0 (ndarray): Initial state covariance matrix, with shape (num_states, num_states)
            Q =  R1 (ndarray): Process noise covariance matrix, with shape (num_states, num_states)
            R =  R2 (ndarray): Measurement noise covariance matrix, with shape (num_outputs, num_outputs)

        Returns:
            x_hat (ndarray): Array of estimated states, with shape (num_measurements, num_states)
            y_hat (ndarray): Array of estimated outputs, with shape (num_measurements, num_outputs)
        """
        # Initialize arrays
        n_y = y.shape[0]
        n = A.shape[0]
        m = C.shape[0]
        x_hat = np.zeros((n_y, n))
        y_hat = np.zeros((n_y, m))

        # Initialize state estimates and covariance
        x_hat[0] = x0.reshape(-1)
        P = R0
        Q = R1
        R = R2

        for k in range(n_y - 1):
            # Prediction step
            x_hat_priori = A @ x_hat[k] + B @ u[k]
            P_priori = A @ P @ A.T + Q

            # Update step
            K = P_priori @ C.T @ np.linalg.inv(C @ P_priori @ C.T + R)
            x_hat[k + 1] = x_hat_priori + K @ (y[k] - C @ x_hat_priori - D @ u[k])
            P = (np.eye(n) - K @ C) @ P_priori

            # Compute estimated output
            y_hat[k] = C @ x_hat[k + 1] + D @ u[k]

            # Calculate maximum likelihood


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
        return result

    import numpy as np
    from scipy.optimize import minimize

    def kalman_filter_mle(self,Y, model):
        """
        Calculate the maximum likelihood estimate (MLE) of the Kalman filter parameters.

        Parameters:
            Y (ndarray): A 2D array of shape (num_timesteps, num_observed_variables) containing the observed measurements.
            model (KalmanFilterModel): An object representing the state-space model for the Kalman filter.

        Returns:
            ndarray: An array of estimated parameter values, ordered as [Q, R, x0, P0], where Q and R are the process and measurement noise covariance matrices, x0 is the initial state estimate, and P0 is the initial error covariance matrix.
        """

        # Define the likelihood function as a function of the parameters
        def likelihood(params):
            Q, R, x0, P0 = params
            num_timesteps, num_observed_variables = Y.shape

            # Initialize the Kalman filter with the estimated parameters
            model.initialize(Q, R, x0, P0)

            # Calculate the log-likelihood of the observations given the model parameters
            log_likelihood = 0
            for i in range(num_timesteps):
                observation = Y[i]
                predicted_observation = model.observe()
                log_likelihood += np.log(model.pdf(observation, predicted_observation))
                model.update(observation)

            # Return the negative of the log-likelihood, since we want to maximize it
            return -log_likelihood

        # Initialize the parameter estimates to the model defaults
        initial_params = [model.Q, model.R, model.x, model.P]

        # Use the Nelder-Mead optimization algorithm to find the parameter values that maximize the likelihood function
        result = minimize(likelihood, initial_params, method='Nelder-Mead')

        # Return the estimated parameter values
        return result.x

    def maximum_a_posteriori_normal(self, data, prior_mean, prior_std_dev, prior_weight):
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
