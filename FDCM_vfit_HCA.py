# %%
"""
THIS IT

"""
import pandas as pd
import numpy as np
import linecache
import matplotlib.pyplot as plt
from numpy import exp, log, sqrt, diag, real, imag
from numpy.linalg import inv, det, norm, eig, eigvals, lstsq
from scipy.linalg import expm, schur
from scipy import signal
from scipy.signal import tf2ss
from scipy import stats

class HCA:

    def __init__(self, **kwargs):
        self.opts = HCA_Definitions(**kwargs)

        return

    def load_cable_data(self, path, file, n_conductors):
        n = n_conductors
        fpath = f'{path}\\{file}.out'

        conv = {0: lambda x: str(x)}

        # df = pd.read_csv(fpath, skiprows=59,nrows=n,converters=conv)
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                cnt = 0
                if 'SERIES IMPEDANCE MATRIX (Z)' in line:
                    # print(f'line: {i}')
                    zline = i + 1 + 1
                    # z = np.loadtxt(fpath,skiprows=i+1,max_rows=7,converters=conv,delimiter=',')
                elif 'SHUNT ADMITTANCE MATRIX (Y)' in line:
                    yline = i + 1 + 1
                    # y = np.genfromtxt(fpath,skip_header=i+1,max_rows=7,autostrip=True)
        f.close()

        Z = np.zeros((n, n), dtype=np.complex128)
        Y = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            z_i = linecache.getline(fpath, zline).strip().split(' ' * 3)
            y_i = linecache.getline(fpath, yline).strip().split(' ' * 3)

            Z[i, :] = [complex(float(x.split(',')[0]), float(x.split(',')[1])) for x in z_i]
            Y[i, :] = [complex(float(x.split(',')[0]), float(x.split(',')[1])) for x in y_i]

        return Z, Y

    def rect_map(self, mod, arg):

        z = mod * (np.cos(arg * np.pi / 180) + 1j * np.sin(arg * np.pi / 180))

        return z

    def ULM_propagation_matrix(self, path, file):

        files = ['hmp', 'hpp']
        self.hm = hm = np.genfromtxt(f'{path}\\{file}_hmp.out', skip_header=1)
        self.hp = hp = np.genfromtxt(f'{path}\\{file}_hpp.out', skip_header=1)

        self.s = s = hm[:, 1] * 1j

        # Select only calculated values
        self.hm = hm = hm[:, 2:][:, 0::2]
        self.hp = hp = hp[:, 2:][:, 0::2]

        self.h = h = self.rect_map(hm, hp)

        f = h
        mvf = MVF(n_poles=hm.shape[1])

        fit, SER, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s, mode='ULM')

        mvf.plot(f, s, fit)

        return

    def load_propagation_matrix(self, path, file):

        self.hm = hm = np.genfromtxt(f'{path}\\{file}_hmp.out', skip_header=1)
        self.hp = hp = np.genfromtxt(f'{path}\\{file}_hpp.out', skip_header=1)

        self.s = s = hm[:, 1] * 1j

        # Select only calculated values
        self.hm = hm = hm[:, 2:][:, 0::2]
        self.hp = hp = hp[:, 2:][:, 0::2]

        self.h = h = self.rect_map(hm, hp)

        self.f = h

        return

    def load_char_admittance_matrix(self, path, file):

        fpath = f'{path}\\{file}'
        self.ycm = ycm = np.genfromtxt(f'{path}\\{file}_ycmp.out', skip_header=1)
        self.ycp = ycp = np.genfromtxt(f'{path}\\{file}_ycpp.out', skip_header=1)

        self.s = s = ycm[:, 1] * 1j

        # Select only calculated values
        self.ycm = ycm = ycm[:, 2:][:, 0::2]
        self.ycp = ycp = ycp[:, 2:][:, 0::2]

        self.yc = yc = self.rect_map(ycm, ycp)

        return

    def fit_propagation_matrix(self, Z, Y, L):
        # FDCM approach
        if self.opts.mode.upper() == 'FDCM':

            # Step 1) The modal decomposition given by (5) is performed.
            G, H, Hm, T, lambd, D = self.do_modal_decomposition(Z, Y, L)

            # Step 2) Calculate time delays
            self.tau = tau = self.calc_time_delays(lambd)

            # Step 3) Group eigenvalues #TODO: Revise grouping of eigenvalue together with time delay
            self.grouped_eigenvalues = grouped_eigenvalues = self.group_eigenvalues(H, tau)

            # Step 4) The modal decomposition given by (5) is performed.

            # Step 5) Evaluate fitting error
        elif self.opts.mode.upper() == 'ULM':
            pass
        else:
            raise ValueError('Fitting method HCA.opts.mode must be "FDCM" or "ULM".')

        return

    def fit_char_admittance_matrix(self, Z, Y):

        return

    def do_modal_decomposition(self, Z, Y, L):
        """
        Step 1) The modal decomposition given by (5) is performed.
        """
        # Initialization
        s = 1j * np.linspace(self.opts.f_min, self.opts.f_max, self.opts.Ns)
        N = Z.shape[0]

        # Calculate the propagation matrix
        G = sqrt(Y @ Z)  # Gamma
        H = expm(G * L)  # Propagation matrix

        # Get the eigenvalues e^lambda_i
        Hm = diag(eigvals(H))

        # Transformation matrix
        # T = eig(Y@Z)[1]
        # T1 = eig(H)[1]

        # Get the eigenvalues, lambda_i
        lambd = eigvals(-sqrt(Y @ Z) * L)
        lambd, T = eig(-sqrt(Y @ Z) * L)  # TODO: Figure out the correct T

        P = T @ inv(T).T

        plt.imshow(P.real);
        plt.show();
        plt.close()
        plt.imshow(P.imag);
        plt.show();
        plt.close()
        plt.imshow(abs(P));
        plt.show();
        plt.close();

        print(lambd)

        # Calculate the matrix D_i
        # "where D is a matrix obtained by premultiplying the ith column of T by the ith row of T^-1."
        D = np.zeros((N, N, N), dtype=np.complex128)
        for i in range(N):
            D[:, :, i] = T[:, i] @ inv(T).T[i, :] * exp(lambd[i])
            # D[:,:,i] = T[:,i] @ inv(T).T[i,:]
            # D[:,:,i] = inv(T)[:,i] @ T[i,:]
            # D[:,:,i] = H / Hm[i]
            # D[:,:,i] = (T @ inv(T))[i,:] @ (T @ inv(T))[:,i]

        # print(D)
        # Error propagation
        error = norm(H - T @ Hm @ inv(T))
        print('Error: ', error)

        # Create propogation matrix H from D
        H_from_D = np.zeros_like(H)
        for i in range(N):
            H_from_D += D[:, :, i] * exp(lambd[i])
        error = norm(H - H_from_D)
        print('H: ', H)
        print('H_from_D: ', H_from_D)
        print('Error: ', error)

        # Create propogation matrix H from D w.r.t. frequency s
        H_from_D = np.zeros((*H.shape, self.opts.Ns), dtype=np.complex128)
        print(H_from_D.shape)
        for i in range(N):
            for j in range(len(s)):
                H_from_D[:, :, j] += D[:, :, i] * exp(lambd[i]) * exp(-s[j] * lambd.imag[i])

        print()

        error = norm(H - H_from_D[:, :, 50])
        print('Error: ', error)

        return G, H, Hm, T, lambd, D

    def calc_time_delays(self, lambd):
        """
        Step 2) Time delays associated with modal propagation
        functions are initially estimated by applying Bode's
        magnitude-phase relation that holds for minimum-
        phase systems (MPSs) [11].
        """

        tau = abs(lambd.imag)

        print('time delays: ', tau)

        return tau

    def group_eigenvalues(self, H, tau):
        """
        Step 3) Similar eigenvalues of $H$ and their eigenvectors are
        grouped by summing them, and a single time delay
        is assigned to the group. This can be interpreted as
        reducing the rank of H. The new modal contributions
        become smooth functions of frequency. Equation (8)
        becomes ---

        $H=\Sum_{i=1}^{Nr}$\hat{H}_{i}\exp^{-s\tau_{i}}

        where is Nr the number of modal contribution
        groups corresponding to the reduced rank of H.
        --------------------------------------------------------
        Groups similar eigenvalues of matrix H and their eigenvectors by summing them,
        and assigns a single time delay to the group.

        Parameters:
        H (numpy.ndarray): A square matrix.
        epsilon (float): A threshold value for grouping eigenvalues.
        tau (float): A single time delay to assign to the grouped eigenvalues.

        Returns:
        grouped_eigenvalues (list): A list of tuples, each containing the eigenvalue, eigenvector sum, and time delay of a group.
        """
        eigenvalues, eigenvectors = np.linalg.eig(H)
        sorted_indices = np.argsort(eigenvalues)

        # Initialize the first group with the first eigenvalue and eigenvector
        current_eigenvalue = eigenvalues[sorted_indices[0]]
        current_eigenvector_sum = eigenvectors[:, sorted_indices[0]]
        current_tau = tau
        grouped_eigenvalues = []

        # Iterate over the rest of the eigenvalues and group them as necessary
        for i in range(1, len(eigenvalues)):
            if abs(eigenvalues[sorted_indices[i]] - current_eigenvalue) < self.opts.err_tol:
                # Add the current eigenvector to the current group
                current_eigenvector_sum += eigenvectors[:, sorted_indices[i]]
            else:
                # Add the current group to the list of grouped eigenvalues and start a new group
                grouped_eigenvalues.append((current_eigenvalue, current_eigenvector_sum, current_tau))
                current_eigenvalue = eigenvalues[sorted_indices[i]]
                current_eigenvector_sum = eigenvectors[:, sorted_indices[i]]
                current_tau = tau

        # Add the last group to the list of grouped eigenvalues
        grouped_eigenvalues.append((current_eigenvalue, current_eigenvector_sum, current_tau))

        print(grouped_eigenvalues)

        return grouped_eigenvalues

    def evaluate_fitting_error(self):
        """
        Step 5) If the fitting error in Step 4 is over the specified
        limit, the initial delays are slightly adjusted through
        a search process that tunes delays together with the
        poles and residues of the rational approximation.
        This usually allows for further minimizing the fitting
        error.
        """

        return

    def evaluate_fitting_error(self):
        """
        Step 5) If the fitting error in Step 4 is over the specified
        limit, the initial delays are slightly adjusted through
        a search process that tunes delays together with the
        poles and residues of the rational approximation.
        This usually allows for further minimizing the fitting
        error.
        """

        return

    def plot_cable(self):

        fig, ax = plt.subplots(2, 1, dpi=200, figsize=(9, 6), sharex=True)

        s = self.s.imag

        # Magnitude plot
        for f in self.hm.T:
            ax[0].plot(s, f, color='blue', zorder=2)
            ax[0].set(yscale='log', ylabel='Magnitude')

        for f in self.hp.T:
            # Phase plot
            ax[1].plot(s, f, color='green', zorder=2)
            ax[1].set(xlabel='f [Hz]', ylabel='Phase [deg]')

        # Formatting options
        for i in range(2):
            ax[i].grid()
            ax[i].axhline(0, color='k', lw=0.75)
            ax[i].set(xlim=(s.min(), s.max()), xscale='log')

        return

    def load_cable(self, path, file):
        self.load_propagation_matrix(path, file)

        self.load_char_admittance_matrix(path, file)

        return

    def fit_cable(self,
                  path: str,
                  file: str,
                  n_conductors: int,
                  L: float):

        self.load_cable(path, file)

        if self.opts.mode == 'ULM':
            #
            self.ULM_fit_propagation_matrix(path, file)

            #
            self.ULM_fit_char_admittance_matrix(path, file)
        elif self.opts.mode == 'ULM_single':
            self.H = np.empty((self.h.shape), object)
            self.Y = np.empty((self.h.shape), object)

            s = self.s

            cnt = 0
            for i in range(n_conductors):
                for j in range(n_conductors):
                    y = self.yc[:, cnt]
                    h = self.h[:, cnt]

                    # Propagation fitting
                    mvf = MVF(rescale=True, n_iter=12, n_poles=3, asymp=2)
                    fit, SER, _, (diff, rms_error) = mvf.vectfit(h, s)
                    mvf.plot(y, s, fit)
                    self.H[i, j] = SER

                    # Characteristic admittance fitting
                    mvf = MVF(rescale=True, n_iter=12, n_poles=3, asymp=2)
                    fit, SER, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(y, s)
                    mvf.plot(y, s, fit)
                    self.Y[i, j] = SER

                    # Increment
                    cnt += 1

        elif self.opts.mode == 'FDCM':
            # load cable
            Z, Y = self.load_cable_data(path, file, n_conductors)

            # Fit propagation matrix
            self.fit_propagation_matrix(Z, Y, L)

            # Fit admittance matrix
            self.fit_char_admittance_matrix(Z, Y)

        return


# %%

path = r'C:\Users\BENVI\Documents\validation\PSCAD\DTU projects\HCA\cable_test.if15_x86'
file = r'Cable_2'
n_conductors = 7
length = 150e3  # meter

# HCA
# hca = HCA(mode='ULM')
hca = HCA(mode='ULM_single')
# hca.fit_cable(path,file,n_conductors,length)

# hca.load_cable(path,file)
# hca.plot_cable()
hca.fit_cable(path, file, 7, length)


# %%
class FDCM:

    def __init__(self, path, file, N, L, f_range=(0.005, 1e6), f_steps=800):
        """
        Step 0) Load data

        """
        self.path = path
        self.file = file
        self.N = N
        self.L = L
        self.s = np.linspace(*f_range, f_steps) * 2j * np.pi

        return

    def load_cable(self):
        path = self.path
        file = self.file
        N = self.N
        L = self.L

        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                cnt = 0
                if 'SERIES IMPEDANCE MATRIX (Z)' in line:
                    print(f'line: {i}')
                    zline = i + 1 + 1
                    # z = np.loadtxt(fpath,skiprows=i+1,max_rows=7,converters=conv,delimiter=',')
                elif 'SHUNT ADMITTANCE MATRIX (Y)' in line:
                    yline = i + 1 + 1
                    # y = np.genfromtxt(fpath,skip_header=i+1,max_rows=7,autostrip=True)
        f.close()

        Z = np.zeros((N, N), dtype=np.complex128)
        Y = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            z_i = linecache.getline(fpath, zline).strip().split(' ' * 3)
            y_i = linecache.getline(fpath, yline).strip().split(' ' * 3)

            Z[i, :] = [complex(float(x.split(',')[0]), float(x.split(',')[1])) for x in z_i]
            Y[i, :] = [complex(float(x.split(',')[0]), float(x.split(',')[1])) for x in y_i]

        return Z, Y

    def modal_decomposition(self, Z, Y):
        """
        Step 1) The modal decomposition given by (5) is performed.
        """
        # Calculate the propagation matrix
        G = sqrt(Y @ Z)  # Gamma
        H = expm(G * L)  # Propagation matrix

        # Get the eigenvalues e^lambda_i
        Hm = diag(eigvals(H))

        # Transformation matrix
        T = eig(Y @ Z)[1]
        # T = eig(H)[1]

        # Get the eigenvalues, lambda_i
        lambd = eigvals(-sqrt(Y @ Z) * L)

        # Exponential equivalates to the eigenvalues of H
        lambds = exp(lambd)

        print('Eigenvalues:', lambd, lambds, np.diag(Hm), sep='\n')

        # Calculate the matrix D_i TODO: UNSUCCESFUL
        D = np.zeros((N, N, N), dtype=np.complex128)
        for i in range(N):
            D[:, :, i] = T[:, i] @ inv(T)[i,]

        # Storing values
        self.G = G
        self.H = H
        self.T = T
        self.lambd = np.diag(Hm)
        self.D = D

        # calculate the time delays
        tau = lambd.imag

        # Error propagation
        error = norm(H - T @ Hm @ inv(T))
        print('Error: ', error)

        H_ = np.zeros_like(D)
        for i in range(N):
            H_ += D[:, :, i] * lambd[i]

        error = norm(H - H_)
        print('Error: ', error)

        return G, H, T, D, lambd

    def fit(self):

        # Load cable
        z, y = self.load_cable()

        # Modal decomposition
        G, H, T, D, lambd = self.modal_decomposition(z, y)

        # Calculate time delays
        tau = lambd.imag

        # Grouping modal contribution

        return


fdcm = FDCM(path, file, N, L)
fdcm.fit()

# %%
"""
Step 2) Time delays associated with modal propagation
functions are initially estimated by applying Bode's
magnitude-phase relation that holds for minimum-
phase systems (MPSs) [11].
"""

tau = lambd.imag
tau.sort()
print('time delays: ', tau)
# tau = 1/abs(lambd)
# wn = -lambd.real/lambd

# %%
"""
Step 3) Similar eigenvalues of and their eigenvectors are 
grouped by summing them, and a single time delay
is assigned to the group. This can be interpreted as 
reducing the rank of H. The new modal contributions
become smooth functions of frequency. Equation (8) 
becomes --- where is Nr the number of modal contribution
groups corresponding to the reduced rank of H.
"""


def group_time_constants(time_constants, threshold):
    """
    Groups similar time constants into a dictionary with lists of the index of the time constant as the value.

    Parameters:
    - time_constants: a list of time constants to group
    - threshold: a threshold value for similarity between time constants

    Returns:
    - a dictionary where the keys are the rounded time constants and the values are lists of the index of the time constant
    """
    time_constant_dict = {}

    for i, T in enumerate(time_constants):
        rounded_T = round(T, 2)  # round to two decimal places to group similar time constants
        for key in time_constant_dict.keys():
            if abs(key - rounded_T) < threshold:
                # add the index of the time constant to the existing list of indices for the rounded time constant
                time_constant_dict[key].append(i)
                break
        else:
            # no similar time constant was found, create a new entry in the dictionary
            time_constant_dict[rounded_T] = [i]

    return time_constant_dict


grps = group_time_constants(tau, 1e-2)

print(grps)

# %%
"""
Step 4) The fitting is performed on each modal contribution
group to obtain poles and residues of simultaneously.
The exponential time delay factor is removed 
from the modal contributions prior to fitting

It is underlined here that a common set of poles is
used for each modal contribution
"""

# %%
"""
Step 5) If the fitting error in Step 4 is over the specified
limit, the initial delays are slightly adjusted through
a search process that tunes delays together with the
poles and residues of the rational approximation.
This usually allows for further minimizing the fitting
error.
"""
file = r'Cable_2'
fpath = f'{path}\\{file}.out'

# %%
# Participation matrix
plt.imshow(np.where(abs(T @ inv(T).T) > 0.2, abs(T @ inv(T).T), np.nan))



