# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:30:29 2023

@author: BENVI
"""

import numpy as np
from scipy.optimize import least_squares
import control as ctrl
import pandas as pd
import sympy as sym
from numpy.polynomial import polynomial as P
import vectfit as vf


#%% Modified Vector Fitting
# https://github.com/PhilReinhold/vectfit_python/blob/master/vectfit.py

test_s = 1j*np.linspace(1, 1e5, 800)

# Poles are produced in complex conjugate pairs
test_poles = [
    -4500,
    -41000,
    -100+5000j, -100-5000j,
    -120+15000j, -120-15000j,
    -3000+35000j, -3000-35000j,
]


# As are the associated resdiues
test_residues = [
    -3000,
    -83000,
    -5+7000j, -5-7000j,
    -20+18000j, -20-18000j,
    6000+45000j, 6000-45000j,
]

# d == offset, h == slope
test_d = .2
test_h = 2e-5
test_f = vf.model(test_s, test_poles, test_residues, test_d, test_h)

# test data
# zeros_real = np.poly1d([1])
# poles_real = np.poly1d([1,6,2,3,1,1,1,1])
# G = ctrl.tf(np.array(zeros_real),np.array(poles_real))
# f = np.logspace(0.1,1e4,num=10000)
# s = np.array([abs(G(s)) for s in freq])

# Run algorithm, results hopefully match the known model parameters
poles, residues, d, h = vf.vectfit_auto(test_f, test_s, n_poles=6)

fig, ax =plt.subplots(1,1,dpi=200)

ax.plot(test)

# convert poles to a transfer function



#%%
test_s = 1j*np.linspace(1, 1e5, 800)
test_poles = [
    -4500,
    -41000,
    -100+5000j, -100-5000j,
    -120+15000j, -120-15000j,
    -3000+35000j, -3000-35000j,
    -200+45000j, -200-45000j,
    -1500+45000j, -1500-45000j,
    -500+70000j, -500-70000j,
    -1000+73000j, -1000-73000j,
    -2000+90000j, -2000-90000j,
]

test_residues = [
    -500000,
    -83000,
    -5+7000j, -5-7000j,
    -20+18000j, -20-18000j,
    6000+45000j, 6000-45000j,
    40+60000j, 40-60000j,
    90+10000j, 90-10000j,
    50000+80000j, 50000-80000j,
    1000+45000j, 1000-45000j,
    -5000+92000j, -5000-92000j
]

test_d = .2
test_h = 2e-5

test_f = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
test_f += test_d + test_h*test_s

vectfit_auto(test_f, test_s)

poles, residues, d, h = vectfit_auto_rescale(test_f, test_s)

fitted = model(test_s, poles, residues, d, h)

figure()
plot(test_s.imag, test_f.real,color='blue')
plot(test_s.imag, test_f.imag,color='green')
plot(test_s.imag, fitted.real,color='lightblue',ls='--',zorder=3)
plot(test_s.imag, fitted.imag,color='lightgreen',ls='--',zorder=3)
show()

#%%

import numpy as np
from scipy.optimize import least_squares

def mvf(freq, data, poles, zeros):
    # Initialize transfer function parameters
    num_poles = len(poles)
    num_zeros = len(zeros)
    num_terms = num_poles + num_zeros
    coef = np.zeros(num_terms, dtype=complex)

    # Construct initial guess for coefficient values
    coef[:num_poles] = -poles
    coef[num_poles:] = zeros
    resid = np.zeros(data.shape[0], dtype=complex)

    # Define residual function for least-squares optimization
    def residual(coef):
        num_coef = coef[:num_poles]
        den_coef = coef[num_poles:]
        num = np.polyval(num_coef[::-1], freq)
        den = np.polyval(den_coef[::-1], freq)
        tf = num / den
        resid = np.sum(np.abs(tf - data))
        return resid

    # Use least-squares optimization to fit transfer function parameters
    sol = least_squares(residual, coef)
    num_coef = sol.x[:num_poles]
    den_coef = sol.x[num_poles:]
    num = np.polyval(num_coef[::-1], freq)
    den = np.polyval(den_coef[::-1], freq)
    tf = num / den

    return tf
#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
from control import tf

def vector_to_tf(vector, num_poles, num_zeros):
    poles = vector[:num_poles] + 1j * vector[num_poles:2*num_poles]
    zeros = vector[2*num_poles:2*num_poles+num_zeros] + 1j * vector[2*num_poles+num_zeros:]
    return tf(np.real(np.poly(zeros)), np.real(np.poly(poles)))

def residual_function(freq, vector, num_poles, num_zeros):
    tf_fit = vector_to_tf(vector, num_poles, num_zeros)
    return np.abs([tf_fit(2j*np.pi*f) for f in freq])

def fit_mvf(freq, data, num_poles, num_zeros, initial_guess):
    def residual_function(x):
        tf_fit = vector_to_tf(x, num_poles, num_zeros)
        return np.abs([tf_fit(2j*np.pi*f) for f in freq])
    
    vector = np.concatenate((np.real(initial_guess[:num_poles]), np.imag(initial_guess[:num_poles]), 
                             np.real(initial_guess[num_poles:]), np.imag(initial_guess[num_poles:])))
    
    popt, pcov = curve_fit(residual_function, np.array([]), data, p0=vector, method='lm', maxfev=100000)
    poles = popt[:num_poles] + 1j * popt[num_poles:2*num_poles]
    zeros = popt[2*num_poles:2*num_poles+num_zeros] + 1j * popt[2*num_poles+num_zeros:]
    return np.concatenate((np.real(zeros), np.real(poles)))


def mvf(freq, data, poles, zeros):
    num_poles = len(poles)
    num_zeros = len(zeros)
    initial_guess = np.concatenate((np.real(poles), np.real(zeros), np.imag(poles), np.imag(zeros)))
    vector = fit_mvf(freq, data, num_poles, num_zeros, initial_guess)
    return vector_to_tf(vector, num_poles, num_zeros)


zeros_real = np.poly1d([1])
poles_real = np.poly1d([1,1,1,-31,-3100])

G = ctrl.tf(np.array(zeros_real),np.array(poles_real))

freq = np.linspace(0.0,10,num=100)
data = np.array([abs(G(s)) for s in freq])

plt.plot(freq,data)

tf = mvf(freq,data,poles_real.roots,zeros_real.roots)


#%%
f = data
test_s = freq
poles = [1,1,1]

# residues, d, h = calculate_residues(f, s, poles, rcond=rcond)
residues, d, h = calculate_residues(f, test_s, poles)

test_f = vf.model(test_s, poles, residues, d, h)

vectfit_auto(test_f, test_s)
figure()
plot(test_s.imag, test_f.real,color='blue')
plot(test_s.imag, test_f.imag,color='green')
# plot(test_s.imag, fitted.real,color='lightblue',ls='--',zorder=3)
# plot(test_s.imag, fitted.imag,color='lightgreen',ls='--',zorder=3)
show()

#%%


def mvf_to_tf(freq, data, poles, zeros):

    # Compute transfer function using MVF
    tf_coeffs = mvf(freq, data, poles, zeros)

    # Extract numerator and denominator coefficients
    num_coeffs = tf_coeffs[0]
    den_coeffs = np.concatenate(([1], tf_coeffs[1:]))
    
    # Create transfer function object
    tf = control.tf(num_coeffs, den_coeffs)

    return tf

def mvf_to_tf(freq, data, poles, zeros):
    # Compute transfer function using MVF
    tf_coeffs = mvf(freq, data, poles, zeros)

    # Extract numerator and denominator coefficients
    num_coeffs = tf_coeffs[0]
    den_coeffs = np.concatenate(([1], tf_coeffs[1:]))

    # Reorder coefficients to match initial pole and zero guesses
    num_coeffs_out = np.zeros_like(poles)
    den_coeffs_out = np.zeros_like(zeros)

    for i, pole in enumerate(poles):
        if pole in tf_coeffs:
            num_coeffs_out[i] = num_coeffs[tf_coeffs.index(pole)]
        else:
            num_coeffs_out[i] = 0.0

    for i, zero in enumerate(zeros):
        if zero in tf_coeffs:
            den_coeffs_out[i] = den_coeffs[tf_coeffs.index(zero)]
        else:
            den_coeffs_out[i] = 0.0

    # Create transfer function object
    tf = control.tf(num_coeffs_out, den_coeffs_out)

    return tf

def mvf(freq, data, poles, zeros):
    # Initialize transfer function parameters
    num_poles = len(poles)
    num_zeros = len(zeros)
    num_terms = num_poles + num_zeros
    coef = np.zeros(num_terms, dtype=complex) #dtype=complex

    # Construct initial guess for coefficient values
    coef[:num_poles] = -poles
    coef[num_poles:] = zeros
    resid = np.zeros(data.shape[0], dtype=complex) #dtype=complex

    # Define residual function for least-squares optimization
    def residual(coef):
        num_coef = coef[:num_poles]
        den_coef = coef[num_poles:]
        num = np.polyval(num_coef[::-1], freq)
        den = np.polyval(den_coef[::-1], freq)
        tf = num / den
        # print(num, den, tf)
        resid = np.sum(np.abs(tf - data))
        return resid

    # Use least-squares optimization to fit transfer function parameters
    sol = least_squares(residual, abs(coef))
    num_coef = sol.x[:num_poles]
    den_coef = sol.x[num_poles:]
    num = np.polyval(num_coef[::-1], num_zeros)
    den = np.polyval(den_coef[::-1], num_poles)
    tf = num / den

    return tf


def vectfit_step(f, s, poles):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = zeros((Ns, 2*N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    x, residuals, rnk, s = lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = diag(poles)
    b = ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = real(p), imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - outer(b, c)
    H = real(H)
    new_poles = sort(eigvals(H))
    unstable = real(new_poles) > 0
    new_poles[unstable] -= 2*real(new_poles)[unstable]
    return new_poles

def mvf(freq, data, poles):
    s = 1j*freq
    f = data
    N = len(poles)
    Ns = len(s)
    # Initialize transfer function parameters
    # K = len(freq)
    # N_z = len(zeros)
    # N_p = len(poles)
    cindex = np.zeros(N)
    
    # print(K,N_z,N_p)
    

    # --------- Stage 1: Pole identification ---------
    # a_num = np.ones(N_p + 1)

    # Construct: Ax=b for poles 
    # ! [1] "In the fitting process we use only positive frequencies. In order to
    # preserve the conjugacy property we have to formulate (A.7) in 
    # terms of real quantities", Appendix A
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = np.zeros((Ns, 2*N+2), dtype=complex)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    

    print(poles)

    # A = np.ones(shape=(K,N),dtype=complex)
    # for i, p in enumerate(poles):
    #     if np.imag(p) == 0:
    #         A[:,i] = np.array([1/(freq[j]-p) for j in range(len(freq))])
    #         A[:,N + 2 + i] = np.array([-data[j]/(freq[j]-p) for j in range(len(freq))])
            
    #     elif np.imag(p) > 0:
    #         A[:,i] = np.array([1/(freq[j]-p)+1/(freq[j]-np.conjugate(p)) for j in range(len(freq))])
    #         A[:,N + 2 + i] = np.array([-data[j]/(freq[j]-p)+(-data[j])/(freq[j]-np.conjugate(p)) for j in range(len(freq))])

    #         A[:,i+1] = np.array([1j/(freq[j]-p)-1j/(freq[j]-np.conjugate(p)) for j in range(len(freq))])
    #         A[:,N + 2 + i + 1] = np.array([-data[j]*1j/(freq[j]-p)-(-data[j]*1j)/(freq[j]-np.conjugate(p)) for j in range(len(freq))])
            
    #     elif np.imag(p) < 0:
    #         continue        
    
    # A[:,N + 1] = np.array(freq)
    # A = A_p

    b = np.array(data)
        
    x = np.linalg.inv(A.T @ A) @ A.T @ b    
    
    # extracting features from the solution vector, x
    c = x[:N]
    d = x[N]
    h = x[N+1]
    c_ = x[N+1:]
    
    
    residues = x[:N]
    d = x[N]
    h = x[N+1]
    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = diag(poles)
    b = ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = real(p), imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)
    
    # --------- Stage 1.b: Zero calculation => new poles ---------
    # The zeros are calculated as the eigenvalues of the matrix
    H = np.diag(poles) - np.ones(N) @ c.T
    
    # print('H:\n',H)
    
    zeros_ = np.linalg.eig(H)[0]

    new_poles = np.sort(np.linalg.eigvals(H))
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2*real(new_poles)[unstable]
    
    print('Zeros (estimated):\n',new_poles)
    # --------- Stage 2: Residue identification ---------
    

    # --------- Create polynomial coefficients ---------


    # s = sym.Symbol('s')
    
    # eq (4): Finding roots for the approximated rational function
    # zrs = 0
    # pls = 0
    # for i, p in enumerate(poles):
    #     # fs +=(c[i]/(s-p)+d+s*h)-(c_[i]/(s-p)+1)  # eq (4, 8)
    #     pls += (c[i]/(s-p)+d+s*h)
    #     zrs += (c_[i]/(s-p)+1)

    # fs =zrs/pls

    # sol = sym.solve(fs,s)
    # print('f(s):\n',sol)

    # pls = sym.solve(pls,s)
    # zrs = sym.solve(zrs,s)

    # # Initialize 
    # den = [1] + [0] * len(zrs)
    # for root in zrs:
    #     den = P.polymul(den, [1, -root])

    # num = [1] + [0] * len(pls)
    # for root in pls:
    #     num = P.polymul(num, [1, -root])

    # print('Roots:\n',sol)
    # print('Zeros:\n',zrs)
    # print('Poles:\n',pls)

    return new_poles


zeros_real = np.poly1d([1])
poles_real = np.poly1d([1,1,1])
poles_init = np.poly1d([1,1,1])

G = ctrl.tf(np.array(zeros_real),np.array(poles_real))

freq = np.linspace(0,1e3,num=100)
data = np.array([abs(G(s)) for s in freq])

f = data
s = 1j*freq


# residues,d,h,poles_new = mvf(f,s,poles_init.roots)

poles_new = mvf(f,s,poles_init.roots)
print('')
print('real')
print(poles_real.roots)
print('init')
print(poles_init.roots)
print('calc')
print(poles_new)

#%%


from numpy.linalg import eig, inv

# Transfer function
A = np.eye(3)*(-1)
B = np.array([1,1,1])
C = np.diag([0,1,1])
D = 0

# H = ctrl.ss2tf(sys)

zeros_real = np.poly1d([1])
poles_real = np.poly1d([1,1,1,1,1])

G = ctrl.tf(np.array(zeros_real),np.array(poles_real))

freq = np.logspace(0.1,1e1,num=10000)
data = np.array([abs(G(s)) for s in freq])

plt.plot(freq,data)

# A, B, C, D = ctrl.tf2ss(G)

zeros_init = np.poly1d([0.5])
poles_init = np.poly1d([1.1,0.9,1,0.9,1])
# zeros = np.array([0.5])
# poles = np.array([1.1,0.9,1])


# MVF

zeros_calc,poles_calc = mvf(freq, data, poles_init.roots, zeros_init.roots)
# zeros_calc,poles_calc = mvf(freq, data, poles_init, zeros_init)


print('Poles (init, real, calc):',poles_init.roots,poles_real.roots,poles_calc,sep='\n')
print('Zeros (init, real, calc):',zeros_init.roots,zeros_real.roots,zeros_calc,sep='\n')

#%%

def tfe(f,data,tol = 1e-3):
    """
        Based on the 
    """
    # 1 - Read frequency response data and the tolerance level for the residual


    # 2 - Partition the fresuency scale by comideriag the form of the observation
    # data (If the partitioning is not possible (x necessary, take the whole range
    # as a single section; this is aspecial case where r=1)


    # 3 - Estimate the order of the partial terms

    
    # 4 - Form the set of equations as described by expression (6)
    
    
    # 5 - Perform column scaling, check the condition number

    
    # 6 - Identify the parameten of each partial term by the Gauss-Seidel iterations
    # (In the special case of a single section, a single step will give the solution
    # directly)


    # 7 - If the residual is below the tolerance level and all the poles are in the left 
    # half-plane, stop the process

    
    # 8 - Perform iterative improvement until the residual becomes smaller than the
    # given tolerance.

#%%

def mvf(freq, data, poles, zeros):
    # Initialize transfer function parameters
    num_poles = len(poles)
    num_zeros = len(zeros)
    num_terms = num_poles + num_zeros
    coef = np.zeros(num_terms, dtype=complex) #dtype=complex

    # Construct initial guess for coefficient values
    coef[:num_poles] = -poles
    coef[num_poles:] = zeros
    resid = np.zeros(data.shape[0], dtype=complex) #dtype=complex

    # Define residual function for least-squares optimization
    def residual(coef):
        num_coef = coef[:num_poles]
        den_coef = coef[num_poles:]
        num = np.polyval(num_coef[::-1], freq)
        den = np.polyval(den_coef[::-1], freq)
        tf = num / den
        # print(num, den, tf)
        resid = np.sum(np.abs(tf - data))
        return resid

    # Use least-squares optimization to fit transfer function parameters
    sol = least_squares(residual, abs(coef))
    num_coef = sol.x[:num_poles]
    den_coef = sol.x[num_poles:]
    num = np.polyval(num_coef[::-1], num_zeros)
    den = np.polyval(den_coef[::-1], num_poles)
    tf = num / den

    return tf

test_s = 1j*np.linspace(1, 1e5, 800)



#%%
df = pd.DataFrame(tf_coeff)

print(df)

# # # Extract numerator and denominator coefficients
# # num_coeffs = tf_coeffs[0]
# # den_coeffs = np.concatenate(([1], tf_coeffs[1:]))

# # Create transfer function object

# # tf = mvf_to_tf(freq, data, poles, zeros)
# print(G)
# print(tf)

# # tf = control.tf(num_coeffs, den_coeffs)

# ctrl.bode(tf)
# ctrl.bode(G)

#%% Bayesian method

import numpy as np
import pymc3 as pm

def mvf_bayesian(freq, data, poles, zeros, prior_poles=None, prior_zeros=None):
    # Define priors for pole and zero locations
    if prior_poles is None:
        prior_poles = pm.Normal('prior_poles', mu=0, sd=10, shape=len(poles))
    if prior_zeros is None:
        prior_zeros = pm.Normal('prior_zeros', mu=0, sd=10, shape=len(zeros))

    # Define model for transfer function parameters
    with pm.Model() as model:
        # Transfer function numerator and denominator coefficients
        num_coef = pm.Deterministic('num_coef', -poles)
        den_coef = pm.Deterministic('den_coef', zeros)

        # Define prior distributions for pole and zero locations
        num_poles = pm.Deterministic('num_poles', -prior_poles)
        num_zeros = pm.Deterministic('num_zeros', prior_zeros)
        den_poles = pm.Deterministic('den_poles', -prior_poles)
        den_zeros = pm.Deterministic('den_zeros', prior_zeros)

        # Compute transfer function using partial fraction expansion
        num = pm.math.polyval(num_poles[::-1], freq)
        den = pm.math.polyval(den_poles[::-1], freq)
        tf = num / den

        # Likelihood function for observed data
        like = pm.Normal('like', mu=tf, sd=1, observed=data)

        # Run MCMC sampler to estimate posterior distribution
        trace = pm.sample(5000, tune=1000, chains=2, target_accept=0.9)

    # Extract posterior distribution for transfer function
    num_coef_samples = trace['num_coef']
    den_coef_samples = trace['den_coef']
    num = np.polyval(np.median(num_coef_samples, axis=0)[::-1], freq)
    den = np.polyval(np.median(den_coef_samples, axis=0)[::-1], freq)
    tf = num / den

    return tf

