_# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:39:46 2023

@author: BENVI
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
"""

# REFERENCE: https://github.com/PhilReinhold/vectfit_python

Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)
All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers

 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.

Other remarkable papers used:
    
 [4] 


"""


def cc(z):
    return z.conjugate()

class Definitions:    
    def __init__(self,**kwargs):        
        # Defined options by Gustavsen
        self.relax=1;      # %Use vector fitting with relaxed non-triviality constraint
        self.stable=1;     # %Enforce stable poles
        self.asymp=2;      # %Include only D in fitting (not E), default = 2
        self.cmplx_ss=1;   # %Create complex state space model
        # self.skip_pole=0;  # %Do NOT skip pole identification
        # self.skip_res=0;   # %Do NOT skip identification of residues (C,D,E) 
        # self.spy1=0;       # %No plotting for first stage of vector fitting
        # self.spy2=1;       # %Create magnitude plot for fitting of f(s) 
        # self.logx=1;       # %Use logarithmic abscissa axis
        # self.logy=1;       # %Use logarithmic ordinate axis 
        # self.errplot=1;    # %Include deviation in magnitude plot
        # self.phaseplot=0;  # %exclude plot of phase angle (in addition to magnitiude)
        # self.legend=1;     # %Do include legends in plots   
        
        # Custom options
        self.rcond = -1                         # Least square r condition https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        self.n_iter = 10                        # Number of iterations for optimizing pole position
        self.rescale = True                     # Rescale option
        self.inc_real = False                   # Increment real initial pole
        self.max_poles = 100                    # Max considered poles
        self.max_iters = 100                    # Max considered poles
        self.print_optimization_status = True   # Print statement after optimization
        self.err_tol = 1e-6   # Print statement after optimization
        self.plot_ci = True                     # Plot confidence interval band
        self.plot_err = True                     # Plot confidence interval band
        self.plot_ser = True                     # Plot confidence interval band
        self.alpha = 0.05                       # Alpha for confidence interval band
        for k,v in kwargs.items():
            setattr(self,k,v)

        return

class HCA_Definitions:    
    def __init__(self,**kwargs):        

        self.err_tol = 1e-1             # Print statement after optimization
        self.f_min = 1             # Print statement after optimization
        self.f_max = 1e6
             # Print statement after optimization
        self.Ns = 101               # Print statement after optimization
        self.mode = 'ULM'               # Print statement after optimization

        for k,v in kwargs.items():
            setattr(self,k,v)

        return

class MVF:
    
    def __init__(self,**kwargs):

        self.opts = Definitions(**kwargs)
        self.poles_list = []

        return
    
    def model(self, s, poles, residues, d, h):
        # DeprecationWarning: Calling np.sum(generator) is deprecated, and in the future will give a different result. Use np.sum(np.fromiter(generator)) or the python sum builtin instead.
        # fit = sum(np.fromiter((r/(s-p) for p, r in zip(poles, residues)),dtype=np.complex128)) + d + s*h
       
        fit = np.sum(r/(s-p) for p, r in zip(poles, residues)) + d + s*h
        return fit 

    def test(self):
        test_s = 1j*np.linspace(1, 1e5, 800)
        self.test_poles = test_poles = [
            -500,
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
        
        self.test_residues = test_residues = [
            -5000,
            -83000,
            -5+7000j, -5-7000j,
            -20+18000j, -20-18000j,
            6000+45000j, 6000-45000j,
            # 2000+5000j, 2000-5000j,
            40+60000j, 40-60000j,
            90+10000j, 90-10000j,
            50000+80000j, 50000-80000j,
            1000+45000j, 1000-45000j,
            -5000+92000j, -5000-92000j
        ]
        
        self.test_d = test_d = .2
        self.test_h = test_h = 2e-5
        # test_d = .0
        # test_h = 0
    
        weight = np.array([1/i for i in range(1,len(test_s)+1)])
    
        # Creating frequency response from test data
        test_f = np.sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
        test_f += test_d + test_h*test_s
    
        # Get fit, poles, residues, d, and h scalars
        fit, SER, (poles, residues, d, h), (diff, rms_error) = self.vectfit(test_f, test_s, weight=weight)
        
        # Plot test data
        self.plot(test_f,test_s,fit)
        
        return fit, SER, (poles, residues, d, h), (diff, rms_error)
    
    def get_cindex(self,poles):
        cindex = np.zeros(len(poles))
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
        
        return cindex

    def build_A_matrix(self,f,s,poles,N,Ns,cindex,identification_mode,weight):
        # use the new poles to extract the residues
        # Scaling for last row of LS-problem (pole identification)
        # TODO: Implement scaling and weight option
       
        if identification_mode == 'poles':
            A = np.zeros((Ns, 2*N+2), dtype=np.complex128)
        else:
            A = np.zeros((Ns, N+2), dtype=np.complex128)

        for i, p in enumerate(poles):
            if cindex[i] == 0:
                A[:, i] = 1/(s - p)
            elif cindex[i] == 1:
                A[:, i] = 1/(s - p) + 1/(s - cc(p))
            elif cindex[i] == 2:
                A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

            if identification_mode == 'poles':
                A[:, N+2+i] = -A[:, i] * f
    
        A[:, N] = 1
        A[:, N+1] = s
        
        return A

    def solve_least_squares(self, A, f):
        b = f
        A = np.vstack((real(A), imag(A)))
        b = np.concatenate((real(b), imag(b)))
        cA = np.linalg.cond(A)
        
        if cA > 1e13:
            # print ('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
            # print ('Cond(A)', cA)
            reduce_order = True
        else: 
            reduce_order = False
            
        x, residuals, rnk, s = lstsq(A, b, rcond=self.opts.rcond)  
        
        return x, residuals, rnk, s, reduce_order
    
    def scale_poles(self):
        return
    def scale_residues(self):
        return
    
    def ULM_build_A_matrix(self,f,s,poles,N,Ns,cindex,identification_mode,weight):
        # use the new poles to extract the residues
        print(weight.shape)
        # Scaling for last row of LS-problem (pole identification)
        # TODO: Implement scaling and weight option
        if len(f.shape) == 1:
            Nc = 1
        else:
            Nc = min(f.shape)
        scale = 0
        for m in range(Nc):
            if len(weight.shape) == 1:
                print(weight.shape,f.shape)
                if len(f.shape) == 1:                
                    scale += (np.linalg.norm(weight * f)) ** 2
                else:
                    scale += (np.linalg.norm(weight * f[:, m])) ** 2
                    
            else:
                scale = scale + (np.linalg.norm(weight[:, m] * f[m, :])) ** 2
        scale = np.sqrt(scale) / Ns        
        
        if identification_mode == 'poles':
            A = np.zeros((Ns, 2*N+2), dtype=np.complex128)
        else:
            A = np.zeros((Ns, N+2), dtype=np.complex128)

        for i, p in enumerate(poles):
            if cindex[i] == 0:
                A[:, i] = 1/(s - p)
            elif cindex[i] == 1:
                A[:, i] = 1/(s - p) + 1/(s - cc(p))
            elif cindex[i] == 2:
                A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

            if identification_mode == 'poles':
                A[:, N+2+i] = -A[:, i] * f
    
        A[:, N] = 1
        A[:, N+1] = s


        # ====================== GUSTAVSEN ================
        Dk = A

        if self.opts.asymp==1:
          offs=0 
        elif self.opts.asymp==2:
          offs=1  
        else:
          offs=2
        
        if len(f.shape) == 1:
            Nc = 1
        else:
            Nc = min(f.shape)        

        AA = np.zeros((Nc * (N + 1), N + 1))
        bb = np.zeros((Nc * (N + 1), 1))
        Escale = np.zeros((1, len(AA[0, :])))

        for n in range(1,Nc+1):
            A = np.zeros((Ns, (N + offs) + N + 1))

            if f.shape == weight.shape and len(f.shape) > 1:
                weig = np.array(weight)[:, n - 1]
            else:
                weig = np.array(weight)
            
            # TODO: Confirm range with offset
            for m in range(0, N + offs):    # left block
                A[0:Ns, m] = (weig * Dk[0:Ns, m])

            inda = N + offs
            for m in range(0, N):           # right block
                print(A.shape,Dk.shape,f.shape)
                A[0:Ns, inda + m] = -weig * Dk[0:Ns, m ] * np.matrix(f)[n - 1, 0:Ns].T

            A = np.concatenate((A.real, A.imag), axis=0)

            # Integral criterion for sigma:
            offset = (N + offs)
            if n == Nc:
                for mm in range(0, N):
                    A[2 * Ns - 1, offset + mm ] = scale * np.sum(Dk[:, mm])

            # Compute the qr factorization of a matrix. # https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
            Q, R = np.linalg.qr(A, mode='reduced')
            ind1 = N + offs
            ind2 = N + offs + N
            R22 = R[ind1 - 1:ind2, ind1 - 1:ind2]
            print(R22.shape)
            print((n+1) * (N),n * (N + 1))
            # AA[n * (N + 1):(n+1) * (N), :] = R22
            print(n,N)
            AA[(n-1) * (N + 1):n * (N + 1), :] = R22

            if n == Nc:
                bb[(n - 1) * (N + 1):n * (N + 1), 0] = Q[-1, N + offs:n * (N + offs + N + 1)] * Ns * scale

            for col in range(1, len(AA[0, :]) + 1):
                Escale[0, col - 1] = 1 / np.linalg.norm(AA[:, col - 1])
            AA[:, col - 1] = Escale[0, col - 1] * AA[:, col - 1]

            x = np.linalg.lstsq(AA, bb, rcond=None)[0]
            x = x * Escale.T

        return AA, bb, x
    
    def ULM_calculate_residues(self,f, s, poles, weight):
        Ns = len(s)
        N  = len(poles)
    
        cindex = self.get_cindex(poles)
    
        # use the new poles to extract the residues
        
        A = self.build_A_matrix(f,s,poles,N,Ns,cindex,'residues',weight)
        
        # TODO: implement scale residues
        
        # Solve Ax == b using pseudo-inverse
        x, residuals, rnk, s, reduce_order = self.solve_least_squares(A, f)
    
        # Recover complex values
        x = np.complex64(x)
        for i, ci in enumerate(cindex):
           if ci == 1:
               r1, r2 = x[i:i+2]
               x[i] = r1 - 1j*r2
               x[i+1] = r1 + 1j*r2
    
        # Get elements from solutions
        residues = x[:N]
        d = x[N].real
        h = x[N+1].real
        
        return residues, d, h, reduce_order, x
    
    def ULM_calculate_poles(self,f,s,poles, weight):
        # Loading variables
        Ns = len(s)
        N = len(poles)
        
        cindex = self.get_cindex(poles)

        # First linear equation to solve. See Appendix A
        A = self.build_A_matrix(f,s,poles,N,Ns,cindex,'poles',weight)
            
        # Solve Ax == b using pseudo-inverse
        x, residuals, rnk, s, reduce_order = self.solve_least_squares(A, f)
    
        # Get elements from solutions
        residues = x[:N]
        # d = x[N]
        # h = x[N+1]
    
        # We only want the "tilde" part in (A.4)
        x = x[-N:]
    
        # Calculation of zeros: Appendix B
        A = diag(poles)
        b = np.ones(N)
        c = x
        for i, (ci, p) in enumerate(zip(cindex, poles)):
            if ci == 1:
                x, y = real(p), imag(p)
                A[i, i] = A[i+1, i+1] = x
                A[i, i+1] = -y
                A[i+1, i] = y
                b[i] = 2
                b[i+1] = 0
       
        # Finding H
        H = A - np.outer(b, c)
        H = real(H)
        new_poles = np.sort(eigvals(H))
        unstable = real(new_poles) > 0
        new_poles[unstable] -= 2*real(new_poles)[unstable]
        return new_poles, reduce_order
    
    def calculate_residues(self,f, s, poles, weight):
        Ns = len(s)
        N  = len(poles)
    
        cindex = self.get_cindex(poles)
    
        # use the new poles to extract the residues
        A = self.build_A_matrix(f,s,poles,N,Ns,cindex,'residues',weight)
        
        # TODO: implement scale residues
        
        # Solve Ax == b using pseudo-inverse
        x, residuals, rnk, s, reduce_order = self.solve_least_squares(A, f)
    
        # Recover complex values
        x = np.complex64(x)
        for i, ci in enumerate(cindex):
           if ci == 1:
               r1, r2 = x[i:i+2]
               x[i] = r1 - 1j*r2
               x[i+1] = r1 + 1j*r2
    
        # Get elements from solutions
        residues = x[:N]
        d = x[N].real
        h = x[N+1].real
        
        return residues, d, h, reduce_order, x

    def calculate_poles(self,f,s,poles, weight):
        # Loading variables
        Ns = len(s)
        N = len(poles)
        
        cindex = self.get_cindex(poles)

        # First linear equation to solve. See Appendix A
        A = self.build_A_matrix(f,s,poles,N,Ns,cindex,'poles',weight)
            
        # TODO: implement scale poles

        # Solve Ax == b using pseudo-inverse
        x, residuals, rnk, s, reduce_order = self.solve_least_squares(A, f)
    
        # Get elements from solutions
        residues = x[:N]
        d = x[N]
        h = x[N+1]
    
        # We only want the "tilde" part in (A.4)
        x = x[-N:]
    
        # Calculation of zeros: Appendix B
        A = diag(poles)
        b = np.ones(N)
        c = x
        for i, (ci, p) in enumerate(zip(cindex, poles)):
            if ci == 1:
                x, y = real(p), imag(p)
                A[i, i] = A[i+1, i+1] = x
                A[i, i+1] = -y
                A[i+1, i] = y
                b[i] = 2
                b[i+1] = 0
       
        # Finding H
        H = A - np.outer(b, c)
        H = real(H)
        new_poles = np.sort(eigvals(H))
        unstable = real(new_poles) > 0
        new_poles[unstable] -= 2*real(new_poles)[unstable]
        return new_poles, reduce_order

    def get_init_poles(self,s,n_poles,loss_ratio=1e-2):

        w = imag(s)
        pole_locs = np.linspace(w[0], w[-1], n_poles+2)[1:-1]
        lr = loss_ratio
        init_poles = poles = np.concatenate([[p*(-lr + 1j), p*(-lr - 1j)] for p in pole_locs])
        
        # Increment real?
        if self.opts.inc_real:
            poles = concatenate((poles, [1]))
        
        self.poles_list.append(poles)
        
        return poles

    def calculate_ci(self,f, fit):
        """
        Calculates the confidence interval for a series of data.
    
        Parameters:
        -----------
        model_data: array-like
            The predictor variable values used in the model.
        fitted_data: array-like
            The predicted values generated by the model.
        self.opts.alpha: float, optional
            The level of significance for the confidence interval. Default is 0.05.
    
        Returns:
        --------
        ci: tuple
            The lower and upper bounds of the confidence interval.
        """
        
        # Calculate the prediction error
        prediction_error = np.array(fit) - np.array(f)
    
        # Calculate the standard error of the prediction error
        n = len(f)
        if len(f.shape) == 1:
            Nc = 1
        else:
            Nc = min(f.shape)
        dof = n - Nc  # Degrees of freedom
        mse = np.sum(prediction_error ** 2) / (dof)  # Mean squared error
        se = np.sqrt(mse / n)
    
        # Calculate the critical value from the t-distribution
        cv = stats.t.ppf(1 - self.opts.alpha / 2, dof)
    
        # Calculate the confidence interval
        mean_pe = np.mean(prediction_error)
        ci = (mean_pe - cv * se, mean_pe + cv * se)
    
        return ci


    def evaluate_error(self,f,fit):
        if len(f.shape) == 1:
            Nc = 1
        else:
            Nc = min(f.shape)
        Ns = len(f)
                    
        # Error
        error = fit - f
        
        # Get confidence interval
        ci = self.calculate_ci(f,fit)

        # Calculate the RMS error        
        rms_error = np.sqrt(np.sum(np.abs(error) ** 2)) / np.sqrt(Nc * Ns)
        
        signal_error_ratio = abs(f) / abs(error)
        
        return error, rms_error, ci, signal_error_ratio


    def fit_wrapper(self,f,s,poles,weight):
        for _ in range(self.opts.n_iter):

            poles,reduce_order1 = self.calculate_poles(f, s, poles, weight)

            self.poles_list.append(poles)
    
        residues, d, h, reduce_order2, x_residues = self.calculate_residues(f, s, poles, weight)

        
        return poles, residues, d, h, (reduce_order1 or reduce_order2), x_residues

    def optimize_fit(self,f,s,weight):
        # TODO: Implement optimization of model order
        # initial values
        Ns = len(s)

        # Optimize fitting
        accepted_fit = False
        n_poles = self.opts.n_poles
        cnt = 0
        while not accepted_fit and self.opts.max_iters > cnt and n_poles < self.opts.max_poles:           
            # Initialize poles
            poles = self.get_init_poles(s,n_poles)
            
            # Fit
            poles, residues, d, h, reduce_order, x_residues = self.fit_wrapper(f,s,poles,weight)
        
            # Model the fitted poles and residues to frequency response
            fit = self.model(s, poles, residues, d, h)

            # Evaluate the error
            diff, rms_error, *_ = self.evaluate_error(f,fit)

            # Optimization acceptance criteria
            if rms_error < self.opts.err_tol and not reduce_order:
                accepted_fit = True
            elif reduce_order:
                n_poles -= 1
            else:
                n_poles += 1
                
            cnt += 1
        
        if self.opts.print_optimization_status:
            print('#================================================#','# Fitting summary','#================================================#',sep='\n')
            print(f'Accepted fit:\t\t\t{accepted_fit}',
                  f'Error tol.:\t\t\t\t{self.opts.err_tol}',
                  f'RMS error:\t\t\t\t{rms_error}',
                  f'Starting poles:\t\t\t{self.opts.n_poles}',
                  f'Number of poles:\t\t{n_poles}',
                  f'Iterations:\t\t\t\t{cnt}',
                  sep='\n')
            print('#================================================#',sep='\n')

        return poles, residues, d, h, diff, rms_error, x_residues

    def vectfit(self,f,s,weight=None,mode=None):
        """
        f = complex data to fit
        s = j*frequency

        returns adjusted poles
        """       
        # Input validation
        if not s.imag.any():
            s *= 1j

        # Generate weights if none is provided
        if weight is None:       
            weight = np.array([1 for i in range(1,len(s)+1)])
            # weight = [1/i for i in range(1,len(s)+1)]

        # Optimizing fit
        if mode=='ULM':                        
            self.ULM_calculate_poles(f,s,poles,N,Ns,cindex,'poles',weight)
            
        else:
            poles, residues, d, h, diff, rms_error, x_residues = self.optimize_fit(f, s, weight)
    
            # Get state equation realization
            # SER = self.get_state_equation_realization(f,s,poles,x_residues,weight)
            SER = self.poles_residues_to_state_space(poles,residues)
    
            # Model the frequency response from poles and residues
            fit = self.model(s, poles, residues, d, h)
        
        return fit, SER, (poles, residues, d, h), (diff, rms_error)
    
    def poles_residues_to_state_space(self,poles, residues, k = None):
        if k is None:
            k = 0

        residues = np.array(residues)
        poles = np.array(poles)

        # scipy.signal.residue(b, a, tol=0.001, rtype='avg')
        
        # Digital
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invres.html#scipy.signal.invres
        num, den = signal.invres(residues, poles, 0, tol=0.001, rtype='avg')
        # num, den = signal.invres(residues, poles, k, tol=0.001, rtype='avg')

        # tf = num / den
    
        # Convert the transfer function to state-space realization
        A, B, C, D = tf2ss(num, den)
    
        # Split each complex state variable into two real variables
        A_real = np.hstack((np.real(A), -np.imag(A)))
        A_real = np.vstack((A_real, np.hstack((np.imag(A), np.real(A)))))
        B_real = np.vstack((np.real(B), np.imag(B)))
        C_real = np.hstack((np.real(C), np.imag(C)))
        
        # Compute the complex Schur decomposition of A
        # U, T = schur(A)
        
        # # Split each complex state variable into two real variables
        # U_real = np.hstack((np.real(U), -np.imag(U)))
        # U_real = np.vstack((U_real, np.hstack((np.imag(U), np.real(U)))))
        # T_real = np.vstack((np.real(T), np.imag(T)))
        # T_real = np.vstack((T_real, np.zeros((T_real.shape[0]//2, T_real.shape[1]//2))))
        
        # Construct the real state-space system
        # SER = signal.StateSpace(U_real, B, C @ U_real, D)
        
        # Construct the real state-space system
        # SER = signal.StateSpace(A_real, B_real, C_real, D.real)
        SER = signal.StateSpace(A, B, C, D)
    
        # SER = {'A':A_real,
        #        'B':B_real,
        #        'C':C_real,
        #        'D':D,
        #        }
    
        # Return the state-space matrices
        return SER
    
    def plot(self,f,s,fit,xmax=None,plot_err:bool=True,plot_ser:bool=True):
        diff, rms_error, ci, signal_error_ratio = self.evaluate_error(f,fit)
        
        fig, ax = plt.subplots(2,1,dpi=200,figsize=(9,6),sharex=True)
        
        s_f = s.imag/(2*np.pi)
        
        # Magnitude plot
        ax[0].plot(s_f, abs(f),color='blue',zorder=2)
        ax[0].plot(s_f, abs(fit),color='lightblue',ls='--',zorder=3)
        ax[0].set(yscale='log',ylabel='Magnitude')
        # ax[0].set(ylabel='Magnitude')
        if self.opts.plot_ci:
            ax[0].fill_between(s_f, abs(f)+(ci[0]),abs(f)-(ci[1]),alpha=0.25,color='red',zorder=5)
            # ax[0].plot(s_f, abs(f+ci[0]),color='lightblue',ls='--',zorder=3)            

        if plot_err:
            ax0 = ax[0].twinx()
            ax0.set(yscale='log')
            ax0.set_ylabel(f'Error: {rms_error}', color='tab:blue')
            ax0.plot(s_f,abs(diff), color='tab:blue',alpha=0.25)
            ax0.tick_params(axis='y', labelcolor='tab:blue')

        if plot_ser:
            ax1 = ax[0].twinx()
            # ax1.set(yscale='log')
            ax1.set_ylabel('$SER^{-1}$', color='tab:red')
            ax1.plot(s_f,1/signal_error_ratio, color='tab:red',alpha=0.25)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.spines["right"].set_position(("axes", 1.1))

        # Phase plot
        ax[1].plot(s_f, np.angle(f,deg=True),color='green',zorder=2)        
        ax[1].plot(s_f, np.angle(fit,deg=True),color='lightgreen',ls='--',zorder=3)
        ax[1].set(xlabel='f [Hz]',ylabel='Phase [deg]')

        # Formatting options
        for i in range(2):
            ax[i].grid()
            ax[i].axhline(0,color='k',lw=0.75)
            ax[i].set(xlim=(s_f.min(),(xmax,s_f.max())[xmax is None]))
        
        plt.show()
        plt.close()
        
        return

mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2)

# fit, SER, (poles, residues, d, h), (diff, rms_error) = mvf.test()

#%%
def rect_form(m,p,deg=True):
    if deg: s = np.pi/180
    
    z = m*(np.cos(p*s)+1j*np.sin(p*s))
    
    return z

def calculate_foster_network(p,r,d,h,verbose=False,plot=True,mode:bool = 'real'):
    M = len(p)
    Z_foster = {}    
    for i in range(M+1):
        if i == 0:
            Z_foster[f'R{i}'] = (d - sum([rm/pm for rm, pm in zip(r,p)])).real
            Z_foster[f'L{i}'] = h

        else:
            # If real pole and real residue            
            if r[i-1].imag == 0 and p[i-1].imag == 0:
                Z_foster[f'R{i}'] = (r[i-1]/p[i-1]).real
                Z_foster[f'L{i}'] = - (r[i-1]/p[i-1] / p[i-1]).real      
                
            else: # If complex pole or residue                
                # Simplify names
                s = r[i-1] # 
                g = p[i-1]

                # Assign values
                Z_foster[f'R{i}'] = ((-2*g.real*s.real**3 - 2*s.real**2 * s.imag * g.imag)/(g.imag**2 * (s.real**2 + s.imag**2))).real
                Z_foster[f'L{i}'] = ((2*s.real**3)/(g.imag**2 * (s.real**2+ s.imag**2))).real
                Z_foster[f'C{i}'] = (1/(2*s.real)).real
                Z_foster[f'r{i}'] = ((2*s.real**2)/(g.imag * s.imag - g.real * s.real)).real      
    
    if dataframe:
        idxs = np.arange(0,M+1)
        for abc in ['R','L','C','r']:
            keys = Z_foster
        df = pd.DataFrame({},index=idxs)            
    
    if verbose:
        for k,v in Z_foster.items():
            print(f'{k}={v}')        
    
    if plot:
        fig, ax = plt.subplots(2,1,dpi=200,figsize=(8,6))
        Rs = {k:v for k,v in Z_foster.items() if k[0] == 'R'}
        Ls = {k:v for k,v in Z_foster.items() if k[0] == 'L'}
        ax[0].bar(list(Rs.keys()),list(Rs.values()))
        ax[1].bar(list(Ls.keys()),list(Ls.values()))        

        ax[0].set_xticklabels(list(Rs.keys()), rotation = 90, ha="center")
        ax[1].set_xticklabels(list(Ls.keys()), rotation = 90, ha="center")

        # for i in range(2):
        #     ax[i].set(yscale='log')
    
    return Z_foster

# calculate_foster_network(poles,residues,d,h,verbose=True)
# =======

#%%
def fourier_method(p,r,t1,t2,dt=1e-6):
    # inv_fourier_transform = lambda p,r,t: 
    
    t = np.arange(t1,t2+dt,dt)
    y = np.zeros((len(t)))

    fig, ax = plt.subplots(1,1,dpi=200)
    for i, t_ in enumerate(t):
        y[i] = np.array([r_*np.exp(p_*t_) for r_, p_ in zip(r,p)]).sum() 

    ax.plot(t,y)   
    
    return

# fourier_method(poles,residues,0,0.2)

#%% CALCULATED FREQUENCY RESPONSE 
f_calc = pd.read_csv(f'data\\freq\\cable_1C_freq_calc.txt',header=0,index_col=0)
f_calc = f_calc[f_calc.index <= 2e3]

f = f_calc['real'] + f_calc['imag']*1j
s = f_calc.index.values*1j*2*np.pi

# Get fit, poles, residues, d, and h scalars
mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2,plot_ser=False,plot_err=False)
fit, SER, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s)
mvf.plot(f,s,fit,xmax=2000,plot_ser=False,plot_err=True)

#%% CALCULATED FREQUENCY RESPONSE w ARTIFICIAL ATTENUATION
f_calc = pd.read_csv(f'data\\freq\\cable_1C_freq_calc.txt',header=0,index_col=0)
# f_calc = f_calc[f_calc.index <= 2e3]

f = f_calc['real'] + f_calc['imag']*1j
f = f*np.linspace(1,0.0,len(f))
s = f_calc.index.values*1j*2*np.pi

# Get fit, poles, residues, d, and h scalars
mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2,plot_ser=False,plot_err=False)
fit, SER2, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s)
mvf.plot(f,s,fit,xmax=2000,plot_ser=False,plot_err=True)

#%%
f_fdcm = np.genfromtxt(r'data\freq\Harm_1c_fdcm.out',skip_header=1)
f = rect_form(f_fdcm[:,1],f_fdcm[:,2])
s = f_fdcm[:,0]*1j*2*np.pi

# Get fit, poles, residues, d, and h scalars
mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2,plot_ser=False,plot_err=False)
fit, SER3, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s)
mvf.plot(f,s,fit,xmax=2000,plot_ser=False,plot_err=True)

#%%
def print_pscad_branches_from_foster_network(Z_fn):
    return


Z_fn = calculate_foster_network(poles,residues,d,h,verbose=True)

#%%
f_fdcm = np.genfromtxt(r'data\freq\Harm_1c_pi.out',skip_header=1)
f = rect_form(f_fdcm[:,1],f_fdcm[:,2])
s = f_fdcm[:,0]*1j*2*np.pi

# Get fit, poles, residues, d, and h scalars
mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2,plot_ser=False,plot_err=False)
fit, SER4, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s)
mvf.plot(f,s,fit,xmax=2000,plot_ser=False,plot_err=True)

Z_f = calculate_foster_network(poles,residues,d,h,verbose=True)

#%% 3 CONDUCTORS
f_fdcm = np.genfromtxt(r'data\cable_3c\Harm_3c.out',skip_header=1)

#%%
import re

df = pd.read_csv(r'data\cable_3c\Harm_3c.out',delimiter='\t')
n_phases = 3
n_freqs = int(len(df)/n_phases)

print(df)

Z = np.zeros((n_phases,n_phases,n_freqs),dtype=np.complex128)
f = np.zeros((n_freqs))
for i in range(0,n_phases*n_freqs-n_phases+1,3):
    k = int(i/3)
    for j in range(n_phases):
        line = df.loc[i + j].values[0].strip()
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?', line)
        numbers = [float(num) for num in numbers]
        # print(numbers)
        print(i,j,k,len(numbers),numbers)

        if j == 0:
            f[k] = numbers[0]
            Z[j,j,k] = complex(numbers[1],numbers[2])
        else:
            # Mutual impedance
            for j_ in range(j):
                Z[j,j_,k] = Z[j_,j,k] = complex(numbers[j_],numbers[j_+1])

            # Mutual impedance
            Z[j,j,k] = complex(numbers[-2],numbers[-1])

    print(Z[:,:,k])

# Put inside data frame
data = {}
for i in range(n_phases):
    for j in range(n_phases):
        data[f'Z{i+1}{j+1}'] = Z[i,j,:]
    
df = pd.DataFrame(data,index=f)

#%%
f = rect_form(f_fdcm[:,1],f_fdcm[:,2])
s = f_fdcm[:,0]*1j*2*np.pi

# Get fit, poles, residues, d, and h scalars
mvf = MVF(rescale=True,n_iter=12,n_poles=3,asymp=2,plot_ser=False,plot_err=False)
fit, SER4, (poles, residues, d, h), (diff, rms_error) = mvf.vectfit(f, s)
mvf.plot(f,s,fit,xmax=2000,plot_ser=False,plot_err=True)

#%%
from sympy import Matrix

m = Matrix(SER.A)

#%%
P, J = m.jordan_form()


#%%
import pickle # https://www.digitalocean.com/community/tutorials/python-pickle-example
from sympy import Matrix
import copy

class HCA:
    
    def __init__(self):
        self.data = []
        
        return

    def css2rss(self,SER):
        SER_real = copy.deepcopy(SER)
        
        for elm in ['A','B','C','D']:
            real_matrix = np.vstack((np.hstack((getattr(SER,elm).real, -getattr(SER,elm).imag)),
                           np.hstack((getattr(SER,elm).imag, getattr(SER,elm).real))))
            setattr(SER_real,elm, real_matrix)

        
        return SER_real
    
    def pickle(self,SER,
              cable:str=None,
              idx:tuple=None):
        
        
        
        SER_real = self.css2rss(SER)
        
        
        pickle_data = {'SER':SER,
                       'A':SER_real.A,
                       'B':SER_real.B,
                       'C':SER_real.C,
                       'D':SER_real.D,
                       'cable':cable,
                       'idx':idx
                       }
        
        self.data.append(pickle_data)        
        
        return
    
    def store_pickles(self,path,file):
        filepath = f'{path}\\{file}.dat'
        # open a file, where you ant to store the data
        file = open(filepath, 'wb')
        
        # dump information to that file
        pickle.dump(self.data, file)
        
        # close the file
        file.close()        
    
        return
        
    def load_pickles(self,path,file):
        filepath = f'{path}\\{file}.dat'

        # open a file, where you ant to store the data
        file = open(filepath, 'rb')
        
        # dump information to that file
        data = pickle.load(file)
        
        # close the file
        file.close()        
        
        return
        

        

























