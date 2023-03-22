# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:19:53 2023

@author: BENVI
"""



class Definitions:    
    def __init__(self,**kwargs):        
        self.relax=1;      # %Use vector fitting with relaxed non-triviality constraint
        self.stable=1;     # %Enforce stable poles
        self.asymp=2;      # %Include only D in fitting (not E), default = 2
        self.skip_pole=0;  # %Do NOT skip pole identification
        self.skip_res=0;   # %Do NOT skip identification of residues (C,D,E) 
        self.cmplx_ss=1;   # %Create complex state space model
        self.spy1=0;       # %No plotting for first stage of vector fitting
        self.spy2=1;       # %Create magnitude plot for fitting of f(s) 
        self.logx=1;       # %Use logarithmic abscissa axis
        self.logy=1;       # %Use logarithmic ordinate axis 
        self.errplot=1;    # %Include deviation in magnitude plot
        self.phaseplot=0;  # %exclude plot of phase angle (in addition to magnitiude)
        self.legend=1;     # %Do include legends in plots   
        
        for k,v in kwargs.items():
            setattr(self,k,v)

        return

# %function [SER,poles,rmserr,fit,opts]=vectfit3(f,s,poles,weight,opts)
# %function [SER,poles,rmserr,fit]=vectfit3(f,s,poles,weight,opts)
# %function [SER,poles,rmserr,fit]=vectfit3(f,s,poles,weight)
# % 
# %     ===========================================================
# %     =   Fast Relaxed Vector Fitting                           =
# %     =   Version 1.0                                           =
# %     =   Last revised: 08.08.2008                              = 
# %     =   Written by: Bjorn Gustavsen                           =
# %     =   SINTEF Energy Research, N-7465 Trondheim, NORWAY      =
# %     =   bjorn.gustavsen@sintef.no                             =
# %     =   http://www.energy.sintef.no/Produkt/VECTFIT/index.asp =
# %     =   Note: RESTRICTED to NON-COMMERCIAL use                =
# %     ===========================================================
# %
# % PURPOSE : Approximate f(s) with a state-space model 
# %
# %         f(s)=C*(s*I-A)^(-1)*B +D +s*E
# %                       
# %           where f(s) is a singe element or a vector of elements.  
# %           When f(s) is a vector, all elements become fitted with a common
# %           pole set.
# %
# % INPUT :
# %
# % f(s) : function (vector) to be fitted. 
# %        dimension : (Nc,Ns)  
# %                     Nc : number of elements in vector
# %                     Ns : number of frequency samples 
# % 
# % s : vector of frequency points [rad/sec] 
# %        dimension : (1,Ns)  
# %
# % poles : vector of initial poles [rad/sec]
# %         dimension : (1,N)  
# %
# % weight: the rows in the system matrix are weighted using this array. Can be used 
# %         for achieving higher accuracy at desired frequency samples. 
# %         If no weighting is desired, use unitary weights: weight=ones(1,Ns). 
# %
# %         Two dimensions are allowed:
# %           dimension : (1,Ns) --> Common weighting for all vector elements.   
# %           dimension : (Nc,Ns)--> Individual weighting for vector elements.  
# %
# % opts.relax==1 --> Use relaxed nontriviality constraint 
# % opts.relax==0 --> Use nontriviality constraint of "standard" vector fitting
# %
# % opts.stable=0 --> unstable poles are kept unchanged
# % opts.stable=1 --> unstable poles are made stable by 'flipping' them
# %                   into the left half-plane
# % 
# %
# % opts.asymp=1 --> Fitting with D=0,  E=0 
# % opts.asymp=2 --> Fitting with D!=0, E=0 
# % opts.asymp=3 --> Fitting with D!=0, E!=0 
# %
# % opts.spy1=1 --> Plotting, after pole identification (A)
# %              figure(3): magnitude functions
# %                cyan trace  : (sigma*f)fit              
# %                red trace   : (sigma)fit
# %                green trace : f*(sigma)fit - (sigma*f)fit
# %
# % opts.spy2=1 --> Plotting, after residue identification (C,D,E) 
# %              figure(1): magnitude functions
# %              figure(2): phase angles  
# %
# % opts.logx=1 --> Plotting using logarithmic absissa axis             
# %
# % opts.logy=1 --> Plotting using logarithmic ordinate axis
# %
# % opts.errplot=1   --> Include deviation in magnitude plot
# %
# % opts.phaseplot=1 -->Show plot also for phase angle
# %
# %
# % opts.skip_pole=1 --> The pole identification part is skipped, i.e (C,D,E) 
# %                    are identified using the initial poles (A) as final poles.
# %                 
# % opts.skip_res =1 --> The residue identification part is skipped, i.e. only the 
# %                    poles (A) are identified while C,D,E are returned as zero. 
# %
# %
# % opts.cmplx_ss  =1 -->The returned state-space model has real and complex conjugate 
# %                    parameters. Output variable A is diagonal (and sparse). 
# %              =0 -->The returned state-space model has real parameters only.
# %                    Output variable A is square with 2x2 blocks (and sparse).
# %
# % OUTPUT :
# % 
# %     fit(s) = C*(s*I-(A)^(-1)*B +D +s.*E
# %
# % SER.A(N,N)    : A-matrix (sparse). If cmplx_ss==1: Diagonal and complex. 
# %                                Otherwise, square and real with 2x2 blocks. 
# %                           
# % SER.B(N,1)    : B-matrix. If cmplx_ss=1: Column of 1's. 
# %                       If cmplx_ss=0: contains 0's, 1's and 2's)
# % SER.C(Nc,N)   : C-matrix. If cmplx_ss=1: complex
# %                       If cmplx_ss=0: real-only
# % SERD.D(Nc,1)  : constant term (real). Is non-zero if asymp=2 or 3.
# % SERE.E(Nc,1)  : proportional term (real). Is non-zero if asymp=3.
# %
# % poles(1,N)    : new poles 
# %
# % rmserr(1) : root-mean-square error of approximation for f(s). 
# %                   (0 is returned if skip_res==1)  
# % fit(Nc,Ns): Rational approximation at samples. (0 is returned if
# %             skip_res==1).
# %
# %
# % APPROACH: 
# % The identification is done using the pole relocating method known as Vector Fitting [1], 
# % with relaxed non-triviality constraint for faster convergence and smaller fitting errors [2], 
# % and utilization of matrix structure for fast solution of the pole identifion step [3]. 
# %
# %******************************************************************************** 
# % NOTE: The use of this program is limited to NON-COMMERCIAL usage only.
# % If the program code (or a modified version) is used in a scientific work, 
# % then reference should be made to the following:  
# %
# % [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency       
# %     domain responses by Vector Fitting", IEEE Trans. Power Delivery,        
# %     vol. 14, no. 3, pp. 1052-1061, July 1999.                                
# %
# % [2] B. Gustavsen, "Improving the pole relocating properties of vector
# %     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
# %     July 2006.   
# %
# % [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
# %     "Macromodeling of Multiport Systems Using a Fast Implementation of
# %     the Vector Fitting Method", IEEE Microwave and Wireless Components 
# %     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
# %********************************************************************************
# % This example script is part of the vector fitting package (VFIT3.zip) 
# % Last revised: 08.08.2008. 
# % Created by:   Bjorn Gustavsen.
# %
# % 

import numpy as np
from numpy import imag, real, diag, zeros, ones, sqrt
from numpy.linalg import eigvals, eig, inv, norm
from scipy import signal
import matplotlib.pyplot as plt


def vectfit3(f,s,poles,weight,**kwargs):
        
    # %Tolerances used by relaxed version of vector fitting
    TOLlow = 1e-18; TOLhigh = 1e18;

    # if nargin<5
    #   opts=def;
    # else 
    #   %Merge default values into opts  
    #   A=fieldnames(def);    
    #   for m=1:length(A)
    # if ~isfield(opts,A(m))
    #   dum=char(A(m)); dum2=getfield(def,dum); opts=setfield(opts,dum,dum2);
    # end
    #   end  
    # end 
    
    opts = Definitions(**kwargs)
    
    # Input validation
    if (opts.relax!=0) and (opts.relax)!=1:
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.relax:{opts.relax}')
    if (opts.asymp!=1) and (opts.asymp)!=2 and (opts.asymp)!=3:
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.asymp:{opts.asymp}')
    if (opts.stable!=0) and (opts.stable!=1):
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.stable:{opts.stable}')
    if (opts.skip_pole!=0) and (opts.skip_pole)!=1:
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.skip_pole:{opts.skip_pole}')
    if (opts.skip_res!=0) and (opts.skip_res)!=1:
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.skip_res:{opts.skip_res}')
    if (opts.cmplx_ss!=0) and (opts.cmplx_ss)!=1:
      raise AttributeError(f'ERROR in vectfit3.m: ==> Illegal value for opts.cmplx_ss:{opts.cmplx_ss}')

    # # Input formatting
    # poles = poles.reshape((1))
    # a,b = poles.shape
    # if s[0] == 0 and a == 1:
    #     print(poles[0])
    #     if (poles[0] == 0 and poles[1] != 0):
    #         poles[0] = -1
    #     elif (poles[1]==0 and poles[0] != 0):
    #         poles[0] = -1
    #     elif poles[0] == 0 and poles[1] == 0:
    #         poles[0] = -1+1j*10
    #         poles[1] = -1-1j*10

    print(s.shape)
    s = s.reshape((1,len(s)))
    a,b=s.shape
    if a<b:
        s=s.T

    if len(f.shape) == 1:
        f = f.reshape((1, len(s)))
    else:
        row, col = f.shape
        if row>col:
            f=f.T

    if len(weight.shape) == 1:
        weight = weight.reshape((1, len(s)))
    else:
        row, col = weight.shape
        if row>col:
            weight=weight.T


    # % Some sanity checks on dimension of input arrays:
    if len(s)!=len(f[0,:]):
      raise ValueError('Error in vectfit3.m!!! ==> Second dimension of f does not match length of s.')

    if len(s) != len(weight[0,:]):
      raise ValueError('Error in vectfit3.m!!! ==> Second dimension of weight does not match length of s.')

    if len(weight[0,:])!=1:
      if len(weight[0,:])!=len(f[0,:]):
          raise ValueError('Error in vectfit3.m!!! ==> First dimension of weight is neither 1 nor matches first dimension of f.')

    # Initializing variables
    rmserr=[]; # %SERC=[];
    LAMBD=diag(poles)
    Ns=len(s)
    N=len(LAMBD);
    Nc=len(f[:,0])
    B=np.ones((N,1)) # %I=diag(ones(1,N));

    SERA=poles
    SERC=zeros((Nc,N))
    SERD=zeros((Nc,1))
    SERE=zeros((Nc,1))

    roetter=poles
    fit=zeros((Nc,Ns))
    
    weight=weight.T
    if len(weight[0,:])==1:
      common_weight=1
    elif len(weight[0,:])==Nc:
      common_weight=0
    else:
      print('ERROR in vectfit3.m: Invalid size of array weight')

    
    if opts.asymp==1:
      offs=0 
    elif opts.asymp==2:
      offs=1  
    else:
      offs=2

    # ========================================================================
    # ========================================================================
    # POLE IDENTIFICATION:
    # ========================================================================
    # ========================================================================

    if opts.skip_pole != 1:

        Escale = np.zeros(Nc + 1)

        # =======================================================
        # Finding out which starting poles are complex :
        # =======================================================
        cindex = np.zeros(N)
        for m in range(N):
            if np.imag(LAMBD[m, m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        # =======================================================
        # Building system - matrix :
        # =======================================================
        Dk = np.zeros((Ns, N))
        for m in range(N):
            if cindex[m] == 0:  # real pole
                Dk[:, m] = 1. / (s - LAMBD[m, m])
            elif cindex[m] == 1:  # complex pole, 1st part
                Dk[:, m] = (1. / (s - LAMBD[m, m]) + 1. / (s - np.conj(LAMBD[m, m]))).reshape(Ns)
                Dk[:, m + 1] = (1j / (s - LAMBD[m, m]) - 1j / (s - np.conj(LAMBD[m, m]))).reshape(Ns)

        if opts.asymp == 1 or opts.asymp == 2:
            Dk[:, N-1] = 1
        elif opts.asymp == 3:
            Dk[:, N-1] = 1
            Dk[:, N] = s

        # Scaling for last row of LS-problem (pole identification)
        scale = 0
        for m in range(Nc):
            if len(weight[0, :]) == 1:
                scale = scale + (np.linalg.norm(weight[:, 0] * f[m, :])) ** 2
            else:
                scale = scale + (np.linalg.norm(weight[:, m] * f[m, :])) ** 2

        scale = np.sqrt(scale) / Ns

        if opts.relax == 1:
            AA = np.zeros((Nc * (N + 1), N + 1))
            bb = np.zeros((Nc * (N + 1), 1))
            Escale = np.zeros((1, len(AA[0, :])))

            for n in range(1, Nc + 1):
                A = np.zeros((Ns, (N + offs) + N + 1))

                if common_weight == 1:
                    weig = weight
                else:
                    weig = weight[:, n - 1]
                weig = weig.reshape(Ns)

                # TODO: Confirm range with offset
                for m in range(0, N + offs):  # left block
                    A[0:Ns, m] = (weig * Dk[0:Ns, m])

                inda = N + offs
                for m in range(0, N + 1):  # right block
                    A[0:Ns, inda + m - 1] = -weig * Dk[0:Ns, m - 1] * f[n - 1, 0:Ns].T

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
                AA[(n - 1) * (N + 1):n * (N + 1), :] = R22

                if n == Nc:
                    bb[(n - 1) * (N + 1):n * (N + 1), 0] = Q[-1, N + offs:n * (N + offs + N + 1)] * Ns * scale

                for col in range(1, len(AA[0, :]) + 1):
                    Escale[0, col - 1] = 1 / np.linalg.norm(AA[:, col - 1])
                AA[:, col - 1] = Escale[0, col - 1] * AA[:, col - 1]

                x = np.linalg.lstsq(AA, bb, rcond=None)[0]
                x = x * Escale.T

        # Situation: No relaxation, or produced D of sigma extremely small and large. Solve again, without relaxation
        if opts.relax == 0 or abs(x[-1]) < TOLlow or abs(x[-1]) > TOLhigh:
            AA = np.zeros((Nc * (N), N))
            bb = np.zeros((Nc * (N), 1))
            if opts.relax == 0:
                Dnew = 1
            else:
                if x[-1] == 0:
                    Dnew = 1
                elif abs(x[-1]) < TOLlow:
                    Dnew = np.sign(x[-1]) * TOLlow
                elif abs(x[-1]) > TOLhigh:
                    Dnew = np.sign(x[-1]) * TOLhigh

            for n in range(Nc):
                A = np.zeros((Ns, (N + offs) + N))
            Escale = np.zeros(N)

            if common_weight == 1:
                weig = weight
            else:
                weig = weight[:, n]

            for m in range(N + offs):
                A[0:Ns, m] = weig * Dk[0:Ns, m]

            inda = N + offs
            for m in range(N):
                A[0:Ns, inda + m] = -weig * Dk[0:Ns, m] * f[n, 0:Ns].T

            b = Dnew * weig * f[n, 0:Ns].T
            A = np.concatenate((np.real(A), np.imag(A)))
            b = np.concatenate((np.real(b), np.imag(b)))

            offset = N + offs
            Q, R = np.linalg.qr(A, mode='reduced')
            ind1 = N + offs + 1
            ind2 = N + offs + N
            R22 = R[ind1 - 1:ind2, ind1 - 1:ind2]
            AA[n * N:(n + 1) * N, :] = R22
            bb[n * N:(n + 1) * N, 0] = np.dot(Q[:, ind1 - 1:ind2].T, b)

            for col in range(len(AA[0, :])):
                Escale[col] = 1. / np.linalg.norm(AA[:, col])
            AA[:, col] = Escale[col] * AA[:, col]

            # if opts.use_normal == 1:
            #    x = np.dot(AA.T, AA) \ np.dot(AA.T, bb)
            # else:
            x = np.linalg.lstsq(AA, bb, rcond=None)[0]
            # end
            x = x * Escale
            x = np.concatenate((x, Dnew)).T

        # ************************************

        C = x[:-1]
        D = x[-1]  # NEW!!

        # We now change back to make C complex :
        # **************
        for m in range(N):
            if cindex[m] == 1:
                for n in range(1):  # Nc+1
                    r1 = C[m]
                    r2 = C[m + 1]
                    C[m] = r1 + 1j * r2
                    C[m + 1] = r1 - 1j * r2
        # **************

        # plotting for first stage of vector fitting
        if opts.spy1 == 1:
            RES3 = np.zeros(D.shape)
            Dk = np.zeros((Ns, N))
            for m in range(N):
                Dk[:, m] = 1. / (s - LAMBD[m, m])
            RES3[:, 0] = D + np.dot(Dk, C)  # (sigma)rat
            freq = s / (2 * np.pi * 1j)
            if opts.logx == 1:
                if opts.logy == 1:
                    fig, ax = plt.subplots()
                    ax.loglog(freq, np.abs(RES3), 'b')
                    ax.set_xlim([freq[0], freq[Ns - 1]])
                else:  # logy=0
                    fig, ax = plt.subplots()
                    ax.semilogx(freq, np.abs(RES3), 'b')
                    ax.set_xlim([freq[0], freq[Ns - 1]])
            else:  # logx=0
                if opts.logy == 1:
                    fig, ax = plt.subplots()
                    ax.semilogy(freq, np.abs(RES3), 'b')
                    ax.set_xlim([freq[0], freq[Ns - 1]])
                else:  # logy=0
                    fig, ax = plt.subplots()
                    ax.plot(freq, np.abs(RES3), 'b')
                    ax.set_xlim([freq[0], freq[Ns - 1]])
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude')
            # ax.set_title('Sigma')
            if opts.legend == 1:
                ax.legend(['sigma'])
            plt.draw()

        # =========================================
        # We now calculate the zeros for sigma :
        # =========================================
        # oldLAMBD=LAMBD;oldB=B;oldC=C;
        m = 0
        for n in range(N):
            m += 1
            if m < N:
                if abs(LAMBD[m - 1, m - 1]) > abs(np.real(LAMBD[m - 1, m - 1])):  # complex number?
                    LAMBD[m, m - 1] = -np.imag(LAMBD[m - 1, m - 1])
                    LAMBD[m - 1, m] = np.imag(LAMBD[m - 1, m - 1])
                    LAMBD[m - 1, m - 1] = np.real(LAMBD[m - 1, m - 1])
                    LAMBD[m, m] = LAMBD[m - 1, m - 1]
                    B[m - 1, 0] = 2
                    B[m, 0] = 0
                    koko = C[m - 1]
                    C[m - 1] = np.real(koko)
                    C[m] = np.imag(koko)
                    m += 1

        ZER = LAMBD - np.matmul(B, np.transpose(C)) / D
        roetter = np.linalg.eig(ZER)[0].real
        unstables = roetter > 0

        if opts.stable == 1:
            roetter[unstables] = roetter[unstables] - 2 * roetter[unstables].real

        roetter = np.sort(roetter)
        N = roetter.size

        # =============================================
        # Sort poles so that the real ones come first:
        for n in range(N):
            for m in range(n + 1, N):
                if np.imag(roetter[m]) == 0 and np.imag(roetter[n]) != 0:
                    trans = roetter[n]
                    roetter[n] = roetter[m]
                    roetter[m] = trans

        N1 = 0
        for m in range(N):
            if np.imag(roetter[m]) == 0:
                N1 = m

        if N1 < N:
            roetter[N1 + 1:N] = np.sort(roetter[N1 + 1:N])  # N1: n.o. real poles
        # N2=N-N1;                                             # N2: n.o. imag.poles

        roetter = roetter - 2j * np.imag(roetter)  # 10.11.97 !!!
        SERA = roetter

        #%=========================================================================
        #%=========================================================================
        #%          RESIDUE IDENTIFICATION:
        #%=========================================================================
        #%=========================================================================

        if opts.skip_res != 1:

            # We now calculate SER for f, using the modified zeros of sigma as new poles:

            # LAMBD=roetter
            LAMBD = roetter

            # B=ones(N,1)
            B = np.ones((N, 1))

            # Finding out which poles are complex:
            cindex = np.zeros(N)
            for m in range(N):
                if np.imag(LAMBD[m]) != 0:
                    if m == 0:
                        cindex[m] = 1
                    else:
                        if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                            cindex[m] = 1
                            cindex[m + 1] = 2
                        else:
                            cindex[m] = 2

        # We now calculate the SER for f (new fitting), using the above calculated zeros as known poles:
        if opts.asymp == 1:
            A = np.zeros((2 * Ns, N))
            BB = np.zeros((2 * Ns, Nc))
        elif opts.asymp == 2:
            A = np.zeros((2 * Ns, N + 1))
            BB = np.zeros((2 * Ns, Nc))
        else:
            A = np.zeros((2 * Ns, N + 2))
            BB = np.zeros((2 * Ns, Nc))

        # I3 = np.diag(np.ones(Nc))
        # I3[:, Nc] = []
        Dk = np.zeros((Ns, N))
        for m in range(N):
            if cindex[m] == 0:  # real pole
                Dk[:, m] = (1 / (s - LAMBD[m])).reshape(Ns)
            elif cindex[m] == 1:  # complex pole, 1st part
                Dk[:, m] = (1 / (s - LAMBD[m]) + 1 / (s - LAMBD[m].conj())).reshape(Ns)
                Dk[:, m + 1] = (1j / (s - LAMBD[m]) - 1j / (s - LAMBD[m].conj())).reshape(Ns)

        # If all frequency data is equally weighted
        if common_weight == 1:
            Dk = np.zeros((Ns, N))
            # Obtain the Dk matrix
            for m in range(N):
                if cindex[m] == 0:  # real pole
                    Dk[:, m] = (weight / (s - LAMBD[m])).reshape(Ns)
                elif cindex[m] == 1:  # complex pole, 1st part
                    Dk[:, m] = (weight / (s - LAMBD[m]) + weight / (s - LAMBD[m].conj())).reshape(Ns)
                    Dk[:, m + 1] = (1j * weight / (s - LAMBD[m]) - 1j * weight / (s - LAMBD[m].conj())).reshape(Ns)

            # Create the A matrix for the least squares: Ax=b
            if opts.asymp == 1:
                A[0:Ns, 0:N] = Dk
            elif opts.asymp == 2:
                A[0:Ns, 0:N] = Dk
                A[0:Ns, N] = weight
            else:
                A[0:Ns, 0:N] = Dk
                A[0:Ns, N] = weight
                A[0:Ns, N + 1] = weight.reshape(Ns) * s.reshape(Ns)

            # Create the b matrix for the least squares: Ax=b
            for m in range(Nc):
                BB[0:Ns, m] = weight.reshape(Ns) * f[m, :].reshape(Ns)

            A[Ns:2 * Ns, :] = np.imag(A[0:Ns, :])
            A[0:Ns, :] = np.real(A[0:Ns, :])
            BB[Ns:2 * Ns, :] = np.imag(BB[0:Ns, :])
            BB[0:Ns, :] = np.real(BB[0:Ns, :])

            if opts.asymp == 2:
                A[0:Ns, N] = A[0:Ns, N]
            elif opts.asymp == 3:
                A[Ns + 1:2 * Ns, N + 1] = A[Ns + 1:2 * Ns, N + 1]

        # clear Escale
        Escale = np.zeros((1, A.shape[1]))
        for col in range(A.shape[1]):
            Escale[0, col] = np.linalg.norm(A[:, col], ord=2)
            A[:, col] = A[:, col] / Escale[0, col]

        # Solve system of linear equations
        X = np.linalg.lstsq(A, BB, rcond=None)[0]
        # Normalize the solution with respect to E_scale
        for n in range(Nc):
            X[:, n] = X[:, n] / Escale.T

        # clear A
        X = X.T
        C = X[:, :N]
        if opts.asymp == 2:
            SERD = X[:, N]
        elif opts.asymp == 3:
            SERE = X[:, N + 1]
            SERD = X[:, N]
        else:
            SERD = np.zeros((Nc, 1))
            SERE = np.zeros((Nc, 1))
            C = np.zeros((Nc, N))
            for n in range(Nc):
                if opts.asymp == 1:
                    A[0:Ns, 0:N] = Dk
                elif opts.asymp == 2:
                    A[0:Ns, 0:N] = Dk
                    A[0:Ns, N] = 1
                else:
                    A[0:Ns, 0:N] = Dk
                    A[0:Ns, N] = 1
                    A[0:Ns, N + 1] = s

                for m in range(A.shape[1]):
                    A[0:Ns, m] = weight[:, n] * A[0:Ns, m]

                BB = weight[:, n] * f[n, :].T
                A[Ns + 1:2 * Ns, :] = np.imag(A[0:Ns, :])
                A[0:Ns, :] = np.real(A[0:Ns, :])
                BB[Ns + 1:2 * Ns] = np.imag(BB[0:Ns])
                BB[0:Ns] = np.real(BB[0:Ns])

                if opts.asymp == 2:
                    A[0:Ns, N] = A[0:Ns, N]
                elif opts.asymp == 3:
                    A[0:Ns, N] = A[0:Ns, N]
                    A[Ns + 1:2 * Ns, N + 1] = A[Ns + 1:2 * Ns, N + 1]

                # clear Escale
                Escale = np.zeros((1, A.shape[1]))
                for col in range(A.shape[1]):
                    Escale[0, col] = np.linalg.norm(A[:, col], ord=2)
                    A[:, col] = A[:, col] / Escale[0, col]

                x = np.linalg.lstsq(A, BB, rcond=None)[0]
                x = x / Escale.T

                if opts.asymp == 2:
                    SERD[n] = x[N]
                elif opts.asymp == 3:
                    SERE[n] = x[N + 1]
                    SERD[n] = x[N]
                else:
                    C[n, 0:N] = x[0:N].T

        # %===============================================================
        # We now change back to make C complex.
        for m in range(N):
            if cindex[m] == 1:
                for n in range(Nc):
                    r1 = C[n, m]
                    r2 = C[n, m + 1]
                    C[n, m] = r1 + 1j * r2
                    C[n, m + 1] = r1 - 1j * r2

        B = np.ones((N, 1))

        SERA = LAMBD
        SERB = B
        SERC = C

        Dk = np.zeros((Ns, N))
        for m in range(N):
            Dk[:, m] = 1 / (s - SERA[m])

        for n in range(Nc):
            fit_n = np.dot(Dk, SERC[n, :]).T
            if opts.asymp == 2:
                fit_n += SERD[n]
            elif opts.asymp == 3:
                fit_n += SERD[n] + s * SERE[n]
            fit[n, :] = fit_n

        fit = fit.T
        f = f.T
        diff = fit - f
        rmserr = np.sqrt(np.sum(np.abs(diff) ** 2)) / np.sqrt(Nc * Ns)


        # PLOTTING
        if opts.spy2 == 1:
            freq = s / (2 * np.pi * 1j)
            if opts.logx == 1:
                if opts.logy == 1:
                    fig1, ax1 = plt.subplots()
                    ax1.loglog(freq, abs(f), 'b')
                    ax1.loglog(freq, abs(fit), 'r--')
                    if opts.errplot == 1:
                        ax1.loglog(freq, abs(f - fit), 'g')
                else:
                    fig1, ax1 = plt.subplots()
                    ax1.semilogx(freq, abs(f), 'b')
                    ax1.semilogx(freq, abs(fit), 'r--')
                    if opts.errplot == 1:
                        ax1.semilogx(freq, abs(f - fit), 'g')
                if opts.phaseplot == 1:
                    fig2, ax2 = plt.subplots()
                    ax2.semilogx(freq, 180 * np.unwrap(np.angle(f)) / np.pi, 'b')
                    ax2.semilogx(freq, 180 * np.unwrap(np.angle(fit)) / np.pi, 'r--')
            else:
                if opts.logy == 1:
                    fig1, ax1 = plt.subplots()
                    ax1.semilogy(freq, abs(f), 'b')
                    ax1.semilogy(freq, abs(fit), 'r--')
                    if opts.errplot == 1:
                        ax1.semilogy(freq, abs(f - fit), 'g')
                else:
                    fig1, ax1 = plt.subplots()
                    ax1.plot(freq, abs(f), 'b')
                    ax1.plot(freq, abs(fit), 'r--')
                    if opts.errplot == 1:
                        ax1.plot(freq, abs(f - fit), 'g')
                if opts.phaseplot == 1:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(freq, 180 * np.unwrap(np.angle(f)) / np.pi, 'b')
                    ax2.plot(freq, 180 * np.unwrap(np.angle(fit)) / np.pi, 'r--')

            ax1.set(xlim=[freq[0], freq[Ns]], xlabel='Frequency [Hz]', ylabel='Magnitude')
            if opts.legend == 1:
                if opts.errplot == 1:
                    ax1.legend(['Data', 'FRVF', 'Deviation'])
                else:
                    ax1.legend(['Data', 'FRVF'])
            if opts.phaseplot == 1:
                ax2.set(xlim=[freq[0], freq[Ns]], xlabel='Frequency [Hz]', ylabel='Phase angle [deg]')
                if opts.legend == 1:
                    ax2.legend(['Data', 'FRVF'])

            plt.draw()
            plt.show()

        fit = np.transpose(fit)

    A = SERA
    poles = A
    if opts.skip_res!=1:
        B = SERB
        C = SERC
        D = SERD
        E = SERE
    else:
        B = ones(N, 1)
        C = zeros(Nc, N)
        D = zeros(Nc, Nc)
        E = zeros(Nc, Nc)
        rmserr = 0
    # %============================================================
    # % Convert into real state - space model
    # %============================================================
    if opts.cmplx_ss != 1:
        print(A,A.diagonal())
        # A = sparse.diags(A.diagonal())  # Convert A to a sparse diagonal matrix

        cindex = np.zeros(N)
        for m in range(N):
            if np.imag(A[m, m]) != 0:
                if m == 0:
                    cindex[m] = 1
                else:
                    if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                        cindex[m] = 1
                        cindex[m + 1] = 2
                    else:
                        cindex[m] = 2

        n = 0
        for m in range(N):
            n = n + 1
            if cindex[m] == 1:
                a = A[n - 1, n - 1]
                a1 = np.real(a)
                a2 = np.imag(a)
                c = C[:, n - 1]
                c1 = np.real(c)
                c2 = np.imag(c)
                b = B[n - 1, :]
                b1 = 2 * np.real(b)
                b2 = -2 * np.imag(b)
                Ablock = np.array([[a1, a2], [-a2, a1]])

                A[n - 1:n + 1, n - 1:n + 1] = Ablock
                C[:, n - 1] = c1
                C[:, n] = c2
                B[n - 1, :] = b1
                B[n, :] = b2

    # else:
    #   A = sparse.diags(A.diagonal())  # A is complex, make it diagonal

    SER = {'A':A,
           'B':B,
           'C':C,
           'D':D,
           }

    return SER,poles,rmserr,fit,opts

#%% ==============================================
import control as ctrl
zeros_real = np.poly1d([1])
zeros_init = np.poly1d([1])
poles_real = np.poly1d([1,1,1,1,1])
poles_init = np.poly1d([1.1,0.9,0.95,1,1])


G = ctrl.tf(np.array(zeros_real),np.array(poles_real))

freq = np.linspace(0,1e3,num=100)
data = np.array([abs(G(s)) for s in freq])

f = data
s = 1j*freq


# residues,d,h,poles_new = mvf(freq,data,poles_init.roots,zeros_init.roots)
weight = ones(len(s))

SER,poles_new,rmserr,fit,opts = vectfit3(f,s,poles_init.roots,weight,asymp=1) # ,opts,**kwargs

asdf
print(poles_real.roots)
print(poles_init.roots)
print(poles_new)


