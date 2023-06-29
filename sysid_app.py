# Control
import control as ctrl

# numerical packages
import numpy as np
from numpy.linalg import inv, det, eigvals, eig, norm 
from numpy import sqrt, log, real, imag
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint
from scipy import stats, signal, linalg

# plot packages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib import cm
import linecache

# Control systems
from control import obsv

# # Default 
# plt.rcParams.update({'lines.markeredgewidth': 1})
# plt.rcParams.update({'font.size':18})
# # plt.rcParams['text.usetex'] = False
# plt.rcParams['text.usetex'] = False
# plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath} \usepackage{amssymb}"
prop_cycle = plt.rcParams['axes.prop_cycle']
clrs = prop_cycle.by_key()['color']

#
# from scipy.stats import norm
# Other packages
from tqdm import tqdm
import SITOBBDS_utils as utils
from numerical_methods import NumericalMethods
import datetime
import copy
import os
import pandas as pd

def get_machine_precision(show=True):
    float32 = np.finfo(np.float32).eps
    float64 = np.finfo(np.float64).eps
    if show:
        print('float32:\t',float32)
        print('float64:\t',float64)

    return float32, float64


# https://pypi.org/project/sysidentpy/
# class ParamWrapper:
#     def __init__(self,val:float,base:float,unit:str=None):
#         self.v = val
#         self.base = base
#         self.unit = unit
#         return
#
#     def __get__(self):
#         return self.val



class ParamWrapper:
    
    def __init__(self,params,model,pu:bool=True,seq=False):
        # ------ Load default system parameters ------
        self.Vbase = 66e3
        self.Sbase = 100e6
        if pu: self.V = self.Vbase/self.Vbase
        self.V_amp = self.Vbase * np.sqrt(2)
        self.f = 50
        self.omega = 2*np.pi*self.f
        self.phi = 0

        designed_models = ['c1_s0_o3','c1_s0_o3_load','c1_s0_o3_load_test','c1_s0_o2','c1_s0_o2_load','c1_s0_o2_ZY','c1_s0_o3_load1','c1_s0_o3_load2','c1_s0_pde_load']
       
        alpha = np.exp(2/3*np.pi*1j)

        self.T_seq = np.array([[1, 1,           1],
                               [1, alpha**2,    alpha],
                               [1, alpha,       alpha**2]])

        # ------ Load model default parameters 1-conductor system ------
        self.Rin = 0.05 # BRK = 0.05

        self.Rload = 200 # BRK = 0.05
        if model in designed_models and 'test' not in model and 'grid' not in model:
            self.params = ['Rin','Rload','R','L','C','G']
            #self.params = ['Rin','Rload','R','L','C1','C2']
            self.R = 5.50451976 # 0.5
            self.L = 0.575768561E+02/self.omega # 1e-3
            self.C = 0.585164562E-02/self.omega/2 # / 2 # 1e-6/2
            self.G = 0.179068548E-04/2 # 0 => G= 1
            self.C1 = self.C
            self.C2 = self.C
        elif model in designed_models and 'test' in model:
            self.params = ['Rin','Rload','R','L','C','G']
            #self.params = ['Rin','Rload','R','L','C1','C2']
            self.R = 0.5 # 0.5
            self.L = 1e-3/self.omega # 1e-3
            self.C = 1e-6 # / 2 # 1e-6/2
            self.G = 0 # 0 => G= 1
            self.C1 = self.C
            self.C2 = self.C

           
        else:
            if seq:
                self.params = [f'{ABC}{i}' for ABC in ['R','L','G','C'] for i in range(3)] + ['Rin','Rload']
    
                self.n = n = 3
                path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + f'\\data\\{model}'
                self.Z, self.Y = self.load_cable_data(path, model, 3)
                for i, idx in enumerate([0,1,2]):
                    setattr(self,f'R{idx}',self.Z_seq[i,i].real)
                    setattr(self,f'L{idx}',self.Z_seq[i,i].imag / self.omega )
                    setattr(self,f'G{idx}',self.Y_seq[i,i].real )
                    setattr(self,f'C{idx}',self.Y_seq[i,i].imag / self.omega)

            else:
                self.params = [f'{ABC}{i}{j}' for ABC in ['R','L','G','C'] for i in range(3) for j in range(3)] + ['Rin']
    
                if 'grid' in model:
                    self.SCR = 10
                    self.XR = 10
                    self.Zg = self.Vbase**2/(self.SCR*self.Sbase)
                    theta = np.arctan(self.XR)
                    self.Rg = self.Zg * np.cos(theta)
                    self.Xg = self.Zg * np.sin(theta)
                    self.Lg = self.Xg/self.omega
                    self.params += ['SCR','XR','Rg','Lg']
                else:
                    self.params += ['Rload']
                    
    
                self.n = n = 3
                path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + f'\\data\\{model}'
                self.Z, self.Y = self.load_cable_data(path, model, 3)
                for i in range(3):
                    for j in range(3):                        
                        setattr(self,f'R{i}{j}',self.Z[i,j].real)
                        setattr(self,f'L{i}{j}',self.Z[i,j].imag / self.omega)
                        setattr(self,f'G{i}{j}',self.Y[i,j].real)
                        setattr(self,f'C{i}{j}',self.Y[i,j].imag / self.omega)

        # ------ Get custom parameters ------
        if params is not None:
            for k, v in params.items():
                setattr(self,k,v)
        
        if ('SCR' in params.keys() or 'XR' in params.keys()) and 'grid' in model:
            rg, xg, lg = self.calculate_grid_impedance(self.SCR, self.XR)
            self.Rg = rg
            self.Xg = xg
            self.Lg = lg

        # ------ Per unitizing ------
        if pu:
            self.Ibase = self.Sbase / (np.sqrt(3) * self.Vbase)
            self.Zbase = self.Vbase**2 / self.Sbase           
    
            # Adjusting 
            # TODO: HANDLE THIS CONTROL FLOW PROPERLY FOR CABLE SYSTEMS!
            # if model in designed_models:
            for k in self.params:
                if k not in ['SCR','XR']:
                    # print(k,getattr(self, k),getattr(self, k)/self.Zbase)
                    setattr(self, k, getattr(self, k)/self.Zbase)

        # ------ Converting param list into dict ------
        #
        # if model in designed_models:
        data = {}
        for p in self.params:
            
            data[p] = getattr(self, p)
        self.params = data
        return

    def calculate_grid_impedance(self,scr,xr):
        self.Zg = Zg = self.Vbase**2/(scr*self.Sbase)
        theta = np.arctan(xr)
        rg = Zg * np.cos(theta)
        xg = Zg * np.sin(theta)
        lg = self.Xg/self.omega    
        
        return rg, xg, lg

    def load_cable_data(self,path, file, n_conductors, mode: int = 0):
        n = n_conductors
        if 'grid' in file:
            file = file[:-5]
            path = path[:-5]
        fpath = f'{path}\\{file}.out'
        
        select_mode = ['LONG-LINE CORRECTED ',
                  'SEQUENCE ',
                  ''][mode]

        z_searchword = f'{select_mode}SERIES IMPEDANCE MATRIX'.replace(('','SERIES ')[select_mode == 'SEQUENCE '],'')
        y_searchword = f'{select_mode}SHUNT ADMITTANCE MATRIX'.replace(('','SHUNT ')[select_mode == 'SEQUENCE '],'')

        z_searchword = f'LONG-LINE CORRECTED SERIES IMPEDANCE MATRIX'
        y_searchword = f'LONG-LINE CORRECTED SHUNT ADMITTANCE MATRIX'
        
        conv = {0: lambda x: str(x)}

        # df = pd.read_csv(fpath, skiprows=59,nrows=n,converters=conv)
        Z = np.zeros((n,n),dtype=np.complex128)
        Y = np.zeros((n,n),dtype=np.complex128)

        print(z_searchword,y_searchword,fpath)

        with open(fpath,'r') as f:
            for i, line in enumerate(f):
                cnt = 0
                # if 'SERIES IMPEDANCE MATRIX (Z)' in line:
                if z_searchword in line:
                    # print(f'line: {i}')
                    zline = i + 1 + 1
                    # z = np.loadtxt(fpath,skiprows=i+1,max_rows=7,converters=conv,delimiter=',')
                # elif 'SHUNT ADMITTANCE MATRIX (Y)' in line:
                elif y_searchword in line:
                    yline = i + 1 + 1
                    # y = np.genfromtxt(fpath,skip_header=i+1,max_rows=7,autostrip=True)            
        f.close()

        for i in range(n):
            z_i = linecache.getline(fpath, zline).strip().split(' '*3)
            y_i = linecache.getline(fpath, yline).strip().split(' '*3)
            
            Z[i,:] = [complex(float(x.split(',')[0]),float(x.split(',')[1])) for x in z_i]
            Y[i,:] = [complex(float(x.split(',')[0]),float(x.split(',')[1])) for x in y_i]

            zline += 1
            yline += 1
            
        # print('Z\n',Z)
        # print('Y\n',Y)

        if select_mode != 'SEQUENCE ':
            self.Z_ph = Z
            self.Y_ph = Y
            # self.z_ph = Z/self.Zbase
            # self.y_ph = Y/self.Zbase
            self.Z_seq = inv(self.T_seq) @ Z @ self.T_seq
            self.Y_seq = inv(self.T_seq) @ Y @ self.T_seq

        return Z,Y

class OptionWrapper:
    
    def __init__(self,opts):
        self.err_tol = 1e-12
        self.method = 'BFGS'        
        self.hess = None        
        self.hessp = None        
        self.jac = None
        self.disp = False
        self.verbose = False
        self.epsilon = 1e-5
        self.cnstr_lb_factor = 0.5
        self.cnstr_ub_factor = 1.5

        # ------ Get custom parameters ------
        if opts is not None:
            for k, v in opts.items():
                setattr(self,k,v)

        
        return

class SITOBBDS:
    def __init__(self,opts:dict=None,N_pi=1):
        self.method = NumericalMethods()
        self.models = ['c1_s0_o3',
                       'c1_s0_o3_load',
                       'c1_s0_o2',
                       'Cable_2']

        self.opts = OptionWrapper(opts)
        self.opt_params = None
        self.MLE_silence = False
        self.N_pi = N_pi
        

        return
        
    def load_model(self,model=None,p=None,verbose:bool = False, seq=False):
        if model is None: model = self.model
        # if p is None: p = self.p
        if p is None: p = copy.deepcopy(self.p)
                    
        # Create model
        if isinstance(model, (list,tuple)):
            # ------ CUSTOM MODEL ------ 
            model='Custom'
            self.n = n = 1
            raise NotImplementedError('')
            
        elif isinstance(model,str) and model == 'c1_s0_o3':
            # ------ Single core, no screen, 3rd order ------ 
            self.n = n = 3
            p.C1 = p.C2 = p.C
            

            A = np.array([[-p.R/p.L, 1/p.L, -1/p.L],
                            [-1/(p.C1), -1/(p.Rin * p.C1), 0],
                            [1/(p.C2), 0, 0],
                          ])
            
            B = np.array([[0], [1 / (p.Rin * p.C1)], [0]])
            C = np.eye(n)
            D = np.zeros(1)

        elif isinstance(model,str) and model == 'c1_s0_o3_load':
            # ------ Single core, no screen, 3rd order ------   
            self.n = n = 3
            G = p.G
            Gk = p.G + 1/p.Rin            
            Gk = 1/p.Rin            
            Gm = p.G + 1/p.Rload   
            Gm = 1/p.Rload   

            # A = np.array([
            #                 [-1/(p.Rin * p.C1), -1/(p.C1), 0],
            #                 [1/p.L, -p.R/p.L, -1/p.L],
            #                 [0, 1/(p.C2), -1/(p.Rload*p.C2)],
            #               ])
            
            A = np.array([
                            [-Gk/p.C, -1/p.C, 0],
                            [1/p.L, -p.R/p.L, -1/p.L],
                            [0, 1/p.C, -Gm/p.C],
                          ])
            
            B = np.array([[Gk / p.C], [(p.R)/p.L], [0]])
            # B = np.array([[1 / (p.C)], [0], [0]])
            C = np.eye(n)
            D = np.zeros(1)
            
        elif isinstance(model,str) and model == 'c1_s0_o3_load_test':
            # ------ Single core, no screen, 3rd order ------   
            self.n = n = 3
            G = p.G
            Gk = p.G + 1/p.Rin            
            Gk = 1/p.Rin            
            Gm = p.G + 1/p.Rload   
            Gm = 1/p.Rload   

            # A = np.array([
            #                 [-1/(p.Rin * p.C1), -1/(p.C1), 0],
            #                 [1/p.L, -p.R/p.L, -1/p.L],
            #                 [0, 1/(p.C2), -1/(p.Rload*p.C2)],
            #               ])
            
            A = np.array([
                            [-Gk/p.C, -1/p.C, 0],
                            [1/p.L, -p.R/p.L, -1/p.L],
                            [0, 1/p.C, -Gm/p.C],
                          ])
            
            B = np.array([[Gk / p.C], [(p.R)/p.L], [0]])
            # B = np.array([[1 / (p.C)], [0], [0]])
            C = np.eye(n)
            D = np.zeros(1)
            
        elif isinstance(model,str) and model == 'c1_s0_pde_load':
            # ------ Single core, no screen, 3rd order ------             
            self.n = n = 5
            p.C1 = p.C2 = p.C
            C1, C2, C3 = p.C1/2, p.C1, p.C1/2
            L1 = L2 = p.L/2
            R1 = R2 = p.R/2
            a11 = -1/(p.Rin * C1)
            a12 = -1/(C1)
            a21 = 1/L1
            a22 = -R1/L1  
            a23 = -1/L1
            a32 = 1/(C2)
            a34 = -1/(C2)
            a43 = 1/L2
            a44 = -R2/L2
            a45 = -1/L2
            a54 = 1/(C3)
            a55 = -1/(p.Rload*C3)
            A = np.array([
                            [a11,   a12,    0,      0,      0],
                            [a21,   a22,    a23,    0,      0],
                            [0,     a32,    0,      a34,    0],
                            [0,     0,      a43,    a44,    a45],
                            [0,     0,      0,      a54,    a55],                            
                          ])
            
            B = np.array([[1 / (p.Rin * C1)], [0], [0], [0], [0]])
            C = np.eye(n)
            D = np.zeros(1)            


        elif isinstance(model,str) and model == 'c1_s0_o2':
            # ------ Single core, no screen, 2rd order ------ 
            self.n = n = 2
            A = np.array([[-p.R/p.L,-1/(p.L)],
                          [1/(p.C2),0]])

            B = np.array([[1/p.L],[0]])

            C = np.eye(n)
            D = np.zeros(1)
            
        elif isinstance(model,str) and model == 'c1_s0_o2_load':
            # ------ Single core, no screen, 2rd order ------ 
            self.n = n = 2
            A = np.array([[-p.R/p.L,-1/(p.L)],
                          [1/(p.C2),-1/(p.Rload*p.C2)]])
            B = np.array([[1/p.L],[0]])

            C = np.eye(n)
            D = np.zeros(1)

            self.n = n = 2
            G = p.G
            Gk = p.G + 1/p.Rin            
            Gk = 1/p.Rin            
            Gm = p.G + 1/p.Rload   
            # Gm = 0

            # A = np.array([
            #                 [-1/(p.Rin * p.C1), -1/(p.C1), 0],
            #                 [1/p.L, -p.R/p.L, -1/p.L],
            #                 [0, 1/(p.C2), -1/(p.Rload*p.C2)],
            #               ])
            
            A = np.array([
                            [-p.R/p.L, -1/p.L],
                            [1/p.C, -Gm/p.C],
                          ])
            
            B = np.array([[1/p.L], [0]])
            # B = np.array([[1 / (p.C)], [0], [0]])
            C = np.eye(n)
            D = np.zeros(1)


        elif 'grid' in model:
            if self.opt_params is not None:
                if len([k for k in self.opt_params if k in ['SCR','XR']]) >= 1:
                    Rg, xg, Lg = p.calculate_grid_impedance(p.SCR,p.XR)
                    Lg = max(1e-9,Lg)

                else:
                    Rg = p.Rg
                    Lg = max(1e-9,p.Lg)
            else:
                Rg = p.Rg
                Lg = max(1e-9,p.Lg)
            
            N_phi = 3
            N_pi = 100
            self.n = n = 12

            # Prepare matrices
            A = np.zeros((n,n))
            B = np.zeros((n,6))              
            I = np.eye(3)

            # Impedance 
            Gin = I*1/p.Rin
            Gload = I*1/p.Rload
            Rin = I*p.Rin
            Gin = I*1/p.Rin
            Rgrid = I*Rg
            Lgrid = I*Lg
            R = np.zeros((3,3))
            L = np.zeros((3,3))
            Gk = np.zeros((3,3))
            Gm = np.zeros((3,3))
            Ck = np.zeros((3,3))
            Cm = np.zeros((3,3))
            C = np.zeros((3,3))
            G = np.zeros((3,3))
                                            
            for i in range(N_phi):
                for j in range(N_phi):
                    if i == j:
                        R[i,j]  = getattr(p, f'R00')/N_pi
                        L[i,j]  = getattr(p, f'L00')/N_pi
                        Gk[i,j] = getattr(p, f'G00')/N_pi + Gin[i,j]
                        Gm[i,j] = getattr(p, f'G00')/N_pi
                        G[i,j]  = getattr(p, f'G00')/N_pi 
                        Ck[i,j] = getattr(p, f'C00')/N_pi
                        Cm[i,j] = getattr(p, f'C00')/N_pi
                        C[i,j]  = getattr(p, f'C00')/N_pi 
                        
                    else:
                        R[i,j]  = getattr(p, f'R01')/N_pi
                        L[i,j]  = getattr(p, f'L01')/N_pi
                        Gk[i,j] = getattr(p, f'G01')/N_pi
                        Gm[i,j] = getattr(p, f'G01')/N_pi
                        G[i,j]  = getattr(p, f'G01')/N_pi
                        Ck[i,j] = getattr(p, f'C01')/N_pi
                        Cm[i,j] = getattr(p, f'C01')/N_pi
                        C[i,j]  = getattr(p, f'C01')/N_pi

            # Constructing system matrix A
            # Vk
            A[0:3,0:3] = -inv(C) @ Gk 
            A[0:3,3:6] = -inv(C)
            A[0:3,6:9] = 0
            A[0:3,9:12] = 0

            # ikm
            A[3:6,0:3] = inv(L)
            A[3:6,3:6] = -inv(L) @ R 
            A[3:6,6:9] = -inv(L) 
            A[3:6,9:12] = 0

            # Vm
            A[6:9,0:3] = 0
            A[6:9,3:6] = inv(C)   #@ T.T
            A[6:9,6:9] = -inv(C) @ Gm
            A[6:9,9:12] = -inv(C)

            # Igrid
            A[9:12,0:3] = 0
            A[9:12,3:6] = 0   #@ T.T
            A[9:12,6:9] = inv(Lgrid) 
            A[9:12,9:12] = -inv(Lgrid) @ Rgrid

            
            # Construct input matrix B
            B[:3,:3] = inv(C) @ (Gk) # 
            B[3:6,:3] = -inv(Lgrid)
            B[9:12,3:6] = inv(Lgrid)

            # Reduce
            B[9:12,:3] = -inv(Lgrid)
            B[9:12,3:6] = inv(Lgrid)
            A = A[9:12,9:12]
            B = B[9:12,:]
            self.n = n = 3
            
            C = np.eye(n)
            D = np.zeros(1)

            
        else:
            
            if not self.MLE_silence: print(self.N_pi)
            N_pi = self.N_pi
            N_phi = 3
            
            self.n = n = N_phi* (1 + 2 * N_pi)
            # self.n = n = 9 + 2*3 

            # Prepare matrices
            A = np.zeros((n,n))
            B = np.zeros((n,3))              
            I = np.eye(3)

            # Impedance 
            Gin = I*1/p.Rin
            Gload = I*1/p.Rload
            Rin = I*p.Rin
            Rload = I*p.Rload
            R = np.zeros((3,3))
            L = np.zeros((3,3))
            Gk = np.zeros((3,3))
            Gm = np.zeros((3,3))
            Ck = np.zeros((3,3))
            Cm = np.zeros((3,3))
            C = np.zeros((3,3))
            G = np.zeros((3,3))
                                            
            for i in range(N_phi):
                for j in range(N_phi):
                    if i == j:
                        R[i,j]  = getattr(p, f'R00')/N_pi
                        L[i,j]  = getattr(p, f'L00')/N_pi
                        Gk[i,j] = getattr(p, f'G00')/N_pi + Gin[i,j]
                        Gm[i,j] = getattr(p, f'G00')/N_pi + Gload[i,j]
                        G[i,j]  = getattr(p, f'G00')/N_pi 
                        Ck[i,j] = getattr(p, f'C00')/N_pi
                        Cm[i,j] = getattr(p, f'C00')/N_pi
                        C[i,j]  = getattr(p, f'C00')/N_pi 
                        
                    else:
                        R[i,j]  = getattr(p, f'R01')/N_pi
                        L[i,j]  = getattr(p, f'L01')/N_pi
                        Gk[i,j] = getattr(p, f'G01')/N_pi
                        Gm[i,j] = getattr(p, f'G01')/N_pi
                        G[i,j]  = getattr(p, f'G01')/N_pi
                        Ck[i,j] = getattr(p, f'C01')/N_pi
                        Cm[i,j] = getattr(p, f'C01')/N_pi
                        C[i,j]  = getattr(p, f'C01')/N_pi

            if N_pi > 1:
                for i in range(0,2*N_pi + 1,2):
                    # Check if start
                    if i == 0:
                        start = True
                        c  = C 
                        gk = G + Gin
                    else: 
                        c  = C
                        gk = G 
                        start = False

                    # Check if end
                    if i == 2*N_pi:
                        end = True
                        c  = C
                        gm = G + Gload 
                    else:
                        c  = C
                        end = False
                    if not self.MLE_silence: print(n,i,2*N_pi,i*N_phi,start,end)

                    # Constructing system matrix A
                    # Vk
                    if not end:
                        A[i*N_phi+0*N_phi:i*N_phi+1*N_phi,i*N_phi+0*N_phi:i*N_phi+1*N_phi] = -inv(c) @ gk 
                        A[i*N_phi+0*N_phi:i*N_phi+1*N_phi,i*N_phi+1*N_phi:i*N_phi+2*N_phi] = -inv(c)
                        A[i*N_phi+0*N_phi:i*N_phi+1*N_phi,i*N_phi+2*N_phi:i*N_phi+3*N_phi] = 0
        
                        # ikm
                        
                        A[i*N_phi+1*N_phi:i*N_phi+2*N_phi,i*N_phi+0*N_phi:i*N_phi+1*N_phi] = inv(L)
                        A[i*N_phi+1*N_phi:i*N_phi+2*N_phi,i*N_phi+1*N_phi:i*N_phi+2*N_phi] = -inv(L) @ R 
                        A[i*N_phi+1*N_phi:i*N_phi+2*N_phi,i*N_phi+2*N_phi:i*N_phi+3*N_phi] = -inv(L) 
        
                        # Vm
                        A[i*N_phi+2*N_phi:i*N_phi+3*N_phi,i*N_phi+0*N_phi:i*N_phi+1*N_phi] = 0
                        A[i*N_phi+2*N_phi:i*N_phi+3*N_phi,i*N_phi+1*N_phi:i*N_phi+2*N_phi] = inv(c)   #@ T.T
                    else:
                        A[i*N_phi+0*N_phi:i*N_phi+1*N_phi,i*N_phi+0*N_phi:i*N_phi+1*N_phi] = -inv(c) @ gm
                        # A[i*N_phi+2*N_phi:i*N_phi+3*N_phi,i*N_phi+2*N_phi:i*N_phi+3*N_phi] = -inv(c) @ gm
                    
                # Construct input matrix B
                B[:N_phi,:N_phi] = inv(C) @ (Gk) # 
                B[N_phi:2*N_phi,:N_phi] = inv(L) @ R
 
                # C = np.zeros((9,n))                           
                # C[0:3,0:3] = np.eye(3)
                # C[3:,-6:] = np.eye(6)
                # C = np.zeros((9,n))                           
                C = np.eye(n)
                D = np.zeros(1)
                            
            else:
                # Constructing system matrix A
                # Vk
                A[0:3,0:3] = -inv(C) @ Gk 
                A[0:3,3:6] = -inv(C)# - inv(C) @ Gk @ R
                A[0:3,6:9] = 0

                # ikm
                A[3:6,0:3] = inv(L)# + inv(L) @ R @ Gk 
                A[3:6,3:6] = -inv(L) @ R 
                A[3:6,6:9] = -inv(L) #- inv(L) @ R @ Gm 

                # Vm
                A[6:9,0:3] = 0
                A[6:9,3:6] = inv(C) # + inv(C) @ Gk @ R #@ T.T
                A[6:9,6:9] = -inv(C) @ Gm
                
                # Construct input matrix B
                B[:3,:3] = inv(C) @ Gk # 
                B[3:6,:3] = inv(L) @ R

                C = np.eye(n)
                D = np.zeros(1)

            
            # Constructing system matrix A
            # A = np.zeros((n,n))
            # B = np.zeros((n,3))              

            # # Vk
            # A[0:3,0:3] = -inv(C) @ Gk
            # A[0:3,3:6] = -inv(C) @ (I + inv(R))
            # A[0:3,6:9] = 0

            # # ikm
            # A[3:6,0:3] = inv(L) @ (I + R)
            # A[3:6,3:6] = -inv(L) @ R 
            # A[3:6,6:9] = -inv(L) @ I

            # # Vm
            # A[6:9,0:3] = 0
            # A[6:9,3:6] = inv(C) @ (I) 
            # A[6:9,6:9] = -inv(C) @ Gm
            
            # # Construct input matrix B
            # B[:3,:3] = inv(C) @ (Gk) # 
            # # B[3:6,:3] = inv(L) @ R
                

        if verbose:
            print(A,B,C,D,sep='\n')
            
        return A, B, C, D

    
    def get_model(self,model,
                   discretize:bool=False,
                   dt:float=None,
                   params:dict=None,
                   pu:bool=True,
                   silence=False,
                   seq=False):
        """
        :param model:
        :param discretize:
        :param dt:
        :param params:
        :return:
        """
        # ------ Load parameters------
        self.pu = pu
        self.custom_params = params
        self.p = p = ParamWrapper(params,model,pu=pu,seq=seq)

        # ------ Load model ------
        self.model = model
        A, B, C, D = self.load_model(model,seq=seq)

        # ------ Discretization ------
        if discretize:
            if dt is None:
                raise ValueError('dt must be defined to discretize the system')
            else:                
                # A, B, C, D, dt = scipy.signal.cont2discrete((A, B, C, D),dt)
                A_d,B_d,C,D = self.c2d(A,B,C,D,dt)
            
        # ------ Data storage ------
        # Store state and relevant data
        self.A, self.B, self.C, self.D = A, B, C, D
        self.discrete = discretize
        self.dt = dt
        if discretize:            
            self.condition_number = np.linalg.cond(A_d)
            self.lambd = np.linalg.eigvals(A_d)
            self.A_unstable = np.real(self.lambd).any() > 1
        else:
            self.condition_number = np.linalg.cond(A)
            self.lambd = np.linalg.eigvals(A)
            self.A_unstable = np.real(self.lambd).any() > 0

        # ------ Print statements ------
        # print model configuration
        if not silence:
            if discretize:
                if not self.MLE_silence: print('',f'Model: {model}','A=',A_d,'B=',B_d,'C=',C,'D=',D,'Lambdas=',*list(self.lambd),'',f'Condition number:\t{self.condition_number}',f'A matrix unstable:\t{self.A_unstable}',sep='\n') 
            else:
                if not self.MLE_silence: print('',f'Model: {model}','A=',A,'B=',B,'C=',C,'D=',D,'Lambdas=',*list(self.lambd),'',f'Condition number:\t{self.condition_number}',f'A matrix unstable:\t{self.A_unstable}',sep='\n') 
        
        return
    
    
    def check_observability(self,A,C):

        # # Filter C for zero-rows
        # # Find rows that are not all zeros
        # non_zero_rows = ~np.all(C == 0, axis=1)
        # # Select only those rows
        # C_ = C[non_zero_rows]        
        # n = A.shape[0] # number of states
        # m = (C_ @ A).shape[0]
    
        # O = np.zeros((m*n,n))
        # for i in range(n):
        #     O[i*m:i*m+m,:] = C_ @ (A)**i
        
        # O = obsv(A, C)

        O = np.vstack([np.dot(np.linalg.matrix_power(A, i), C.T) for i in range(A.shape[0])])

        # Compute its rank
        rank = np.linalg.matrix_rank(O)
        
        # Compute the eigenvalues of the system
        eigvals, _ = np.linalg.eig(A)
        
        # Find the unobservable eigenvalues
        unobservable_eigvals = eigvals[rank:]
                        
        print("\nRank of the Observability Matrix:", rank)
        # check if the system is observable
        if rank == A.shape[0]:
            print("\nThe system is observable.")
            observable = True
        else:
            print("\nThe system is not observable.")   
            observable = False
            
        # Check if all unobservable eigenvalues have negative real parts
        if np.all(np.real(unobservable_eigvals) < 0):
            print("The system is detectable.")
            detectable = True
        else:
            print("The system is not detectable.")
            detectable = False

        return observable, detectable
        
    def create_input(self, t1, t2, dt, amp=None, phi=None, t0:float = 0, mode:str='sin'):

        # if Sx is None: Sx = lambda: sx_random*((sx,1)[sx is None])*np.random.randn(m) + mu_x
        # if Sy is None: Sy = lambda: sy_random*((sy,1)[sy is None])*np.random.randn(m) + mu_y        

        if mode not in ['sin','cos','-sin','-cos','step','impulse','abc']:
            raise ValueError("Mode must be 'sin','cos','step','abc', or 'impulse'")

        
        # Parameter selection
        if phi is None: phi = self.p.phi
        if amp is None: amp = self.p.V
        
        # Function evaluation
        if mode == 'sin':
            u = lambda t: amp*np.sin(self.p.omega*t+phi)*(0,1)[t>=t0]
        elif mode == 'cos':
            u = lambda t: amp*np.cos(self.p.omega*t+phi)*(0,1)[t>=t0]
        elif mode == '-cos':
            u = lambda t: -amp*np.cos(self.p.omega*t+phi)*(0,1)[t>=t0]
        if mode == '-sin':
            u = lambda t: -amp*np.sin(self.p.omega*t+phi)*(0,1)[t>=t0]
        elif mode == 'step':
            u = lambda t: amp*(0,1)[t>=t0]
        elif mode == 'impulse':
            u = lambda t: amp*(0,1)[t0<= t and t < t0 + dt]
        elif mode == 'abc':
            if len(amp) != 3 or len(phi) != 3:
                raise ValueError("Please provide 3 amplitudes and phases as input")
            u1 = lambda t: amp[0]*(np.sin(self.p.omega*t+phi[0])-1j*np.cos(self.p.omega*t+phi[0]))*(0,1)[t>=t0]
            u2 = lambda t: amp[1]*(np.sin(self.p.omega*t+phi[1])-1j*np.cos(self.p.omega*t+phi[1]))*(0,1)[t>=t0]
            u3 = lambda t: amp[2]*(np.sin(self.p.omega*t+phi[2])-1j*np.cos(self.p.omega*t+phi[2]))*(0,1)[t>=t0]
            u = (u1,u2,u3)    
            
        
        # Evaluate u(k) for each k
        time = np.arange(t1,t2+dt,dt)
        if mode == 'abc':  
            uk = np.array([
                [u1(k) for k in time],
                [u2(k) for k in time],
                [u3(k) for k in time],
                ])            
        else:
            uk = np.array([u(k) for k in time])

        return u, uk

    def create_noise(self, t1, t2, dt, amp=1,dim = 1,mu=0,seed=1234):
        time = np.arange(t1,t2+dt,dt)
        np.random.seed(seed)
        Nk = amp*np.random.randn(dim,len(time)) + mu
        return Nk
    
    def abc2seq(self,abc):
        # Number of phases
        num_phases = 3
    
        # Construct the transformation matrix
        A = np.array([[1, 1, 1],
                                          [1, np.exp(-1j * 2 * np.pi / num_phases), np.exp(-1j * 4 * np.pi / num_phases)],
                                          [1, np.exp(-1j * 4 * np.pi / num_phases), np.exp(-1j * 8 * np.pi / num_phases)]])
    
        # Perform the sequence transformation
        seq = np.zeros_like(abc,dtype=complex)
        for i in range(abc.shape[1]):
            seq[:,i] = inv(A) @ abc[:,i]
    
    
    
        return seq

    
    #============================= Evaluate functions =============================#
    def evaluate_condition_number(self,params,n=100):
        data = {'Rin':[],
                'Rload':[],
                'condition':[],
                }

        for rin in np.linspace(0.05,100,n):
            if not self.MLE_silence: print('asdf')
            for rload in np.linspace(0.2,1e6,n):
                print(rin,rload)
                params['Rin'] = rin
                params['Rload'] = rload
                
                # ------ Load parameters------
                p = ParamWrapper(params,pu=True)
        
                # ------ Load model ------
                A, B, C, D = self.load_model(p=p)
            
                data['Rin'].append(rin)                
                data['Rload'].append(rload)                
                data['condition'].append(np.linalg.cond(A))                                
                
        return pd.DataFrame(data,columns=list(data.keys()))
    
    #============================= STAT METHODS =============================#
    def quantile_confidence_interval(self, data, alpha):
        # Calculate the sample mean and standard deviation
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
    
        # Calculate the critical value for the desired level of confidence and degree of freedom
        n_samples = len(data)
        df = n_samples - 1
        if n_samples <= 30:
            critical_value = stats.t.ppf((1-alpha)/2, df)
        else:
            critical_value = stats.norm.ppf((1-alpha)/2)
    
        # Calculate the standard error of the sample mean
        standard_error = sample_std / np.sqrt(n_samples)
    
        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = sample_mean - (critical_value * standard_error)
        upper_bound = sample_mean + (critical_value * standard_error)
    
        # Return the lower and upper bounds as a tuple
        return (lower_bound, upper_bound)

    
    
    #============================= PLOT METHODS =============================#

    def plot_simulations(self,t,sims,labels:list=None,colors=clrs,figsize=(9,6),grid:bool=True,save=None,file_extension='pdf',path=None):
        fig, ax = plt.subplots(sims[0].shape[0],1,figsize=figsize,dpi=200,sharex=True)


        # input validation
        if labels is not None and len(sims) != len(labels):
            raise ValueError('Number of simulations and labels are not having same length')
        
        # Define linestyles
        lss=['-','--','-.',':']        
        
        for k, sim in enumerate(sims):
            for i in range(sim.shape[0]):
                idxs = ~np.isnan(sim[i]) & ~(abs(sim[i]) > 2)
                idxs = ~np.isnan(sim[i]) & ~(abs(sim[i]) > 2e6)
                if labels is not None:
                    ax[i].plot(t[idxs],sim[i,idxs],color=colors[k],ls=lss[k%4],label=labels[k])
                else:
                    ax[i].plot(t[idxs],sim[i,idxs],color=colors[k],ls=lss[k%4])
        
        # Add grid lines
        if grid: [ax[i].grid() for i in range(sims[0].shape[0])]
                

        if labels is not None:
            ax[1].legend(ncols=2)
            
        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_sim.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_sim.{file_extension}',dpi=200)
            plt.close()
        return

    def plot_estimation_summary(self,thetahat,thetahat0,save=None,file_extension='pdf',path=None):
        fig, ax = plt.subplots(1,4,dpi=300,figsize=(9,3))
        error = abs(thetahat-self.A)
        titles = ['System\n$\\theta$','Initial guess\n$\\theta_0$','Estimation\n$\\hat{\\theta}$','Error\nnorm$(|\\hat{\\theta}-\\theta|)$'] # ='+str(round(np.linalg.norm(error),2))+'
        for i, data in enumerate([self.A,thetahat0,thetahat,error]):
            # if 'Error' in titles[i]:
            #     im = ax[i].imshow(data, norm=LogNorm(),cmap=cm.Reds)
            #     im = ax[i].imshow(data, norm=LogNorm(),cmap=cm.Reds)

            # else:
            #     im = ax[i].imshow(data,cmap=cm.Reds)
                
            im = ax[i].imshow(data,cmap=cm.Reds)
            ax[i].set(title=titles[i])            

            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)            
            cbar = fig.colorbar(im, cax=cax)
            
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_summ.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_summ.{file_extension}',dpi=200)
            plt.close()

        return
    
    def eigenvalues_analysis(self,M,P_llim=0.01,P_ulim=1e6,vmax=1):
        
        lamb, R = eig(M)
        
        L = inv(R)
        
        P = R @ L.T        
        
        fig, ax = plt.subplots(1,1,dpi=150)
        cax = ax.imshow(np.where((abs(P)>=P_llim) & (abs(P)<=P_ulim),abs(P),np.nan), 
                        # cmap=plt.cm.coolwarm, 
                        vmin=0, vmax=vmax)
        cbar = fig.colorbar(cax, extend='max')
        
        return P
    
    
    def plot_eigenvalues(self,systems,labels=None,save=None,file_extension='pdf',path=None):
        # 
        z_xy = lambda z: ([real(x) for x in z],[imag(x) for x in z])

    
        # input validation
        if labels is not None and len(systems) != len(labels):
            raise ValueError('systems and labels are not having same length')
        
        # Custom markers
        markers = ['o','x','*','.']
        
        # 
        fig, ax = plt.subplots(1,1,dpi=300)
        # Go through all the systems
        for i, sys in enumerate(systems):
            # Get the eigenvalues
            lambd = eigvals(sys)

            x,y = z_xy(lambd)
            if not self.MLE_silence: print(x,y)
            if labels is not None:
                ax.scatter(x,y,label=labels[i],zorder=1e3-i,alpha=0.75,marker=markers[i])
            else:
                ax.scatter(x,y,zorder=1e3-i,alpha=0.75,marker=markers[i%4])         
                    
        ax.axvline(0,color='k',lw=0.75)
        ax.axvline(1,color='k',lw=0.75)
        ax.axhline(0,color='k',lw=0.75)
        ax.grid()
        ax.legend(loc='upper left')

        # plt.gca().set_aspect('equal', adjustable='box')
        ax.set(xlabel='Real Part',ylabel='Imaginary Part')

        if save is not None:
            if path is None:
                plt.savefig(f'img/{save}_eig.{file_extension}',dpi=200)
            else:
                plt.savefig(f'{path}\\{save}_eig.{file_extension}',dpi=200)
            plt.close()

        
        return

    
    def plot_covariance_in_time(self,covs):
        if isinstance(covs,np.ndarray):
            covs = [covs]
        
        fig, ax = plt.subplots(len(covs),1,dpi=200)
        if len(covs) > 1:
            for i, cov in enumerate(covs):
                print(cov.shape)
                x,y = cov.shape[0]*cov.shape[0], cov.shape[2]
                c = abs(cov.reshape((x,y)))
                im = ax[i].imshow(c,norm=LogNorm())
                ax[i].set_xticks([i for i in range(x)])
                ax[i].set_xticklabels([f'{i%cov.shape[0]}{i//cov.shape[0]}' for i in range(x)])        
        
                # Colorbar
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
        else:
            cov = covs[0]
            x,y = cov.shape[0]*cov.shape[0], cov.shape[2]
            c = abs(cov.reshape((x,y)))
            im = ax.imshow(c,norm=LogNorm())
            ax.set_yticks([i for i in range(x)])
            ax.set_yticklabels([f'{i%cov.shape[0]},{i//cov.shape[0]}' for i in range(x)])        
    
            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax)

            fig, ax = plt.subplots(1,1,dpi=200)
            plt.imshow(abs(cov[:,:,-1]),norm=LogNorm())
            


        return    
    
    #============================= PLOT METHODS =============================#
    def simulate(self,A,B,C,D,x0,u,t1,t2,dt,Sx=None,Sy=None):
        time = np.arange(t1,t2+dt,dt)

        # creating indices
        n_y = len(time)
        n = A.shape[0]
        m = C.shape[0]

        # Initialize arrays
        x   = np.zeros((n,n_y))
        y   = np.zeros((m,n_y))

        # Handling noise input
        if Sx is None: Sx = np.zeros((n,len(time)))
        if Sy is None: Sy = np.zeros((m,len(time)))

        print('',f'Simulating discrete-time system:',sep='\n')
        t1_ = datetime.datetime.now()

        if not self.MLE_silence: print(y.shape,x0.shape,C.shape,u.shape,Sx.shape,Sy.shape)

        # initial values
        x[:,0] = x0
        y[:,0] = C @ x0 # + Sy() # TODO: Add noise, be aware of operator for u[k]
        for k,t in enumerate(time[:-1]):
            # --------------- Solving x[k+1] ---------------
            # Calculate x[:, k + 1]
            if len(u.shape) == 1:
                x[:,k+1] = A @ x[:,k] + B @ np.array([u[k]]) + Sx[:,k] #+ Sx() # TODO: Add noise, be aware of operator for u[k]
            else:
                x[:,k+1] = A @ x[:,k] + B @ u[:,k] + Sx[:,k] #+ Sx() # TODO: Add noise, be aware of operator for u[k]
            y[:,k] = C @ x[:,k] + Sy[:,k] # + Sy() # TODO: Add noise, be aware of operator for u[k]
        y[:,-1] = C @ x[:,-1] + Sy[:,-1] # TODO: Add noise, be aware of operator for u[k]
        t2_ = datetime.datetime.now()            
        print(f'Finished in: {(t2_-t1_).total_seconds()} s')

        return x,y
    

    def LS(self,x,y,dt,u=None):
        # Input validation        
        if x.shape != y.shape:
            raise ValueError(f'x and y do not have same dimensions: {x.shape} != {y.shape}')

        # If u are provided
        if u is not None:
            # Initialize
            n,m = x.shape
            n += (u.shape[0],1)[len(u.shape)==1]
            thetahat = np.zeros((n,n,m))
            x = np.vstack([x,u])
            y = np.vstack([y,np.zeros_like(u)])
            
            for k in range(m):
                # thetahat[:,:,k] = inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2) @ x[:,k].reshape((n,1)) @ y[:,k].reshape((1,n))  
                thetahat[:,:,k] =  y[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) @ inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2)
            
        else:
            # Initialize
            n,m = x.shape
            thetahat = np.zeros((n,n,m))
            for k in range(m):
                thetahat[:,:,k] = inv(x[:,k].reshape((n,1)) @ x[:,k].reshape((1,n)) - np.eye(n)*dt**2) @ x[:,k].reshape((n,1)) @ y[:,k].reshape((1,n))  
        
        
        return thetahat
    

    def KalmanFilter(self,A,B,C,D,x0,u,y,R0,R1,R2,t1,t2,dt,optimization_routine:bool=False):
        if optimization_routine and len(A.shape) == 1:
            dim = int(np.sqrt(len(A)))
            A = A.reshape((dim,dim))
            
        time = np.arange(t1,t2+dt,dt)

        if len(u.shape) == 1:
            u = u.reshape(1,-1)

        # creating indices
        n_y = len(time)
        n = A.shape[0]
        m = C.shape[0]

        if not self.MLE_silence: print(n,m)

        # Initialize arrays
        eps = np.zeros((m,n_y))
        x_hat_filt   = np.zeros((n,n_y))
        x_hat_pred   = np.zeros((n,n_y))
        y_hat_pred   = np.zeros((m,n_y))
        P_filt       = np.zeros((n,n,n_y))
        P_pred       = np.zeros((n,n,n_y))
        K            = np.zeros((n,n,n_y))
        R            = np.zeros((n,n,n_y))
        
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
            K[:,:,k] = P_pred[:,:,k] @ C.T @ inv(R[:,:,k])
    
            # Measurement update
            x_hat_filt[:,k] = x_hat_pred[:,k] + K[:,:,k] @ eps[:,k]
            P_filt[:,:,k] = (np.eye(n) - K[:,:,k] @ C) @ P_pred[:,:,k]

            # Time update
            if k < len(time)-1:
                x_hat_pred[:,k+1] = A @ x_hat_filt[:,k] + B @ u[:,k]
                P_pred[:, :, k+1] = A @ P_filt[:,:,k] @ A.T + R1 # TODO: R1 = B @ R1 @ B.T, B is not the input matrix in this context.
        
        return x_hat_pred, y_hat_pred, eps, R

    #============================= ESTIMATION METHODS =============================#    

    def param_estimation(self):
        
        return

    def KF_estimation(self,A,B,C,D,x0,u,y,R0,R1,R2,t1,t2,dt,optimization_routine:bool=False):

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

    
    def dt_to_ct_zoh(self, H_z, Ts, T):
        """
        Transforms a discrete-time system into a continuous-time system using the Zero-Order Hold (ZOH) method.
        
        Parameters
        ----------
        H_z : array_like
            The transfer function of the discrete-time system in z-domain.
        Ts : float
            The sampling period of the discrete-time system.
        T : float
            The duration of the rectangular pulse used in the ZOH method.
            
        Returns
        -------
        H_s : array_like
            The transfer function of the continuous-time system in s-domain.
        """
        w, H_ejw = signal.freqz(H_z)
        H_r = np.sinc(w / np.pi / Ts)
        H_cjw = H_ejw * H_r
        t, h_ct = signal.freqz(H_cjw, [1, 0], worN=2**14)
        h_ct = np.real(h_ct)
        H_s = signal.TransferFunction(h_ct, [1, 0])
        return H_s

    def c2d(self,A,B,C,D,dt,mode='ZOH'):

        if len(A.shape) == 1:
            n = int(np.sqrt(A.shape[0]))
            A = A.reshape((n,n))

        # Calculate the continuous-time state transition matrix using the matrix exponential
        # At = np.block([[A, B], [np.zeros((1, A.shape[1])), np.zeros((1, 1))]])
        At = np.block([[A, B], [np.zeros((B.shape[1], A.shape[1])), np.zeros((B.shape[1], B.shape[1]))]])
        eAt = linalg.expm(At * dt)
        Ad = eAt[:A.shape[0], :A.shape[1]]
        Bd = eAt[:A.shape[0], -B.shape[1]:]          



        # assign to class
        self.A_d, self.B_d = Ad, Bd
        
        return Ad, Bd, C, D

    def d2c(self, A_d, B_d, C_d, D_d, Ts, T):
        
        n = A_d.shape[0]
        # Ad = np.block([[A_d, B_d], [np.zeros((1, n)), 1]])
        # Bd = np.block([B_d, 0]).reshape(n + 1, 1)
        # Cd = np.block([C_d, D_d]).reshape(1, n + 1)
        
        H_z = signal.ss2tf(A_d, B_d, C_d, D_d)[0]
        H_s = self.dt_to_ct_zoh(H_z, Ts, T)
        A_c, B_c, C_c, D_c = signal.tf2ss(H_s.num, H_s.den)
        
        return A_c[:, :n], B_c[:n], C_c[:, :n], D_c
    #============================= MODEL CHECKING METHODS =============================#
    
    def runs_test(residuals, alpha=0.05):
        """
        Performs a runs test for a change in signs in the residuals of a linear regression model.
    
        Parameters:
            residuals (array-like): An array of residuals from a linear regression model.
            alpha (float): The desired significance level (default is 0.05).
    
        Returns:
            result (string): A string indicating whether there is evidence of a change in signs in the residuals.
            p_value (float): The p-value associated with the test statistic.
            test_statistic (float): The value of the test statistic.
        """
        n = len(residuals)
        runs = 1
        for i in range(1, n):
            if np.sign(residuals[i]) != np.sign(residuals[i-1]):
                runs += 1
        expected_runs = (2*n - 1) / 3
        variance = (16*n - 29) / 90
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        if p_value < alpha:
            result = "Evidence of a change in signs in the residuals."
        else:
            result = "No evidence of a change in signs in the residuals."
        return result, p_value, z

    def residual_analysis(self,eps):
        single = len(eps.shape)==1
        n = (eps.shape[0],1)[len(eps.shape)==1]
        fig, ax = plt.subplots(n,1,sharex=True,dpi=200)

        for i in range(n):
            if single:
                ax.plot(eps)
            else:
                ax[i].plot(eps[i,:])
                
        # TODO: Calculate sign changes
            
        return
    #============================= VERIFICATION METHODS =============================#
    def FastFiniteDifferenceDerivatives(fun, x, epsilon, *funargs, order=2):
        # TODO: Implement list of epsilons for parameter sensitivity selectibility
        
        # Evaluate function
        f = fun(x, *funargs)
        
        # Dimensions
        nx = x.size
        nf = f.size
        
        ## First order derivatives
        dfFD = np.zeros((nf, nx))
        for j in range(nx):
            # Pertubation
            x[j] = x[j] + epsilon
            
            # Perturbed function evaluation
            fp = fun(x, *funargs)
            
            # Approximation
            dfFD[:, j] = (fp.ravel() - f.ravel()) / epsilon
            
            # Remove pertubation
            x[j] = x[j] - epsilon
        
        ## Second order derivatives
        if order == 2:
            epssq = epsilon**2
            d2fFD = np.zeros((nx, nx, nf))
            for j in range(nx):
                # Pertubation
                x[j] = x[j] + 2*epsilon
                    
                # Perturbed function evaluation
                fpp = fun(x, *funargs)
                    
                # Pertubation
                x[j] = x[j] - epsilon
        
                # Perturbed function evaluation
                fpz = fun(x, *funargs)
                
                # Approximation
                d2fFD[j, j, :] = (fpp.ravel() - 2*fpz.ravel() + f.ravel()) / epssq
                
                # Reset pertubation
                x[j] = x[j] - epsilon
                
                for k in range(j):
                    # Pertubation
                    x[j] = x[j] + epsilon
                    x[k] = x[k] + epsilon
                    
                    # Perturbed function evaluation
                    fpp = fun(x, *funargs)
                    
                    # Reset pertubation
                    x[k] = x[k] - epsilon
                     
                    # Perturbed function evaluation
                    fpz = fun(x, *funargs)
                    
                    # Pertubation
                    x[k] = x[k] + epsilon
                    x[j] = x[j] - epsilon
                     
                    # Perturbed function evaluation
                    fzp = fun(x, *funargs)
                    
                    # Approximation
                    d2fFD[k, j, :] = d2fFD[j, k, :] = (fpp.ravel() - fpz.ravel() - fzp.ravel() + f.ravel()) / epssq
                    
                    # Reset pertubation
                    x[k] = x[k] - epsilon
        if order == 2:
            return dfFD, d2fFD
        else:
            return dfFD


    #============================= ESTIMATION METHODS =============================#    
    def mmse_estimate(self, A, missing_index, R2, R):
        """
        MMSE = Minimum Mean Squared Error
        Computes the MMSE estimate of a missing element in a matrix A, given its covariance matrix under a Gaussian prior distribution and a Gaussian observation model.
    
        Parameters:
        A (numpy array): The observed matrix with the missing element set to NaN.
        missing_index (tuple): The row and column indices of the missing element in A.
        R2 (numpy array): measurement covariance, The covariance matrix of the observation noise.
        R (numpy array): prediction covariance, The covariance matrix of the prior distribution.
    
        Returns:
        mmse_estimate (float): The MMSE estimate of the missing element in A.
        """
        # Remove the missing element from the matrix A
        obs_indices = np.ones(A.shape, dtype=bool)
        obs_indices[missing_index[0], missing_index[1]] = False
        A_obs = A[obs_indices].reshape(-1, 1)
    
        # Compute the conditional mean and covariance of the missing element given the observed data
        cov_cond = inv(inv(R) + obs_indices.sum() * inv(R2))
        mean_cond = cov_cond @ (obs_indices.sum() * inv(R2) @ A_obs + inv(R) @ A[missing_index])
    
        # Return the MMSE estimate, which is the conditional mean of the missing element
        return mean_cond, cov_cond
    
    def kernel_density_estimate(self,matrix):
        print('',f'Starting Kernel Density Estimation (KDE)',sep='\n')
        t1_ = datetime.datetime.now()

        n = matrix.shape[0]
        KDE = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                
                w, v = np.histogram(matrix[i,j,:])

                # Get index of most likely        
                idx = list(w).index(max(w))
                
                KDE[i,j] = v[idx:idx+2].mean()

        t2_ = datetime.datetime.now()
        print(f'Finished in: {(t2_-t1_).total_seconds()} s')

        return KDE
    
    def ML_Wrapper(self,theta,opt_params,A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt,log):

        # Perturb parameters
        p = copy.deepcopy(self.p)
        for i, param_name in enumerate(opt_params):           
            setattr(p,param_name,np.exp(theta[i]))

        # Get continous system
        A, B, C, D = self.load_model(p = p)
        
        # Discretize continous system
        Ad, Bd, C, D = self.c2d(A,B,C,D,dt)

            
        # Evaluating cost function
        J = self.ML(Ad,Bd,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt)

        # Print verbose
        if self.opts.verbose: 
            for i, param_name in enumerate(opt_params):
                print(i,opt_params[i],theta[i],np.exp(theta[i]))
            print(theta)
    
        return J


    def MC_param(self,opt_params,n_iters, A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt):

        data = {'J':[]}
        for i, param_name in enumerate(opt_params):   
            data[param_name] = []            
    
        for it in tqdm(range(int(n_iters))):
            # Perturb parameters
            p = copy.deepcopy(self.p)
            for i, param_name in enumerate(opt_params):   
                val = getattr(p,param_name)
                delta = val*0.5
                rand_val = np.random.uniform(low=val - delta, high=val + delta, size=None)
                setattr(p,param_name,rand_val)
                data[param_name].append(rand_val)
    
            # Get continous system
            A, B, C, D = self.load_model(p = p)
           
            # Discretize continous system
            Ad, Bd, C, D = self.c2d(A,B,C,D,dt)
    
                
            # Evaluating cost function
            J = self.ML(Ad,Bd,C,D,x0, u, y, R0, R1, R2, t1, t2, dt)
            data['J'].append(J)
                
        return pd.DataFrame(data)

    def ML_sensitivity(self,opt_params,n, A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,delta=0.5,silence=True):
        self.opt_params = opt_params
        self.MLE_silence = silence
        from itertools import product
        def cartesian_product(*sequences,column_names=None):
            # Generate the Cartesian product of the sequences
            cart_product = list(product(*sequences))
        
            # Create a DataFrame from the Cartesian product
            df = pd.DataFrame(cart_product)
        
            # Create column names
            if column_names is None:
                column_names = [f'Sequence_{i+1}' for i in range(len(sequences))]
            df.columns = column_names
        
            return df

        data = {'J':[]}
        iter_list = []
        for i, param_name in enumerate(opt_params):   
            data[param_name] = []
            iter_list.append(np.linspace(delta*getattr(self.p,param_name),(1+delta)*getattr(self.p,param_name),n + (0,1)[n%2 == 0]))

        # 
        df = cartesian_product(*iter_list,column_names=opt_params)
    
        for i, row in tqdm(df.iterrows(),total=df.shape[0]):
            # print(row)
            # Perturb parameters
            p = copy.deepcopy(self.p)
            for i, param_name in enumerate(opt_params):   
                val = getattr(p,param_name)
                setattr(p,param_name,row[param_name])
                data[param_name].append(row[param_name])

            # Get continous system
            A, B, C, D = self.load_model(p = p)
           
            # Discretize continous system
            Ad, Bd, C, D = self.c2d(A,B,C,D,dt)
                    
            # Evaluating cost function
            J = self.ML(Ad,Bd,C,D,x0, u, y, R0, R1, R2, t1, t2, dt)
            data['J'].append(J)

                
        return pd.DataFrame(data)

    
    def ML(self,Ad,Bd,C,D,x0, u, y, R0, R1, R2, t1, t2, dt):        

        # Get covariance and prediction error
        _,_, eps,R = self.KalmanFilter(Ad,Bd,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,optimization_routine=True)

        # Evaluate the cost function from the batch sum
        J = 0
        for k in range(eps.shape[1]):
            J += 1/2*(np.log(det(R[:,:,k])) \
                      + np.log(2*np.pi)\
                      + eps[:,k].T @ inv(R[:,:,k]) @ eps[:,k]) 
        
        # print(J)
        
        return J
        
    def ML_opt(self,opt_params,A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt, thetahat0,log:bool = True):
        # Defining constraints (only reasonable for not optimization without log)
        N = len(opt_params)
        self.vals = vals = np.array([self.p.params[k] for k in opt_params])
        # if log:
        #     constraints = (LinearConstraint(np.eye(N), lb=np.log(vals*self.opts.cnstr_lb_factor), ub=np.log(vals*self.opts.cnstr_ub_factor), keep_feasible=False))
        # else:
        #     constraints = (LinearConstraint(np.eye(N), lb=vals*self.opts.cnstr_lb_factor, ub=vals*self.opts.cnstr_ub_factor, keep_feasible=False))

        if not self.MLE_silence: print(thetahat0)


        # Minimization
        thetahat = minimize(self.ML_Wrapper,
                            args=(opt_params,A,B,C,D,x0, u, y, R0, R1, R2, t_start, t_end, dt, log),
                            x0=thetahat0,
                            method=self.opts.method,
                            # jac = '3-point',
                            jac = self.opts.jac,
                            # constraints=(None,constraints)[self.opts.method == 'SLSQP'],
                            hess = self.opts.hess,
                            hessp = self.opts.hessp,
                            # options={'disp':True,'gtol': 1e-12,'return_all':True,'bounds':(-0.5,0.5),'eps':1e-4},
                            options={'disp':self.opts.disp,
                                     # 'eps':self.opts.epsilon,
                                     # 'finite_diff_rel_step':self.opts.epsilon,
                                     },
                            tol=self.opts.err_tol
                            )
        
        if log:
            thetahat.x = np.exp(thetahat.x)
        # print(thetahat,'\n')

        return thetahat, thetahat0

    def ML_opt_param(self,opt_params,A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt, thetahat0=None, log:bool=True,silence=False):
        self.MLE_silence = silence
        self.opt_params = opt_params
        vals = np.array([self.p.params[k]*(self.p.Zbase,1)[k in ['SCR','XR']] for k in opt_params])

        # ------ Initializing ------
        # Selecting initial value
        if thetahat0 is None:
            thetahat0 = [self.p.params[k] for k in opt_params]       
        ests = np.zeros(len(opt_params))
        devs = np.zeros(len(opt_params))
        thetahat0s = np.zeros(len(opt_params))

        # ------ Optimize ------
        if not self.MLE_silence: print('',f'Starting maximum likelihood estimation of all parameters:',sep='\n')
        t1_ = datetime.datetime.now()
        thetahat, thetahat0_ = self.ML_opt(opt_params,A,B,C,D,x0, u, y, R0, R1, R2, t1, t2, dt,thetahat0,log=log)
        t2_ = datetime.datetime.now()            

        # ------ Evaluate ------
        if self.opts.disp: print('\n',thetahat)

        ests = thetahat.x*np.array([(self.p.Zbase,1)[k in ['SCR','XR']] for k in opt_params])
        devs = ests - vals
        thetahat0s = thetahat0_

        # return resulting A matrix
        p_hat = {}
        for i, p in enumerate(opt_params):
            p_hat[p] = np.exp(ests[i])

        p_hat = ParamWrapper(p_hat,self.model)

        A_hat, _,_,_ = self.load_model(model=self.model,p=p_hat)
        
        opt_data = {'System':vals,
                            # 'Lower bound':vals*self.opts.cnstr_lb_factor,
                            'Estimated':ests,
                            # 'Upper bound':vals*self.opts.cnstr_ub_factor,
                            'Init guess':thetahat0,
                            'Deviations':devs}
    
        res = pd.DataFrame(opt_data,index=opt_params)

        # ------ Print statements ------
        if not self.MLE_silence: 
            print(f'Finished in: {(t2_-t1_).total_seconds()} s')
            print(f'#========= FINAL ESTIMATION SUMMARY =========#')
            print(res)
            print('2-norm:\n',np.linalg.norm(devs))
            try:
                print('Eigenvalues:\n',eigvals(A_hat))
            except Exception as e:
                pass            
            print('')
            
        return ests, thetahat, res, A_hat


