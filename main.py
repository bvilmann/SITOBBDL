#%% ================ Initialize packages and general settings ================ 
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os



#%% DATA READER

rpath = f'{os.getcwd()}\\data'
fname = r'cable_1phase.infx'
file_addr = f'{rpath}\\{fname}'
dread = DataReader()
_,iabc1 = dread.get_signal('Iabc_b1',file_addr)
_,iabc2 = dread.get_signal('Iabc_b2',file_addr)

_,vabc1 = dread.get_signal('Vabc_b1',file_addr)
t_pscad,vabc2 = dread.get_signal('Vabc_b2',file_addr)


#%%

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
Nx = np.array([1e-7,1e-4,1e-4])*0
r123 = np.array([1,1,1])*1e-4

R0=np.diag(r123)*1e5
R1=np.diag(r123)*1e3
R2=np.diag(r123)*1


#%% EVAULATE CONDITION NUMBER
# m = SITOBBDS()
# n = 10
# data = {'Rin':[],
#         'Rload':[],
#         'L':[],
#         'R':[],
#         'condition':[],
#         }
# for rin in np.linspace(0.05,100,n):
#     for rload in np.linspace(0.2,1e5,n):
#         for l in np.linspace(1e-4,1e-2,n):
#             for r in np.linspace(0,100,n):
#                 params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':np.pi/4}
#                 m = SITOBBDS()
#                 m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True,silence=True)
            
#                 data['Rin'].append(rin)                
#                 data['Rload'].append(rload)                
#                 data['L'].append(l)                
#                 data['R'].append(r)                
#                 data['condition'].append(np.linalg.cond(m.A))                                

# df = pd.DataFrame(data,columns=list(data.keys()))

# table = pd.pivot(df.groupby(['Rin','Rload','condition']).mean(),columns='Rin',index='Rload',values='condition')

# fig, ax=plt.subplots(1,1,dpi=300)
# im = ax.imshow(table,norm=LogNorm(),extent=(df.Rin.min(), df.Rin.max(), df.Rload.max(), df.Rload.min()),aspect='auto') # (left, right, bottom, top) 
# ax.set(ylabel='Rload',xlabel='Rin')
# cbar = fig.colorbar(im)
# plt.show()

#%%
# fig, ax = plt.subplots(4,4,figsize=(12,12),dpi=200)

# for i,r in enumerate(list(set(df.columns)-set(['condition']))):
#     for j,c in enumerate(list(set(df.columns)-set(['condition']))):
#         if r == c:
#             continue
            
#         df_temp = df[[r,c,'condition']].drop_duplicates().reset_index()[[r,c,'condition']]        
#         table = pd.pivot(df_temp,columns=r,index=c,values='condition')
#         ax[i,j].imshow(table,norm=LogNorm(),extent=(df[r].min(), df[r].max(), df[c].max(), df[c].min()),aspect='auto') # (left, right, bottom, top))        
#         ax[i,0].set(ylabel=r)
#         ax[-1,j].set(xlabel=c)

opts = {'gradient':'2-point','method':'L-BFGS-B'}
m = SITOBBDS(opts=opts)
params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
# m.get_model('Cable_2',discretize=True,dt=10e-6,params=params,pu=True)
# m.get_model('dummy1',discretize=True,dt=10e-6,params=params,pu=True)


# Create input
u, uk = m.create_input(t1, t2, dt,mode='sin')        

# Get matrices
Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D

# --------------- GET GROUND TRUTH --------------- 5
# Simulate the system
x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)

# Filter the data with the Kalman Filter
xhat, yhat, eps, R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)

# --------------- ESTIMATE --------------- 
# LS - LEAST SQUARES
thetahat = m.LS(xhat,yhat,dt,u=uk)
# thetahat = m.LS(xhat,yhat,dt)

#%%
# KDE - KERNEL DENSITY ESTIMATION
KDE = m.kernel_density_estimate(thetahat)

#%%
# MLE - MAXIMUM LIKELIHOOD ESTIMATION
# thetahat_, thetahat0_,A_hat_m_ = m.ML_opt(A,B,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= 0*np.diag(np.ones(3)),opt_in='m')
# thetahat_mat = thetahat_.x


#%% q
# thetahat_elm_q, thetahat0, A_hat_q = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='e',stop_iter=1)
# thetahat_elm, thetahat0, A_hat_m = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='m')
# thetahat_elm_z, thetahat0, A_hat_z = m.ML_opt_elm(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,0*np.ones((3,3)),opt_in='z')
# #%%
# thetahat_param_s, thetahat0, A_hat_q = m.ML_opt_param(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=0,opt_in='single',log=True)
#
#
# #%% SOLVER DEPENDENCY
# methods = ['Nelder-Mead' ,'Powell' ,'CG' ,'BFGS' ,'Newton-CG' ,'L-BFGS-B' ,'TNC' ,'COBYLA' ,'SLSQP' ,'trust-constr','dogleg' ,'trust-ncg' ,'trust-exact' ,'trust-krylov']
# data = {}
# for method in methods:
#     thetahat_param_s, thetahat0, A_hat_q = m.ML_opt_param(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=0,opt_in='single',log=True)
#
#
# #%%
# # thetahat_param_m, thetahat0, A_hat_q = m.ML_opt_param(A,B,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt, thetahat0=1e-3,opt_in='multi',stop_iter=10,log=False)
#
#
# #%%
# fig, ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
#
#
# sets = ("True","MLE")
# data = {}
# data['True'] = m.p.params.values()
# data['MLE'] = thetahat_param_s
# # data['MLE_all'] = thetahat_param_m
#
# labs = list(m.p.params.keys())
# y = list(m.p.params.values())
#
# keys = list(m.p.params.keys())
# x = np.arange(len(keys))  # the label locations
# width = 0.25  # the width of the bars
# multiplier = 0
#
# for i, (k,v) in enumerate(data.items()):
#     offset = width * multiplier + 0.125
#     rects = ax.bar(x + offset, v, width, label=k,zorder=3)
#     # ax.bar_label(rects, padding=3)
#     multiplier += 1
#
# ax.set_xticks(x + width, keys)
# ax.legend(loc='upper left', ncols=1)
#
# ax.set(yscale='log')
# ax.grid()
#
#
#
# #%%
# A = m.A
# lambd, R = eig(A)
# L = inv(R)
#
# P = opt_sys = R @ L.T
#
# R_ = thetahat_elm @ inv(L.T)
#
# A = R_ @ np.diag(lambd) @ inv(R_)
#
#
# #%%
# asdf
#
# n = m.A_d.shape[0]
# fig, ax =plt.subplots(n+1,n,sharex=True,dpi=200,figsize=(12,9))
#
# for i in range(n+1):
#     for j in range(n):
#         if i ==0:
#             ax[i,j].plot(t,yhat[j,:],color='red',label=['$I_L$','$V_{C1}$','$V_{C2}$'][j])
#             ax[i,j].axhline(0,color='k',ls='-',lw=0.75)
#         else:
#             ax[i,j].set(yscale='symlog')
#
#             ax[i,j].axhline(0,color='k',ls=':',lw=0.75,label='0')
#             ax[i,j].axhline(m.A_d[i-1,j],color='k',ls='--',lw=0.75,label='True',zorder=3)
#             ax[i,j].plot(t,thetahat[i-1,j,:],label='LS')
#             ax[i,j].axhline(KDE[i-1,j],color='forestgreen',label='KDE',zorder=3)
#             ax[i,j].axhline(thetahat_mat[i-1,j],color='red',ls='-',label='MLE_mat')
#             ax[i,j].axhline(thetahat_elm[i-1,j],color='darkred',ls='-',lw=2,label='MLE_elm')
#             # ax[i,j].axhline(m.Ad_elm[i-1,j],color='k',ls=':',lw=0.75,label='MLE_elm')
#
# ax[-1,-1].legend(loc='upper right')
#
# # # ESTIMATIONS OF ELEMENTS
# # thetahat_elm, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)
#
#
# # --------------- VALIDATE ---------------
# # DISCRETIZE
# # Ad_mat,*_ = m.discretize(thetahat_mat, B, C, D, dt)
# # Ad_elm,*_ = m.discretize(thetahat_elm, B, C, D, dt)
#
# # # Simulate the system
# # _, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
# # _, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)
#
#
# #%%
# data = {'Rin':[],
#         'Rload':[],
#         'u_mode':[],
#         'that':[],
#         'phi':[],
#         'success':[],
#         'error_elm':[],
#         'error_mat':[],
#         'condition':[],
#         }
#
# t1 = -0.001
# t2 = 0.02
# t = np.arange(t1,t2+dt,dt)
#
# for u_mode in ['sin','step','impulse']:
#     for rin in [0.05,0.5,1]:
#         for rload in [5,43,100,1e6]:
#             for that in [False,True]:
#                 for phi in [0,np.pi/4]:
#                     # 'sin_0.5_100_False_0'
#                     if that:
#                         that0 = m.A
#                     else:
#                         that0 = -I(m.A.shape[0])*0
#
#                     # Define parameters
#                     params= {'Rin':rin,'V':1,'Vbase':66e3,'Rload': rload,'phi':phi}
#
#                     # Get model
#                     m = SITOBBDS()
#                     m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
#                     save_name=f'{u_mode}_{rin}_{rload}_{that}_{phi}'
#                     print(save_name,np.linalg.cond(m.A))
#
#                     # Create input
#                     u, uk = m.create_input(t1, t2, dt,mode=u_mode)
#
#                     # Get matrices
#                     Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
#
#                     # --------------- GET GROUND TRUTH ---------------
#                     # Simulate the system
#                     x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)
#
#                     # Filter the data with the Kalman Filter
#                     xhat, yhat,eps,R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)
#
#                     try:
#                         # --------------- ESTIMATE ---------------
#                         # ESTIMATION OF ENTIRE A
#                         thetahat_, thetahat0_ = m.ML_opt(Ad,Bd,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,thetahat0= that0)
#                         thetahat_mat = thetahat_.x
#                         # ESTIMATIONS OF ELEMENTS
#                         thetahat_elm, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)
#
#                         # --------------- VALIDATE ---------------
#                         # DISCRETIZE
#                         Ad_mat,*_ = m.discretize(thetahat_mat, B, C, D, dt)
#                         Ad_elm,*_ = m.discretize(thetahat_elm, B, C, D, dt)
#
#                         # Simulate the system
#                         _, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
#                         _, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)
#
#                         # --------------- PLOT ---------------
#                         m.plot_simulations(t, [y,yhat,yhat_mat,yhat_elm],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$','$\\hat{y}_{MLE_{ELM}}$'],save=save_name,file_extension='png')
#
#                         m.plot_estimation_summary(thetahat_elm, thetahat0,save=save_name + '_elm',file_extension='png')
#                         m.plot_estimation_summary(thetahat_mat, thetahat0_,save=save_name + '_mat',file_extension='png')
#                         m.plot_eigenvalues([m.A,thetahat_elm,thetahat_mat],['Ground truth','MLE - Element-wise','MLE - Matrix-wise'],save=save_name,file_extension='png')
#
#                         # --------------- STORE VALUES ---------------
#                         data['Rin'].append(rin)
#                         data['Rload'].append(rload)
#                         data['u_mode'].append(u_mode)
#                         data['that'].append(that)
#                         data['phi'].append(phi)
#                         data['error_elm'].append(np.linalg.norm(thetahat_elm - m.A))
#                         data['error_mat'].append(np.linalg.norm(thetahat_mat - m.A))
#                         data['condition'].append(np.linalg.cond(m.A))
#                         data['success'].append(False)
#
#                     except (np.linalg.LinAlgError) as e:
#                         # --------------- PLOT ---------------
#                         m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'],save=save_name,file_extension='png')
#
#                         # --------------- STORE VALUES ---------------
#                         data['Rin'].append(rin)
#                         data['Rload'].append(rload)
#                         data['u_mode'].append(u_mode)
#                         data['that'].append(that)
#                         data['phi'].append(phi)
#                         data['error_elm'].append(np.nan)
#                         data['error_mat'].append(np.nan)
#                         data['condition'].append(np.linalg.cond(m.A))
#                         data['success'].append(False)
#
# df = pd.DataFrame(data)
# df.to_excel('img/data.xlsx',header=True,index=False)
#
#
# #%%
# m = SITOBBDS()
# # Load model
# print('Available models:\n',m.models)
# params= {'Rin':0.5,'V':1,'Vbase':66e3,'Rload': 100,'phi':np.pi/4}
# m.get_model('c1_s0_o3_load',discretize=True,dt=10e-6,params=params,pu=True)
#
# u, uk = m.create_input(t1, t2, dt,mode='sin')
#
# # Get matrices
# Ad, Bd, A, B, C, D = m.A_d,m.B_d,m.A,m.B,m.C,m.D
#
# # Simulate the system
# x, y = m.simulate(Ad,Bd,C,D,x0,uk,t1,t2,dt)
#
# # Filter the data with the Kalman Filter
# xhat, yhat,eps,R = m.KalmanFilter(Ad, Bd, C, D, x0,uk, y, R0, R1, R2, t1, t2, dt)
#
#
#
# # Plot simulation against the filtered values
# m.plot_simulations(t, [y,yhat],labels=['$y$','$\\hat{y}_{KF}$'])
#
# # asdf
#
# #%% ESTIMATION OF ENTIRE A
# that0 = -I(m.A.shape[0])*0
# that0 = m.A
# thetahat_, thetahat0_ = m.ML_opt(Ad,Bd,C,D, x0, uk, y, R0, R1, R2, t1, t2, dt,V_theta0=1,thetahat0= that0)
#
# #%% ESTIMATIONS OF ELEMENTS
# thetahat, thetahat0 = m.ML_opt_elm(Ad,Bd,C,D,x0, uk, y, R0, R1, R2, t1, t2, dt,that0)
#
# #%%
# Ad_mat,*_ = m.discretize(thetahat_.x, B, C, D, dt)
# Ad_elm,*_ = m.discretize(thetahat, B, C, D, dt)
#
# # Simulate the system
# _, yhat_mat = m.simulate(Ad_mat,Bd,C,D,x0, uk, t1,t2,dt)
# _, yhat_elm = m.simulate(Ad_elm,Bd,C,D,x0, uk, t1,t2,dt)
#
# #%%
# m.plot_simulations(t, [y,yhat,yhat_mat,yhat_elm],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$','$\\hat{y}_{MLE_{ELM}}$'],save='')
# m.plot_simulations(t, [y,yhat,yhat_mat],labels=['$y$','$\\hat{y}_{KF}$','$\\hat{y}_{MLE_{MAT}}$'],save='')
#
# #%% PRINT ESTIMATION SUMMARY
# m.plot_estimation_summary(thetahat_elm, np.zeros((3,3)))
# m.plot_estimation_summary(thetahat_mat, np.zeros((3,3)))
# m.plot_eigenvalues([m.A_d,thetahat_elm,thetahat_mat],['Ground truth','MLE - Element-wise','MLE - Matrix-wise'])
#
# #%%
# asdf
#
# #%% PLOT COVARIANCES IN TIME
# m.plot_covariance_in_time(R)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
