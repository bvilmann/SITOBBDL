import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import Assignment4_helpfunctions.P_matrix_write as P_mat
import Assignment4_helpfunctions.Load_parameters as lp
import Assignment4_helpfunctions.plot_phasor_diagram_sss as pp

class SmallSignalStability:

    def __init__(self):
        self.lp = lp
        
        return

    def load_system(self,data_file):
        self.filename = data_file
        # Reading data from .mat file
        self.data = data = loadmat(f'Assignment4_data/{data_file}.mat')
        self.A = data[list(data.keys())[3]]
        self.ltx_names = data[list(data.keys())[4]]
        self.dat_names = data[list(data.keys())[5]]

        # Get eigenvalues, right eigenvector, and left eigenvector.
        self.lamb, self.Phi = np.linalg.eig(self.A)
        self.Psi = np.linalg.inv(self.Phi)

        # Get participation factors
        self.P = self.Phi * self.Psi.T

        # Get damping and frequencies
        self.omegas = np.array([np.imag(l) for l in self.lamb])
        self.f = np.array([np.imag(l)/(2*np.pi) for l in self.lamb])
        self.zeta = np.array([-np.real(l)/(abs(l)) for l in self.lamb])

        return data


    def plot_eigs(self,leg_loc=None,mode='complex',xlim=None,ylim=None):
        fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=200)
        marks = ['o','*','+','s','d']
        for i, l in enumerate(self.lamb):
            if mode=='complex':
                if np.imag(l) > 0:
                    ax.scatter([np.real(l),np.real(l)],[np.imag(l),-np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
            else:
                if np.imag(l) > 0:
                    ax.scatter([np.real(l),np.real(l)],[np.imag(l),-np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
                elif np.imag(l)< 0:
                    continue
                else:
                    ax.scatter([np.real(l)],[np.imag(l)],label='$\\lambda_{'+str(i+1)+'}$',marker=marks[i//12])
                
        ax.axhline(0,color='k',lw=0.75)
        ax.axvline(0,color='k',lw=0.75)
        # ax.set_title(self.filename)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if leg_loc is not None:
            ax.legend(loc=leg_loc,ncol=(1,3)[len(self.lamb) > 24])
        else:            
            ax.legend(ncol=(1,3)[len(self.lamb) > 24])

        ax.grid()

        plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_eig.pdf')
        plt.close()

        # plt.show()
        return
    
    def solve(self):
        
        return

    def plot_P_matrix(self):
        if self.P.shape[0] > 24:
            fig,ax = plt.subplots(1,1,figsize=(9,9),dpi=200)
        elif self.P.shape[0] == 24:
            fig,ax = plt.subplots(1,1,figsize=(8,8),dpi=200)
        else:
            fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=200)
        im = ax.imshow(np.where(abs(self.P)>0.02,abs(self.P),np.nan),vmin=0, vmax=1)
        ax.set_xticks([i for i in range(len(self.dat_names))])
        ax.set_yticks([i for i in range(len(self.dat_names))])
        ax.set_xticklabels(['$\\lambda_{' + str(i) +'}$' for i in range(1,len(self.dat_names)+1)])
        ax.set_yticklabels(['$'+f'{x[0][0]}'+'$' for x in self.ltx_names])
        # fig.colorbar(im, ax=ax, location='right', anchor=(0.2, 0.2))

        # c = plt.colorbar(im, cax = fig.add_axes([0.78, 0.5, 0.03, 0.38]))

        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        divider = make_axes_locatable(ax)
    
        ax_cb = divider.append_axes("right", size="5%", pad=0.1)
        fig = ax.get_figure()
        fig.add_axes(ax_cb)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm), cax=ax_cb,extend = 'max')

        ax_cb.yaxis.tick_right()
        # ax_cb.yaxis.set_tick_params(labelright=False)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.P.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.P.shape[0], 1), minor=True)

        if self.P.shape[0] > 24:
            ax.set_xticklabels([("$\\lambda_{"+str(i+1)+"}$","")[i%2 == 1] for i in range(len(self.P))])
        
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)

        fig.tight_layout()

        plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_P.pdf')

        # plt.show()
        plt.close()

        return

    def phasor(self,z):
        A = round(abs(z),3)
        ang = round(np.angle(z)*180/np.pi,3)
        return f'{A}<{ang}'

    def damp(self):
        data = {
            # 'lambda':[self.phasor(l) for l in self.lamb],
            'real':np.real(self.lamb),
            'imag':np.imag(self.lamb),
            'f':self.f,
            'zeta':self.zeta,
            }
        self.damp_df = df = pd.DataFrame(data,index= [i for i in range(1,len(self.lamb)+1)])
        print(df)
        return

    def P2latex(self,ncols:int,threshold:float,full_matrix:bool = False,scale=None):
        osc_idx = [i for i, v in enumerate(self.lamb) if (np.imag(v) > 0) or (full_matrix)]            
        lambas_labels = [i+1 for i in osc_idx]

        P_mat.latex_P_matrix(self.P[:,osc_idx],self.ltx_names,True,self.filename,ncols,threshold,lambas_labels,scale=scale)

    def eigs2latex(self,comm_col:bool = False):

        fid = open(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\tabs\\{self.filename}_eig.tex','w') # Open file with write access
        fid.write('\r\\begin{table}[H]')
        fid.write('\r\\centering')
        fid.write('\r\\caption{}')
        fid.write('\r\\label{tab:'+self.filename+'_eig}')
        fid.write('\r\\begin{tabular}{|c|' + '|'.join(['r']*(4,5)[comm_col]) +'|}')

        if comm_col:
            fid.write('\\hline'+'&'.join([r'\textbf{'+col+'}' for col in ['No.','$\\Re(\\lambda)$','$\\Im(\\lambda)$','$f$','$\\zeta$','Dominant States']]) + '\\\\\\hline\r')
        else:
            fid.write('\\hline'+'&'.join([r'\textbf{'+col+'}' for col in ['No.','$\\Re(\\lambda)$','$\\Im(\\lambda)$','$f$','$\\zeta$']]) + '\\\\\\hline\r')
        
        for i, r in self.damp_df.iterrows():

            if r.imag > 0:
                fid.write(f'{i},{i+1}&' + '&'.join([('','$\\pm$')[k == 'imag'] + str(round(v,3)) for k,v in r.items()]) + ('','& ')[comm_col] + '\\\\\r')
                
            elif r.imag < 0:
                continue
            else:                
                fid.write(f'{i}&' + '&'.join([str(round(v,3)) for k,v in r.items()]) + ('','& ')[comm_col] + '\\\\\r')
            
        fid.write('\\hline')
        fid.write('\r\\end{tabular}\r\r')
        fid.write('\r\\end{table}\r\r')

        return

    def plot_phasor(self,lamb_no,gens:list):
        clrs = plt.rcParams['axes.prop_cycle']
        C = [c['color'] for c in list(clrs)]
        lamb_idx = [n-1 for n in lamb_no]
        names = [n[0] for n in self.ltx_names[:,0]]
        idx = []
        nms = []
        for i, name in enumerate(names):
            # print([int(i) for i in str(name) if i.isdigit()])
            G = [int(i) for i in str(name) if i.isdigit()][0]
            # print(G,name)            
            if '\\Delta\\delta' in name and G in gens:
                idx.append(i)
                nms.append(f'${name}$')
        # idx = [i for i, v in enumerate(zip(names,gens)) if '\\Delta\\delta' in v[0] and [int(i) for i in str(v[0]) if i.isdigit()][0] in gens]
        # names = ['$' + v[0] + '$' for i, v in enumerate(zip(names,gens)) if '\\Delta\\delta' in v[0] and [int(i) for i in str(v[0]) if i.isdigit()] in gens]
                
        print(idx)
        print(lamb_idx)
        
        vals = self.Phi[idx,lamb_idx]
        
        print(self.Phi)
        print(vals)
        P = np.zeros((vals.shape[0],2))
        P[:,0] = np.real(vals)
        P[:,1] = np.imag(vals)
 
        # print(P)   
 
        fig, ax = pp.plot_phasors(P,C[:len(idx)],nms)       

        ax.grid()    

        fig.tight_layout()        
        # plt.show()
        plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_{lamb_no}_{gens}.pdf',bbox_inches='tight')

        plt.close()                      
        return


    def dynamic_simulation(self,x0,t0,t1,dt = 0.001):
        t = np.arange(t0,t1,dt)

        dx = np.zeros((len(self.lamb),len(t)), dtype=np.complex)
        
        for k in range(0,len(t)):
            dx[:,k] = self.Phi.dot(np.exp(self.lamb*t[k])*(self.Psi).dot(x0))

        data = {f'${self.ltx_names[i,0][0]}$': list(dx[i,:]) for i in range(len(self.lamb))}
        df = pd.DataFrame(data,index=list(t))        

        return df


    def plot_time_response(self, fig, ax, x0, t0=0, t1=5, xlabel=False):        

        fig, ax = plt.subplots(1,1)

   
        ax.plot(self.t_plot_tr, self.dx_plot_tr[0], label=self.filename)
        
        ax.grid(linestyle='-.', linewidth=.25)
        ax.grid(linestyle=':', which='minor', linewidth=.25, alpha=.5)
        if xlabel:
            ax.set_xlabel('time [s]')
        ax.set_ylabel(self.filename)

        fig.tight_layout()        

        plt.savefig(f'C:\\Users\\bvilm\\Dropbox\\Apps\\Overleaf\\46710 - Stability and control - A3\\img\\{self.filename}_series.pdf')
        
        return fig, ax
    
    def x0(self):
        self.x0 = np.concatenate(([0.087266]+[np.zeros((len(self.dat_names)-1,1))]), axis=None)
        return
    
    def save_time_response(self):
        return
