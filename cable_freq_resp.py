import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

class LCP_reader:
    
    def __init__(self):  
        
        return
    
    def load(self,name,max_freq:float=None):
        for file in ['zm','zp','ym','yp']:
            d = np.genfromtxt(f'data/LCP/{name}_{file}.out',skip_header=1)
            f = d[:,1]
            d = d[:,2:]
    
            if file[1] == 'p':
                print(file)
                d = d*np.pi/180
    
            # Get number of cables
            N = int(np.sqrt(d.shape[1]))            
            if N != np.sqrt(d.shape[1]):
                raise ValueError('Not squared number of data rows')
                
            # Create names
            names = [f'Z{i+1}{j+1}' for i in range(N) for j in range(N)]
            print(names)

            # Dataframe
            df = pd.DataFrame(columns=names,data = d,index=f)
            
            if max_freq is not None:
                df = df[df.index <= max_freq]

            # df.plot(legend=False,title=file,logx=True,logy=True)

            setattr(self,file,df)

        setattr(self,'Z',self.zm*np.cos(self.zp) + 1j*self.zm*np.sin(self.zp))
        setattr(self,'Y',self.ym*np.cos(self.yp) + 1j*self.ym*np.sin(self.yp))
        
        setattr(self,'H',np.exp(-np.sqrt(self.Z*self.Y)))
        setattr(self,'hm',np.abs(self.H))
        setattr(self,'hp',pd.DataFrame(columns=names,data = np.angle(self.H),index=[_ for _ in f if _ <= max_freq]))

        return

        
    def plot(self,mode):
        if mode == 'zy_ii':
            fig, ax = plt.subplots(2,2,dpi=200,figsize=(8,8),sharex=True)
            cols = [col for col in lcp.zp.columns if col[1] == col[2]]
            for i, c in enumerate(cols):
                ax[0,0].plot(lcp.zm[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
                ax[0,1].plot(lcp.zp[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
                ax[1,0].plot(lcp.ym[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
                ax[1,1].plot(lcp.yp[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
            
                ax[0,0].axhline(0,color='k',lw=0.75)
                ax[0,1].axhline(0,color='k',lw=0.75)
                ax[0,1].set(yscale='symlog',xscale='log')
                ax[1,1].set(yscale='symlog',xscale='log')
            ax[-1,-1].legend(ncol=2)            
        elif mode == 'zy_ij':
            fig, ax = plt.subplots(2,2,dpi=200,figsize=(11,11),sharex=True)
            cols = [col for col in lcp.zp.columns if col[1] != col[2]]
            for i, c in enumerate(cols):
                _, a, b = c
                a, b = int(a), int(b)
            
                for var in [a,b]:    
                    if a in [1,3,5]:
                        a = 'c'
                        continue
                    elif a in [2,4,6]:
                        a = 's'
                        continue
                    else:
                        a = 'p'
            
                if a in [1,3,5] or b in [1,3,5]:
                    ls = '--'
                elif a == [2,4,6] or b in [2,4,6]:
                    ls = ':'
                else:
                    ls = '-'
            
                ax[0,0].plot(lcp.zm[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
                ax[0,1].plot(lcp.zp[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
                ax[1,0].plot(lcp.ym[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
                ax[1,1].plot(lcp.yp[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
            
                ax[0,0].axhline(0,color='k',lw=0.75)
                ax[0,1].axhline(0,color='k',lw=0.75)
                ax[0,1].set(yscale='log',xscale='log')
                ax[1,1].set(yscale='log',xscale='log')
            ax[-1,-1].legend(ncol=4)
        if mode == 'h_ii':
            fig, ax = plt.subplots(2,1,dpi=200,figsize=(8,8),sharex=True)
            cols = [col for col in lcp.zp.columns if col[1] == col[2]]
            for i, c in enumerate(cols):
                ax[0].plot(lcp.hm[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
                ax[1].plot(lcp.hp[c],label=c,ls=('--',':','-.')[i%3],lw=2*(1,1.75,0.9)[i%3])
            
                ax[1].axhline(0,color='k',lw=0.75)
                ax[1].set(yscale='symlog',xscale='log')
            ax[-1].legend(ncol=2)            
        elif mode == 'h_ij':
            fig, ax = plt.subplots(2,1,dpi=200,figsize=(11,11),sharex=True)
            cols = [col for col in lcp.zp.columns if col[1] != col[2]]
            for i, c in enumerate(cols):
                _, a, b = c
                a, b = int(a), int(b)
            
                for var in [a,b]:    
                    if a in [1,3,5]:
                        a = 'c'
                        continue
                    elif a in [2,4,6]:
                        a = 's'
                        continue
                    else:
                        a = 'p'
            
                if a in [1,3,5] or b in [1,3,5]:
                    ls = '--'
                elif a == [2,4,6] or b in [2,4,6]:
                    ls = ':'
                else:
                    ls = '-'
            
                ax[0].plot(lcp.hm[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
                ax[1].plot(lcp.hp[c],label=c,ls=ls,lw=2*(1,1.75,0.9)[i%3])
            
                ax[1].axhline(0,color='k',lw=0.75)
                ax[1].set(yscale='symlog',xscale='log')
            ax[0].legend(ncol=4)
        
        return        
        
lcp = LCP_reader()

lcp.load(r'Cable_2',max_freq=5e3)

lcp.zm
#%%
lcp.plot('zy_ii')
lcp.plot('zy_ij')
lcp.plot('h_ii')
lcp.plot('h_ij')










