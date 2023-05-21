

import sys
sys.path.insert(0, '../')  # add parent folder to system path
import numpy as np
import matplotlib.pyplot as plt
# from dynamic_simulation import Solver
# from datareader import DataReader
from solver import SITOBBDS
from PowerSystem import PS
import control as ctrl
import pandas as pd
import os
from Plots import grouped_bar


#%%
r_path = r'C:\Users\bvilm\PycharmProjects\SITOBB\data\estimation results'
files = [f for f in os.listdir(r_path) if '.xlsx' in f]

for i, file in enumerate(files):
    if i == 0:
        df = pd.read_excel(f'{r_path}\\{file}',index_col=0,header=0)
    else:
        df = pd.merge(df, pd.read_excel(f'{r_path}\\{file}',index_col=0,header=0),left_index=True,right_index=True,how='outer')
        # df = pd.merge(df, pd.read_excel(f'{r_path}\\{file}',index_col=0,header=0),left_index=True,right_index=True,join='outer')
    print(i,df)



#%%
first_cols = ['True','Individual']
df = df[first_cols + [col for col in df.columns if col not in first_cols]]

fig, ax = plt.subplots(1,1,figsize=(9,4),dpi=200)
ax = grouped_bar(ax,df,ax_kwargs=dict(yscale='log'),legend_kwargs=dict(loc='lower right',ncol=3),comp_values=df['True'].values)


