import matplotlib.pyplot as plt
import numpy as np

# Define plt parameters
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


class results_wrapper:
    def __init__(self,results):
        for k,v in results.items():
            setattr(self,k,v)
        return

class Plotter:
    def __init__(self):

        return



