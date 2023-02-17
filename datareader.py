import os
import pandas as pd
import numpy as np
from xml.dom import minidom
import warnings
import pandas as pd
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup

class DataReader:
    def __init__(self):
        """
        The DataReader reads the mapping of signals in the auto generated .infx file from PSCAD
        and returns the corresponding signal from the data signal.

        """

        return

    def read_data_file(self, num):
        return np.genfromtxt(self.directory + '\\{}_{:02d}.out'.format(self.project_name, num),skip_header=1)

    def get_signal(self,
                   signal:      str,
                   file_addr:   str,
                   convert:     bool = False
                   ):
        self.directory = os.path.split(file_addr)[0]
        self.project_name = os.path.splitext(os.path.split(file_addr)[1])[0]
        self.xmldoc = minidom.parse(file_addr)

        if convert:
            signal = self.convert_signal_name(signal)

        S = 0  # Sum variable to count number of items extracted from range
        N = 0  # Sum number of opened files

        # Getting index and dimension of channel signal
        idx = int([node.getAttribute('index') for node in self.xmldoc.getElementsByTagName("Analog") if
                     signal in node.getAttribute('name')][0])
        dim = int([node.getAttribute('dim') for node in self.xmldoc.getElementsByTagName("Analog") if
                     signal in node.getAttribute('name')][0])

        # Finding iterator values
        Q = int(idx) // 10 + 1  # file number
        R = int(idx) % 10 + 1   # R = 0 => R_idx = 1
        r = (R - 1 + dim) % 10 + 1  # - 1: arr_idx => infx_idx, + 1: infx_idx => arr_idx
        nfiles = (R - 1 + dim) // 10 + (1,0)[r == 1]  # Files to open: - 1: arr_idx => infx_idx, + 1: cnt must count be > 0

        # Read file:
        data = self.read_data_file(Q)
        N += 1

        # Finding all dimensions if stored in multiple files
        # If dimension plus current index is > 10
        if dim + R > 10:
            for i in range(nfiles):
                if i == 0:
                    #print('A: Getting data:\t',f'[:,{R}:]')
                    y = data[:,R:]
                    S+= data[:,R:].shape[1]
                    continue
                else:
                    #print('B: read:\t\t\t',Q+i)
                    data = self.read_data_file(Q+i)
                    N += 1

                if dim - S >= 10:
                    #print('C: Getting data:\t',f'[:,1:]')
                    ys = data[:,1:]
                    S += data[:,1:].shape[1]
                else:
                    #print('D: Getting data:\t',f'[:,1:{r}]')
                    ys = data[:,1:r]
                    S += data[:,1:r].shape[1]

                y = np.concatenate((y,ys),axis=1)
        else:
            #print('Getting data:\t',f'[:,{R}:{R+dim}]')
            y = data[:,R:R+dim]
            S += data[:, R:R+dim].shape[1]
        t = data[:, 0]

        # Output data validation
        if S != dim:
            warnings.warn(f'{self.directory}\\{self.project_name} is not reading {signal} correctly:\nS != dim\n{S} != {dim}')
        if N != nfiles:
            warnings.warn(f'{self.directory}\\{self.project_name} is not reading {signal} correctly:\nN != nfiles\n{N} != {nfiles}')

        return (t, y)
