"""
Plot planes from joint analysis files.

Usage:
    plot_analysis.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from dedalus.extras import plot_tools


    # Plot writes
with h5py.File("analysis/analysis_s1.h5", mode='r') as file:
    # Load datasets
    E1 = file['tasks']['<E1>']
    t = E1.dims[0]['sim_time']

    I = file['tasks']['<I>']    
    D = file['tasks']['<D>']
    
    cxx = file['tasks']['<cxx>']
    cxy = file['tasks']['<cxy>']
    cyy = file['tasks']['<cyy>']        
    u1 =  file['tasks']['<u1>']      # mean velocity components
    v1 =  file['tasks']['<v1>']      # mean velocity components



    # convert time array into dataframe and save to file
    DF = pd.DataFrame(t[:]) 
    DF.to_csv("t.out", sep = ' ')
    
    # Save to ascii files
    np.savetxt("E1.out", E1[:,0,0], delimiter=" ")
    np.savetxt("I.out", I[:,0,0], delimiter=" ")
    np.savetxt("D.out", D[:,0,0], delimiter=" ")
    np.savetxt("cxx.out", cxx[:,0,0], delimiter=" ")
    np.savetxt("cxy.out", cxy[:,0,0], delimiter=" ")                                                            
    np.savetxt("cyy.out", cyy[:,0,0], delimiter=" ")
    np.savetxt("u1.out", u1[:,0,0], delimiter=" ")
    np.savetxt("v1.out", v1[:,0,0], delimiter=" ")                


# Reload time file and remove first line...
with open('t.out', 'r') as fin:
    data = fin.read().splitlines(True)
with open('t.out', 'w') as fout:
    fout.writelines(data[1:])    
    

