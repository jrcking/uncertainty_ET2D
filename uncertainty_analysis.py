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
    E2 = file['tasks']['<E2>']
    dE = file['tasks']['<dE>']
    dcxx = file['tasks']['<dcxx>']
    dcxy = file['tasks']['<dcxy>']
    dcyy = file['tasks']['<dcyy>']        
    dIP =  file['tasks']['<dIP>']      # inertial production
    dVD =  file['tasks']['<dVD>']      # viscous dissipation
    dPS =  file['tasks']['<dPS>']      # polymeric stress
    u1 =  file['tasks']['<u1>']      # mean velocity components
    u2 =  file['tasks']['<u2>']      # mean velocity components
    v1 =  file['tasks']['<v1>']      # mean velocity components
    v2 =  file['tasks']['<v2>']      # mean velocity components            

    dtrC = file['tasks']['<dtrC>'] #delta(tr(c))  
    dUC1 = file['tasks']['<dUC1>'] #upper convected terms
    dUC2 = file['tasks']['<dUC2>'] #upper convected terms
    dUC3 = file['tasks']['<dUC3>'] #upper convected terms
    dCR = file['tasks']['<dCR>'] #relaxation terms

    dCdC = file['tasks']['<dCdC>'] # Pi_Delta           
    dG = file['tasks']['<dG>'] # square of delta(tr(c))
    dG_ad = file['tasks']['<dG_ad>'] #advection terms
    dG_UC1 = file['tasks']['<dG_UC1>'] #upper convected terms
    dG_UC2 = file['tasks']['<dG_UC2>'] #upper convected terms
    dG_UC3 = file['tasks']['<dG_UC3>'] #upper convected terms
    dG_R = file['tasks']['<dG_R>'] #relaxation terms
    dG_D = file['tasks']['<dG_D>'] #diffusive terms
     
            


    # convert time array into dataframe and save to file
    DF = pd.DataFrame(t[:]) 
    DF.to_csv("t.out", sep = ' ')
    
    # Save to ascii files
    np.savetxt("E1.out", E1[:,0,0], delimiter=" ")
    np.savetxt("E2.out", E2[:,0,0], delimiter=" ")
    np.savetxt("dE.out", dE[:,0,0], delimiter=" ")
    np.savetxt("dcxx.out", dcxx[:,0,0], delimiter=" ")
    np.savetxt("dcxy.out", dcxy[:,0,0], delimiter=" ")                                                            
    np.savetxt("dcyy.out", dcyy[:,0,0], delimiter=" ")
    np.savetxt("dIP.out", dIP[:,0,0], delimiter=" ")   
    np.savetxt("dVD.out", dVD[:,0,0], delimiter=" ")   
    np.savetxt("dPS.out", dPS[:,0,0], delimiter=" ")       
    np.savetxt("u1.out", u1[:,0,0], delimiter=" ")
    np.savetxt("u2.out", u2[:,0,0], delimiter=" ")       
    np.savetxt("v1.out", v1[:,0,0], delimiter=" ")       
    np.savetxt("v2.out", v2[:,0,0], delimiter=" ")                          
    np.savetxt("dtrC.out", dtrC[:,0,0], delimiter=" ")
    np.savetxt("dUC1.out", dUC1[:,0,0], delimiter=" ")              
    np.savetxt("dUC2.out", dUC2[:,0,0], delimiter=" ")              
    np.savetxt("dUC3.out", dUC3[:,0,0], delimiter=" ")                      
    np.savetxt("dCR.out", dCR[:,0,0], delimiter=" ")                          

    np.savetxt("dCdC.out", dCdC[:,0,0], delimiter=" ")           
    np.savetxt("dG.out", dG[:,0,0], delimiter=" ")           
    np.savetxt("dG_ad.out", dG_ad[:,0,0], delimiter=" ")           
    np.savetxt("dG_UC1.out", dG_UC1[:,0,0], delimiter=" ")                       
    np.savetxt("dG_UC2.out", dG_UC2[:,0,0], delimiter=" ")                       
    np.savetxt("dG_UC3.out", dG_UC3[:,0,0], delimiter=" ")                           
    np.savetxt("dG_R.out", dG_R[:,0,0], delimiter=" ")                       
    np.savetxt("dG_D.out", dG_D[:,0,0], delimiter=" ")                           


# Reload time file and remove first line...
with open('t.out', 'r') as fin:
    data = fin.read().splitlines(True)
with open('t.out', 'w') as fout:
    fout.writelines(data[1:])    
    

