# uncertainty_ET2D
Scripts for simulations in uncertainty paper.

Modify *unc_params.py* to set parameters as desired.

*mpirun -n X python precursor.py* to run precursor simulation for X time units.

*mpirun -n X python uncertaintyX.py --restart* to run a single uncertainty evolution simulation. 0p0001, 0p001, 0p05 run short times, 0,10,20,etc run for a full 50 time units with different starting times.

*python uncertainty_analysis.py* will convert all the statistics files into ascii files, for non-python people like me to load with something else for plotting...

*mpirun -n uncertainty_snapshots ./snapshots/*.h5* will create ascii files for all fields (u1,u2,v1,v2,cxx1,cxx2 etc) which can be loaded with whatever for plotting/analysis. Can also be modified to directly product images.


