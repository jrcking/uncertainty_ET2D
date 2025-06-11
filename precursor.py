"""
This script runs an initial simulation to obtain a statistically steady elastic turbulent state.
It saves the output every 1 time units, including a second copy, so the entire thing can be loaded into
the uncertainty simulation.

"""

import sys
import numpy as np
import dedalus.public as d3
import logging
import unc_params
logger = logging.getLogger(__name__)

# Allow restarting via the command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
n = unc_params.n
L = n # We want characteristic Length scale L=L0/n=1
Nx = unc_params.Nx
k0 = 2.0*np.pi/L # base-wavenumber
timestep = unc_params.timestep

# Dimensionless quantities
Re = unc_params.Re
Wi= unc_params.Wi
beta=unc_params.beta
eps = unc_params.eps
kappa=unc_params.kappa
Deltat0=unc_params.Deltat0
ptbnmag=unc_params.ptbnmag

# Forcing magnitude
f0 = 4*np.pi*np.pi/Re  # this will result in Newtonian fixed point with max(U)=1

# Time-integration controls
dealias = 3/2
stop_sim_time = 315
timestepper = d3.RK222
dtype = np.float64

# Substitutions for coefficients
coef_s = beta/Re # Solvent viscous coefficient
coef_p = (1.0-beta)/(Re*Wi) # Polymeric coefficient
ooWi = 1.0/Wi

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-L/2, L/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(-L/2, L/2), dealias=dealias)

# Fields for realisation 1
u1 = dist.Field(name='u1', bases=(xbasis,ybasis))
v1 = dist.Field(name='v1', bases=(xbasis,ybasis))
p1 = dist.Field(name='p1', bases=(xbasis,ybasis))
cxx1 = dist.Field(name='cxx1', bases=(xbasis,ybasis))
cxy1 = dist.Field(name='cxy1', bases=(xbasis,ybasis))
cyy1 = dist.Field(name='cyy1', bases=(xbasis,ybasis))
tau_p1 = dist.Field(name='tau_p1')  # Penalty term for incompressibility
# Copies for 2nd field
u2 = dist.Field(name='u2', bases=(xbasis,ybasis))
v2 = dist.Field(name='v2', bases=(xbasis,ybasis))
cxx2 = dist.Field(name='cxx2', bases=(xbasis,ybasis))
cxy2 = dist.Field(name='cxy2', bases=(xbasis,ybasis))
cyy2 = dist.Field(name='cyy2', bases=(xbasis,ybasis))
p2 = dist.Field(name='p2', bases=(xbasis,ybasis))
tau_p2 = dist.Field(name='tau_p2')

t = dist.Field()
bforcex = dist.Field(name='bforcex', bases=(xbasis,ybasis))  # body force field
bforcey = dist.Field(name='bforcey', bases=(xbasis,ybasis))  # body force field



# Define x and y gradients
dx = lambda A: d3.Differentiate(A, d3.Coordinate('x'))
dy = lambda A: d3.Differentiate(A, d3.Coordinate('y'))

# Define coordinates, vectors and body force
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
bforcex['g'] = f0*np.sin(n*y*k0)
bforcey['g'] = f0*np.sin(n*x*k0)


# Problem definition ==============================================================================
problem = d3.IVP([u1,v1,cxx1,cxy1,cyy1,p1,tau_p1,u2,v2,cxx2,cxy2,cyy2,p2,tau_p2], time=t, namespace=locals())
        
# Momentum
problem.add_equation("dt(u1) + dx(p1) - coef_s*lap(u1) - coef_p*(dx(cxx1)+dy(cxy1)) = \
        -u1*dx(u1) - v1*dy(u1) + bforcex")
problem.add_equation("dt(v1) + dy(p1) - coef_s*lap(v1) - coef_p*(dx(cxy1)+dy(cyy1)) = \
        -u1*dx(v1) - v1*dy(v1) + bforcey")                
        
# sPTT        
problem.add_equation("dt(cxx1) - ooSc*lap(cxx1) = - u1*dx(cxx1) - v1*dy(cxx1) + 2.0*cxx1*dx(u1) + 2.0*cxy1*dy(u1) \
        - (cxx1-1.0)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi")
problem.add_equation("dt(cxy1) - ooSc*lap(cxy1) = - u1*dx(cxy1) - v1*dy(cxy1) + cxx1*dx(v1) + cyy1*dy(u1) \
        + cxy1*(dx(u1)+dy(v1)) - (cxy1)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi")
problem.add_equation("dt(cyy1) - ooSc*lap(cyy1) = - u1*dx(cyy1) - v1*dy(cyy1) + 2.0*cxy1*dx(v1) + 2.0*cyy1*dy(v1) \
        - (cyy1-1.0)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi") 

problem.add_equation("u2-u1=0")
problem.add_equation("v2-v1=0")
problem.add_equation("cxx2-cxx1=0")
problem.add_equation("cxy2-cxy1=0")
problem.add_equation("cyy2-cyy1=0")
problem.add_equation("p2-p1=0")
problem.add_equation("tau_p2-tau_p1=0")

#---------------------------------------------------------
# Divergence free constraints
problem.add_equation("dx(u1) + dy(v1) + tau_p1 = 0")  
problem.add_equation("integ(p1) = 0") # Mean pressure =0


# =================================================================================================

# Solver ==========================================================================================
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
# =================================================================================================

# Initial conditions ==============================================================================

# Initial conditions
if not restart:
    file_handler_mode = 'overwrite'
    u1['g'] = 0.0 + 0.0*np.sin(n*y*k0) + 0.2*np.cos(0.5*n*y*k0+1.1)# + 0.1*rng.standard_normal()# Base flow
    v1['g'] = 0.0 + 0.0*np.sin(n*x*k0) + 0.4*np.sin(0.5*n*x*k0+3.9)  # A little divergence free velocity perturbation
    cxx1['g'] = 1.0 
    cxy1['g'] = 0.0
    cyy1['g'] = 1.0 
else:
    write, initial_timestep = solver.load_state('init0/checkpoints_pre_s45.h5')
    initial_timestep = timestep
    file_handler_mode = 'overwrite'


# Snapshots output ================================================================================
# Fields for visualisation
#snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10.0, max_writes=10)
#snapshots.add_task(cxx1+cyy1, name='trc1')
#snapshots.add_task(cxy1, name='cxy1')
#snapshots.add_task(dx(v1)-dy(u1), name='vorticity1')
#snapshots.add_task(u1, name='u')
#snapshots.add_task(v1, name='v')

# Checkpoints - necessary for restarts!
checkpoints_pre = solver.evaluator.add_file_handler('checkpoints_pre', sim_dt=5.0, max_writes=1, mode=file_handler_mode)
checkpoints_pre.add_tasks(solver.state)

# Statistical analysis ============================================================================
# Mostly volume averaged quantities to explore evolution of uncertainty
analysis = solver.evaluator.add_file_handler('analysis', sim_dt=0.01, max_writes=1000000)

# Kinetic energy of each field
analysis.add_task(d3.Average(0.5*(u1*u1 + v1*v1), ('x', 'y')), layout='g',name='<E1>')

# Power injection
analysis.add_task(d3.Average(u1*bforcex+v1*bforcey, ('x', 'y')), layout='g',name='<I>')
# Dissipation
analysis.add_task(d3.Average(coef_s*(dx(u1)**2 + dy(v1)**2 + 0.5*(dx(v1)+dy(u1))**2), ('x', 'y')), layout='g',name='<D>')

# Mean velocities in each field
analysis.add_task(d3.Average(u1, ('x', 'y')), layout='g',name='<u1>')
analysis.add_task(d3.Average(v1, ('x', 'y')), layout='g',name='<v1>')

# Mean cij
analysis.add_task(d3.Average(cxx1, ('x', 'y')), layout='g',name='<cxx>')
analysis.add_task(d3.Average(cxy1, ('x', 'y')), layout='g',name='<cxy>')
analysis.add_task(d3.Average(cyy1, ('x', 'y')), layout='g',name='<cyy>')


# Flow properties - primarily for logger ==========================================================
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u1)**2, name='u1sq')
flow.add_property((v1)**2, name='v1sq')
flow.add_property(0.5*(u1)*(u1)+0.5*(v1)*(v1),name='E')
# =================================================================================================

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:        
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u1sq'))
            max_v = np.sqrt(flow.max('v1sq'))
            max_dE = flow.max('E')
            #favg = d3.Average(u*u + v*v, ('x', 'y'))
            #meanvel = np.sqrt(favg.evaluate()['g'])
            #print('meanvel:',meanvel)
            logger.info('Time=%e, dt=%e, max(u)=%f, max(v)=%f, max(E)=%e' %(solver.sim_time, timestep, max_u,max_v,max_dE))         
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
