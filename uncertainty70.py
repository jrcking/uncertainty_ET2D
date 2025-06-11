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
Sc=unc_params.Sc
Deltat0=unc_params.Deltat0
ptbnmag=unc_params.ptbnmag


# snapshot output period
dtau=1.0

# Forcing magnitude
f0 = 4*np.pi*np.pi/Re

# Perturbation controls
t0=290;      # perturbation time
#t1 = round(1 + t0/5);
#loadname='./checkpoints_pre/checkpoints_pre_s7.h5'
loadname ="./checkpoints_pre/checkpoints_pre_s"+("{:d}".format(round(0.2*(t0) + 1)))+".h5"


# Computational bits
dealias = 3/2
stop_sim_time = t0+50*dtau
timestepper = d3.RK222
dtype = np.float64

# Substitutions for coefficients
coef_s = beta/Re # Solvent viscous coefficient
coef_p = (1.0-beta)/(Re*Wi) # Polymeric coefficient
ooSc = 1.0/Sc
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

# Fields for realisation 2
u2 = dist.Field(name='u2', bases=(xbasis,ybasis))
v2 = dist.Field(name='v2', bases=(xbasis,ybasis))
p2 = dist.Field(name='p2', bases=(xbasis,ybasis))
cxx2 = dist.Field(name='cxx2', bases=(xbasis,ybasis))
cxy2 = dist.Field(name='cxy2', bases=(xbasis,ybasis))
cyy2 = dist.Field(name='cyy2', bases=(xbasis,ybasis))
tau_p2 = dist.Field(name='tau_p2')  # Penalty term for incompressibility

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

# Perturbation
ft=-ptbnmag*(0+np.exp(-((t-t0)/Deltat0)**2))  #time dependent function used to impose perturbation on realisation 2

# Problem definition ==============================================================================
problem = d3.IVP([u1,v1,cxx1,cxy1,cyy1,p1,tau_p1,u2,v2,cxx2,cxy2,cyy2,p2,tau_p2], time=t, namespace=locals())

# Realisation 1 -----------------------------------------------------------------------------------
problem.add_equation("dt(u1) + dx(p1) - coef_s*lap(u1) - coef_p*(dx(cxx1)+dy(cxy1)) = \
        -u1*dx(u1) - v1*dy(u1) + bforcex")
problem.add_equation("dt(v1) + dy(p1) - coef_s*lap(v1) - coef_p*(dx(cxy1)+dy(cyy1)) = \
        -u1*dx(v1) - v1*dy(v1) + bforcey")                
problem.add_equation("dt(cxx1) - ooSc*lap(cxx1) = - u1*dx(cxx1) - v1*dy(cxx1) + 2.0*cxx1*dx(u1) + 2.0*cxy1*dy(u1) \
        - (cxx1-1.0)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi")
problem.add_equation("dt(cxy1) - ooSc*lap(cxy1) = - u1*dx(cxy1) - v1*dy(cxy1) + cxx1*dx(v1) + cyy1*dy(u1) \
        + cxy1*(dx(u1)+dy(v1)) - (cxy1)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi")
problem.add_equation("dt(cyy1) - ooSc*lap(cyy1) = - u1*dx(cyy1) - v1*dy(cyy1) + 2.0*cxy1*dx(v1) + 2.0*cyy1*dy(v1) \
        - (cyy1-1.0)*(1.0-2.0*eps+eps*(cxx1+cyy1))*ooWi") 

problem.add_equation("dx(u1) + dy(v1) + tau_p1 = 0")  # Divergence free condition
problem.add_equation("integ(p1) = 0") # Mean pressure =0

# Realisation 2 -----------------------------------------------------------------------------------
problem.add_equation("dt(u2) + dx(p2) - coef_s*lap(u2) - coef_p*(dx(cxx2)+dy(cxy2)) = \
        -u2*dx(u2) - v2*dy(u2) + bforcex")
problem.add_equation("dt(v2) + dy(p2) - coef_s*lap(v2) - coef_p*(dx(cxy2)+dy(cyy2)) = \
        -u2*dx(v2) - v2*dy(v2) + bforcey")                
problem.add_equation("dt(cxx2) - ooSc*lap(cxx2) = - u2*dx(cxx2) - v2*dy(cxx2) + 2.0*cxx2*dx(u2) + 2.0*cxy2*dy(u2) \
        - (cxx2-1.0)*(1.0-2.0*eps+eps*(cxx2+cyy2))*ooWi - lap(lap(cxx2))*ft")
problem.add_equation("dt(cxy2) - ooSc*lap(cxy2) = - u2*dx(cxy2) - v2*dy(cxy2) + cxx2*dx(v2) + cyy2*dy(u2) \
        + cxy2*(dx(u2)+dy(v2)) - (cxy2)*(1.0-2.0*eps+eps*(cxx2+cyy2))*ooWi - lap(lap(cxy2))*ft")
problem.add_equation("dt(cyy2) - ooSc*lap(cyy2) = - u2*dx(cyy2) - v2*dy(cyy2) + 2.0*cxy2*dx(v2) + 2.0*cyy2*dy(v2) \
        - (cyy2-1.0)*(1.0-2.0*eps+eps*(cxx2+cyy2))*ooWi - lap(lap(cyy2))*ft") 
                                  
problem.add_equation("dx(u2) + dy(v2) + tau_p2 = 0")  # Divergence free condition
problem.add_equation("integ(p2) = 0") # Mean pressure =0
# =================================================================================================

# Solver ==========================================================================================
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
# =================================================================================================

# Initial conditions ==============================================================================
if not restart:
    file_handler_mode = 'overwrite'
    u1['g'] = 0.0 #+ 0.0*np.sin(n*y*k0) + 0.2*np.cos(y*k0+1.1)
    v1['g'] = 0.0 #+ 0.0*np.sin(n*x*k0) + 0.4*np.sin(x*k0+3.9)
    cxx1['g'] = 1.0 #+ 0.25*np.cos(y*k0+1.1)
    cxy1['g'] = 0.0
    cyy1['g'] = 1.0 #+ 0.34*(np.sin(x*k0+2.54)*np.cos(y*k0+3.2))
    u2['g'] = 0.0 #+ 0.0*np.sin(n*y*k0) + 0.2*np.cos(y*k0+1.1)
    v2['g'] = 0.0 #+ 0.0*np.sin(n*x*k0) + 0.4*np.sin(x*k0+3.9)
    cxx2['g'] = 1.0 #+ 0.25*np.cos(y*k0+1.1)
    cxy2['g'] = 0.0
    cyy2['g'] = 1.0 #+ 0.34*(np.sin(x*k0+2.54)*np.cos(y*k0+3.2))
else:
    write, initial_timestep = solver.load_state(loadname)
    initial_timestep = timestep
    file_handler_mode = 'append'



# =================================================================================================

# Snapshots output ================================================================================
# Fields for visualisation
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=dtau, max_writes=10)
# Field 1
snapshots.add_task(u1, name='u1')
snapshots.add_task(v1, name='v1')
snapshots.add_task(cxx1, name='cxx1')
snapshots.add_task(cxy1, name='cxy1')
snapshots.add_task(cyy1, name='cyy1')
snapshots.add_task(dx(u1),name='du1dx')
snapshots.add_task(dx(v1),name='dv1dx')
snapshots.add_task(dy(u1),name='du1dy')
snapshots.add_task(dy(v1),name='dv1dy')
# Field 2
snapshots.add_task(u2, name='u2')
snapshots.add_task(v2, name='v2')
snapshots.add_task(cxx2, name='cxx2')
snapshots.add_task(cxy2, name='cxy2')
snapshots.add_task(cyy2, name='cyy2')
snapshots.add_task(dx(u2),name='du2dx')
snapshots.add_task(dx(v2),name='dv2dx')
snapshots.add_task(dy(u2),name='du2dy')
snapshots.add_task(dy(v2),name='dv2dy')

# Terms in KE evolution
snapshots.add_task(-coef_p*((cxx2-cxx1-cyy2+cyy1)*dx(u2-u1)+(cxy2-cxy1)*(dy(u2-u1)+dx(v2-v1))),name='dPS')
snapshots.add_task(coef_s*(dx(u2-u1)**2 + dy(u2-u1)**2 + dx(v2-v1)**2 + dy(v2-v1)**2),name='dVD')
snapshots.add_task(-(u2-u1)*(u2-u1)*dx(u1)-(v2-v1)*(v2-v1)*dy(v1)-(u2-u1)*(v2-v1)*(dx(v1)+dy(u1)),name='dIP')

# =================================================================================================
# Checkpoints - necessary for restarts!
#checkpoints_main = solver.evaluator.add_file_handler('checkpoints_main', sim_dt=1, max_writes=1, mode=file_handler_mode)
#checkpoints_main.add_tasks(solver.state)


# Flow properties - primarily for logger ==========================================================
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u1)**2, name='u1sq')
flow.add_property((v1)**2, name='v1sq')
flow.add_property((u2)**2, name='u2sq')
flow.add_property((v2)**2, name='v2sq')
flow.add_property(0.5*(u1-u2)*(u1-u2)+0.5*(v1-v2)*(v1-v2),name='dE')
# =================================================================================================

# Statistical analysis ============================================================================
# Mostly volume averaged quantities to explore evolution of uncertainty
analysis = solver.evaluator.add_file_handler('analysis', sim_dt=0.001, max_writes=400000)

# Kinetic energy of each field
analysis.add_task(d3.Average(0.5*(u1*u1 + v1*v1), ('x', 'y')), layout='g',name='<E1>')
analysis.add_task(d3.Average(0.5*(u2*u2 + v2*v2), ('x', 'y')), layout='g',name='<E2>')

# Mean velocities in each field
analysis.add_task(d3.Average(u1, ('x', 'y')), layout='g',name='<u1>')
analysis.add_task(d3.Average(u2, ('x', 'y')), layout='g',name='<u2>')
analysis.add_task(d3.Average(v1, ('x', 'y')), layout='g',name='<v1>')
analysis.add_task(d3.Average(v2, ('x', 'y')), layout='g',name='<v2>')


# Mean differences in cij
analysis.add_task(d3.Average(cxx2-cxx1, ('x', 'y')), layout='g',name='<dcxx>')
analysis.add_task(d3.Average(cxy2-cxy1, ('x', 'y')), layout='g',name='<dcxy>')
analysis.add_task(d3.Average(cyy2-cyy1, ('x', 'y')), layout='g',name='<dcyy>')

# Uncertainty KE and terms therein.
analysis.add_task(d3.Average(0.5*(u2-u1)*(u2-u1) + 0.5*(v2-v1)*(v2-v1), ('x', 'y')), layout='g',name='<dE>') # E_Delta
analysis.add_task(d3.Average(-(u2-u1)*(u2-u1)*dx(u1)-(v2-v1)*(v2-v1)*dy(v1)-(u2-u1)*(v2-v1)*(dx(v1)+dy(u1)), \
                             ('x', 'y')), layout='g',name='<dIP>') # Inertial production term
analysis.add_task(d3.Average(coef_s*(dx(u2-u1)**2+dy(u2-u1)**2+dx(v2-v1)**2+dy(v2-v1)**2),\
                             ('x', 'y')), layout='g',name='<dVD>') # Viscous dissipation term
analysis.add_task(d3.Average(-coef_p*((cxx2-cxx1)*dx(u2-u1)+(cxy2-cxy1)*(dy(u2-u1)+dx(v2-v1))+(cyy2-cyy1)*dy(v2-v1)),\
                             ('x', 'y')), layout='g',name='<dPS>') # Polymeric stress production term

#Uncertainty in Elastic energy and terms therein
analysis.add_task(d3.Average(cxx2+cyy2-cxx1-cyy1, ('x', 'y')), layout='g',name='<dtrC>') # Delta(tr(c))
analysis.add_task(d3.Average(2*(dx(u1)*(cxx2-cxx1) + (dy(u1)+dx(v1))*(cxy2-cxy1) + dy(v1)*(cyy2-cyy1)), \
                             ('x', 'y')), layout='g',name='<dUC1>') # grad(u1)*deltac
analysis.add_task(d3.Average(2*(dx(u2-u1)*(cxx1) + (dy(u2-u1)+dx(v2-v1))*(cxy1) + dy(v2-v1)*(cyy1)), \
                             ('x', 'y')), layout='g',name='<dUC2>') # grad(deltau)*c1
analysis.add_task(d3.Average(2*(dx(u2-u1)*(cxx2-cxx1) + (dy(u2-u1)+dx(v2-v1))*(cxy2-cxy1) + dy(v2-v1)*(cyy2-cyy1)), \
                             ('x', 'y')), layout='g',name='<dUC3>') # grad(deltau)*deltac
analysis.add_task(d3.Average((cxx2+cyy2-cxx1-cyy1)*(1+2*eps*(cxx1+cyy1)+eps*(cxx2+cyy2-cxx1-cyy1))/Wi, \
                             ('x', 'y')), layout='g',name='<dCR>') # Relaxation terms
analysis.add_task(d3.Average((cxx2-cxx1)**2 + (cyy2-cyy1)**2 + 2.0*(cxy2-cxy1)**2, ('x', 'y')), layout='g',name='<dCdC>') 

#Uncertainty in (my notation) Gamma_Delta (or dG)
analysis.add_task(d3.Average((cxx2+cyy2-cxx1-cyy1)**2,('x', 'y')), layout='g',name='<dG>') # (Delta(tr(c)))^{2}
analysis.add_task(d3.Average(2*(cxx1+cyy1)*(dx(cxx2+cyy2-cxx1-cyy1)*(u2-u1)+dy(cxx2+cyy2-cxx1-cyy1)*(v2-v1)),('x', 'y')), layout='g',name='<dG_ad>') # Advective term in Gamma_Delta eqn
analysis.add_task(d3.Average(4*(cxx2+cyy2-cxx1-cyy1)*(dx(u1)*(cxx2-cxx1) + (dy(u1)+dx(v1))*(cxy2-cxy1) + dy(v1)*(cyy2-cyy1)),('x', 'y')), layout='g',name='<dG_UC1>') # UC1 in Gamma_Delta eqn
analysis.add_task(d3.Average(4*(cxx2+cyy2-cxx1-cyy1)*(dx(u2-u1)*(cxx1) + (dy(u2-u1)+dx(v2-v1))*(cxy1) + dy(v2-v1)*(cyy1)),('x', 'y')), layout='g',name='<dG_UC2>') # UC2 in Gamma_Delta eqn
analysis.add_task(d3.Average(4*(cxx2+cyy2-cxx1-cyy1)*(dx(u2-u1)*(cxx2-cxx1) + (dy(u2-u1)+dx(v2-v1))*(cxy2-cxy1) + dy(v2-v1)*(cyy2-cyy1)),('x', 'y')), layout='g',name='<dG_UC3>') # UC3 in Gamma_Delta eqn
analysis.add_task(d3.Average((2.0/Wi)*(1+2*eps*(cxx1+cyy1)+eps*(cxx2+cyy2-cxx1-cyy1))*(cxx2+cyy2-cxx1-cyy1)**2,('x', 'y')), layout='g',name='<dG_R>') # SOURCE in Gamma_Delta eqn
analysis.add_task(d3.Average((2.0*ooSc)*((dx(cxx2+cyy2-cxx1-cyy1))**2 + (dy(cxx2+cyy2-cxx1-cyy1))**2),('x', 'y')), layout='g',name='<dG_D>') # DIFFUSION in Gamma_Delta eqn

# =================================================================================================
# Main loop 
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = max(np.sqrt(flow.max('u1sq')),np.sqrt(flow.max('u2sq')))
            max_v = max(np.sqrt(flow.max('v1sq')),np.sqrt(flow.max('v2sq')))
            max_dE = flow.max('dE')
            #favg = d3.Average(u*u + v*v, ('x', 'y'))
            #meanvel = np.sqrt(favg.evaluate()['g'])
            #print('meanvel:',meanvel)
            logger.info('Iteration=%i, Time=%e, dt=%e, max(u)=%f, max(v)=%f, max(dE)=%e' %(solver.iteration, solver.sim_time, timestep, max_u,max_v,max_dE))    
                        
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
