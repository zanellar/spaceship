import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import casadi as ca

# Import do_mpc package:
import do_mpc

def get_model_parameters():
    return dict(
        m = 1.0,
        j11 = 1.0,
        j22 = 1.0,
        j33 = 1.0,
    )

def get_policy_val(t_now):
    return dict(
        f1 = np.array([0,0,0]),        # TODO here we can define the policy as a function of time
        tau1 = np.array([0,0,0]),
    )

###################################################

def sim_p_fun(t_now): 
    ''' Return the values of the model parameters '''
    param = get_model_parameters()
    p_template['m'] = param['m']
    p_template['j11'] = param['j11']
    p_template['j22'] = param['j22']
    p_template['j33'] = param['j33']
    return p_template 

def sim_tvp_fun(t_now):
    ''' Return the values of the force and torque (as time-varying parameters) at the current time step '''
    tvp_template['f1'] = get_policy_val(t_now)['f1']
    tvp_template['tau1'] = get_policy_val(t_now)['tau1']
    return tvp_template 

def mpc_p_fun(t_now): 
    ''' Return the values of the model parameters '''
    param = get_model_parameters()
    p_template["_p"] = [param['m'], param['j11'], param['j22'], param['j33']]
    return p_template 

def mpc_tvp_fun(t_now):
    ''' Return the values of the force and torque (as time-varying parameters) at the current time step '''
    for t in range(len(tvp_template['_tvp'])): 
        tvp_template['_tvp',t,'f1'] = get_policy_val(t+t_now)['f1']
        tvp_template['_tvp',t,'tau1'] = get_policy_val(t+t_now)['tau1']
    return tvp_template 


################ Model definition ################

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

### Variables ###

# State variables
xp = model.set_variable(var_type='_x', var_name='xp', shape=(3,1)) # Position (p)
xv = model.set_variable(var_type='_x', var_name='xv', shape=(3,1)) # Velocity (dp)
xn = model.set_variable(var_type='_x', var_name='xn', shape=(3,1)) # Attitude vector (n)
xw = model.set_variable(var_type='_x', var_name='xw', shape=(3,1)) # Angular velocity (omega)

# State derivatives 
dxp = model.set_variable(var_type='_x', var_name='dxp', shape=(3,1))
dxv = model.set_variable(var_type='_x', var_name='dxv', shape=(3,1))
dxn = model.set_variable(var_type='_x', var_name='dxn', shape=(3,1))
dxw = model.set_variable(var_type='_x', var_name='dxw', shape=(3,1)) 

# Control variables (diagonal of inertia matrix)
uj11 = model.set_variable(var_type='_u', var_name='uj11', shape=(1,1))
uj22 = model.set_variable(var_type='_u', var_name='uj22', shape=(1,1))
uj33 = model.set_variable(var_type='_u', var_name='uj33', shape=(1,1))

# Policies
f1 = model.set_variable(var_type='_tvp', var_name='f1', shape=(3,1))
tau1 = model.set_variable(var_type='_tvp', var_name='tau1', shape=(3,1))

# Parameters
j11 = model.set_variable(var_type='_p', var_name='j11', shape=(1,1)) 
j22 = model.set_variable(var_type='_p', var_name='j22', shape=(1,1)) 
j33 = model.set_variable(var_type='_p', var_name='j33', shape=(1,1)) 
m = model.set_variable(var_type='_p', var_name='m', shape=(1,1))

print(model.x.labels()) 
  
#### Equation ####
model.set_rhs('xp', dxp) 
model.set_rhs('xv', dxv)
model.set_rhs('xn', dxn)
model.set_rhs('xw', dxw)

# Nonlinear ODE 
uJ = ca.diag(ca.vertcat(uj11, uj22, uj33))
J = ca.diag(ca.vertcat(j11, j22, j33))

eq_dxp = xv
eq_dxv = f1/m 
eq_dxn = ca.mtimes(xn.T, ca.skew(xw)).T  
eq_dxw = ca.mtimes(ca.inv(J), (- ca.mtimes(ca.skew(xw), ca.mtimes(J, xw)) - ca.mtimes(ca.skew(xw), ca.mtimes(uJ, xw)) + tau1))

model.set_rhs('dxp', eq_dxp) 
model.set_rhs('dxv', eq_dxv)
model.set_rhs('dxn', eq_dxn)
model.set_rhs('dxw', eq_dxw) 

model.setup()

################ Controller ################

# Configuring the MPC controller
mpc = do_mpc.controller.MPC(model)

# Optimizer parameters
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# Parameters 
p_template = mpc.get_p_template(n_combinations=1) # TODO: n_combinations 
mpc.set_p_fun(mpc_p_fun)

# Force and torque policies   
tvp_template = mpc.get_tvp_template()    
mpc.set_tvp_fun(mpc_tvp_fun)

### Objective function ###

# Real torque that is applied to the system
J = ca.diag(ca.vertcat(j11, j22, j33))
tau = ca.mtimes(ca.skew(xw), ca.mtimes(J, xw)) + tau1

# Lagrange term 
mterm = ca.mtimes(tau.T, tau)

# Meyer term
lterm = ca.mtimes(tau.T, tau)

 
# Weights of diagonal elements of R matrix  
mpc.set_rterm(
    uj11=1e-2,
    uj22=1e-2,
    uj33=1e-2
)

mpc.set_objective(mterm=mterm, lterm=lterm)

### Constraints ###

# Lower bounds on states:
# TODO
# Upper bounds on states
# TODO
 
# Lower bounds on inputs:
mpc.bounds['lower','_u', 'uj11'] = -10
mpc.bounds['lower','_u', 'uj22'] = -10
mpc.bounds['lower','_u', 'uj33'] = -10
# Lower bounds on inputs:
mpc.bounds['upper','_u', 'uj11'] = 10
mpc.bounds['upper','_u', 'uj22'] = 10
mpc.bounds['upper','_u', 'uj33'] = 10

# Scaling
# TODO
 
# Uncertain Parameters 
# TODO

mpc.setup()

################ Simulator ################

simulator = do_mpc.simulator.Simulator(model) 
simulator.set_param(t_step = 0.1)

# Parameters 
p_template = simulator.get_p_template() # TODO: n_combinations 
simulator.set_p_fun(sim_p_fun)

# Force and torque policies   
tvp_template = simulator.get_tvp_template()     
simulator.set_tvp_fun(sim_tvp_fun)


simulator.setup()

################ Control loop ################
# Initial state x0 = (xp,xv,xn,xw,dxp,dxv,dxn,dxw) -> shape (24,1)
xp0 = np.array([0,0,0])
xv0 = np.array([0,0,0])
xn0 = np.array([0,0,1])
xw0 = np.array([0,0,0])
x0 = np.concatenate((xp0,xv0,xn0,xw0)).reshape(-1,1)
x0 = np.concatenate((x0,np.zeros_like(x0))).reshape(-1,1) 
simulator.x0 = x0
mpc.x0 = x0 
mpc.set_initial_guess()

# Setting up the Graphic
import matplotlib  
import matplotlib.pyplot as plt 

# Customizing Matplotlib:
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)
 
# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig = plt.figure(figsize=(16,9))
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot state
    g.add_line(var_type='_x', var_name='xp', axis=ax1)
    g.add_line(var_type='_x', var_name='xv', axis=ax1)
    g.add_line(var_type='_x', var_name='xn', axis=ax1)
    g.add_line(var_type='_x', var_name='xw', axis=ax1) 
    # Plot inputs
    g.add_line(var_type='_u', var_name='uj11', axis=ax2)
    g.add_line(var_type='_u', var_name='uj22', axis=ax2)
    g.add_line(var_type='_u', var_name='uj33', axis=ax2)


ax1.set_ylabel('State p, v, n, w')
ax2.set_ylabel('Inertia uJ')
ax2.set_xlabel('time [s]')

# Simulate in closed loop

u0 = np.zeros((3,1))
for i in range(200):
    simulator.make_step(u0)

sim_graphics.plot_results()
sim_graphics.reset_axes()

# plt.show()

u0 = mpc.make_step(x0)

sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()

plt.show()