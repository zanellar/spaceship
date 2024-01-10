import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import casadi as ca

# Import do_mpc package:
import do_mpc

################ Model definition ################

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

### Variables ###

# Variables can be scalars:
phi_1 = model.set_variable(var_type='_x', var_name='phi_1', shape=(1,1))
phi_2 = model.set_variable(var_type='_x', var_name='phi_2', shape=(1,1))
phi_3 = model.set_variable(var_type='_x', var_name='phi_3', shape=(1,1))
# Variables can also be vectors:
dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))
# Two states for the desired (set) motor position:
phi_m_1_set = model.set_variable(var_type='_u', var_name='phi_m_1_set')
phi_m_2_set = model.set_variable(var_type='_u', var_name='phi_m_2_set')
# Two additional states for the true motor position:
phi_1_m = model.set_variable(var_type='_x', var_name='phi_1_m', shape=(1,1))
phi_2_m = model.set_variable(var_type='_x', var_name='phi_2_m', shape=(1,1))

Theta_1 = model.set_variable('parameter', 'Theta_1') 
Theta_2 = model.set_variable('parameter', 'Theta_2')
Theta_3 = model.set_variable('parameter', 'Theta_3')

print(model.x.labels()) 

#### Parameters ####
  
c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
d = np.array([6.78,  8.01,  8.82])*1e-5

tau = 1e-2  
 
#### Equation ####
model.set_rhs('phi_1', dphi[0])
model.set_rhs('phi_2', dphi[1])
model.set_rhs('phi_3', dphi[2])

dphi_next = ca.vertcat(
    -c[0]/Theta_1*(phi_1-phi_1_m)-c[1]/Theta_1*(phi_1-phi_2)-d[0]/Theta_1*dphi[0],
    -c[1]/Theta_2*(phi_2-phi_1)-c[2]/Theta_2*(phi_2-phi_3)-d[1]/Theta_2*dphi[1],
    -c[2]/Theta_3*(phi_3-phi_2)-c[3]/Theta_3*(phi_3-phi_2_m)-d[2]/Theta_3*dphi[2],
)

model.set_rhs('dphi', dphi_next) 
model.set_rhs('phi_1_m', 1/tau*(phi_m_1_set - phi_1_m))
model.set_rhs('phi_2_m', 1/tau*(phi_m_2_set - phi_2_m))

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

### Objective function ###
mterm = phi_1**2 + phi_2**2 + phi_3**2
lterm = phi_1**2 + phi_2**2 + phi_3**2

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(
    phi_m_1_set=1e-2,
    phi_m_2_set=1e-2
)

### Constraints ###

# Lower bounds on states:
mpc.bounds['lower','_x', 'phi_1'] = -2*np.pi
mpc.bounds['lower','_x', 'phi_2'] = -2*np.pi
mpc.bounds['lower','_x', 'phi_3'] = -2*np.pi
# Upper bounds on states
mpc.bounds['upper','_x', 'phi_1'] = 2*np.pi
mpc.bounds['upper','_x', 'phi_2'] = 2*np.pi
mpc.bounds['upper','_x', 'phi_3'] = 2*np.pi

# Lower bounds on inputs:
mpc.bounds['lower','_u', 'phi_m_1_set'] = -2*np.pi
mpc.bounds['lower','_u', 'phi_m_2_set'] = -2*np.pi
# Lower bounds on inputs:
mpc.bounds['upper','_u', 'phi_m_1_set'] = 2*np.pi
mpc.bounds['upper','_u', 'phi_m_2_set'] = 2*np.pi

# Scaling
mpc.scaling['_x', 'phi_1'] = 2
mpc.scaling['_x', 'phi_2'] = 2
mpc.scaling['_x', 'phi_3'] = 2

# Uncertain Parameters 
inertia_mass_1 = 2.25*1e-4*np.array([1., 0.9, 1.1])
inertia_mass_2 = 2.25*1e-4*np.array([1., 0.9, 1.1])
inertia_mass_3 = 2.25*1e-4*np.array([1.])

mpc.set_uncertainty_values(
    Theta_1 = inertia_mass_1,
    Theta_2 = inertia_mass_2,
    Theta_3 = inertia_mass_3
)

mpc.setup()

################ Simulator ################

simulator = do_mpc.simulator.Simulator(model) 
simulator.set_param(t_step = 0.1)

# Uncertain Parameters 
p_template = simulator.get_p_template()
print("p_template: ",p_template.keys()) # dict_keys(['Theta_1', 'Theta_2', 'Theta_3'])
def p_fun(t_now):
    p_template['Theta_1'] = 2.25e-4
    p_template['Theta_2'] = 2.25e-4
    p_template['Theta_3'] = 2.25e-4
    return p_template 
simulator.set_p_fun(p_fun)

simulator.setup()

################ Control loop ################
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0 
mpc.set_initial_guess()

################ Plotting ################
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
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='phi_1', axis=ax1)
    g.add_line(var_type='_x', var_name='phi_2', axis=ax1)
    g.add_line(var_type='_x', var_name='phi_3', axis=ax1)

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='phi_m_1_set', axis=ax2)
    g.add_line(var_type='_u', var_name='phi_m_2_set', axis=ax2)


ax1.set_ylabel('angle position [rad]')
ax2.set_ylabel('motor angle [rad]')
ax2.set_xlabel('time [s]')

u0 = np.zeros((2,1))
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