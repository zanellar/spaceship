import json
import os
import time 
import numpy as np
import matplotlib.pyplot as plt  

from spaceship.model import create_model
from spaceship.mpc import create_mpc
from spaceship.sim import create_simulator
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion
from spaceship.visual.draw3d import Drone3DStopMotion

from spaceship.visual.draw3d import Drone3DStopMotion 

from spaceship.policies import *
from spaceship.utils.paths import PARAMS_PATH, PLOTS_PATH

# Choose parameter file
model_param_file = 'model_test'
mpc_param_file = 'mpc_test'
env_param_file = 'env_test'
sim_param_file = 'sim_test'
sys_param_file = 'sys_test'

# Import parameters 
with open(os.path.join(PARAMS_PATH, model_param_file+".json")) as json_file:
    modelparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, mpc_param_file+".json")) as json_file:
    mpcparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)

with open(os.path.join(PARAMS_PATH, sim_param_file+".json")) as json_file:
    simparams = json.load(json_file)

with open(os.path.join(PARAMS_PATH, sys_param_file+".json")) as json_file:
    systemparams = json.load(json_file)
 
####################################################################  
####################################################################  
####################################################################  

# ------- ADDITIONAL CONTROLLERS ---------

pos_control = globals()[simparams["params_ctrl_pos"]["policy_func"]](target=list(simparams["target_pos"]), params=dict(simparams["params_ctrl_pos"]))
rot_control = globals()[simparams["params_ctrl_rot"]["policy_func"]](target=list(simparams["target_rot"]), params=dict(simparams["params_ctrl_rot"]))

if simparams["disable_nominal_control"]:
    pos_control = NullControl()
    rot_control = NullControl()

# ------------ SYSTEM ------------

system = create_model(
    systemparams, 
    dt=simparams["t_step"], 
    settings=dict(
        ctr_input=simparams["ctr_input_type"], 
        disable_mpc=simparams["disable_mpc"],
        disable_nominal_control=simparams["disable_nominal_control"] 
        ), 
    external_functions=dict(
        f_nominal=pos_control, 
        tau_nominal=rot_control
        )
    )

# ------------ SIMULATOR ------------
simulator = create_simulator(
    system, 
    modelparams, 
    simparams
    )

# ------------ MODEL ------------
model = create_model(
    modelparams, 
    dt=mpcparams["t_step"], 
    settings=dict(
        ctr_input=simparams["ctr_input_type"], 
        disable_mpc=simparams["disable_mpc"],
        disable_nominal_control=simparams["disable_nominal_control"] 
        ), 
    external_functions=dict(
        f_nominal=pos_control, 
        tau_nominal=rot_control
        )
    )

# ------------ MPC ------------
mpc = create_mpc(
    model,  
    modelparams, 
    mpcparams, 
    envparams
    )


####################################################################  
####################################################################  
####################################################################  
 
# Set initial state 
simulator.x0['xp'] = list(simparams["initial_position"])
simulator.x0['xv'] = list(simparams["initial_velocity"])
simulator.x0['xr1'] = list(simparams["initial_orientation_axis1"])
simulator.x0['xr2'] = list(simparams["initial_orientation_axis2"])
simulator.x0['xn'] = list(simparams["initial_orientation_axis3"])
simulator.x0['xw'] = list(simparams["initial_angular_velocity"]) 

x = simulator.x0.cat.full()
mpc.x0 = x 
mpc.set_initial_guess() 

####################################################################  

# Run MPC main loop 
orientations = []
positions = []
slit_orientation_errors = []
slit_distances = []
total_energy = []
kinetic_energy = []
potential_energy = []
zero_work_term = []
torques = []
constraints = []
opt_times = [] 
itr_times = []
target_orientation_errors = []
target_position_errors = []

start_sim_time = time.time()

for k in range(simparams["n_steps"]):  

    print(f"step {k}")
    _start_opt_time = time.time()

    # Solve the optimization problem
    u = mpc.make_step(x)

    _end_opt_time = time.time()

    # Simulate one step of the system dynamics
    x = simulator.make_step(u) 

    _end_itr_time = time.time() 

    # Execution times 
    opt_times.append(_end_opt_time - _start_opt_time)
    itr_times.append(_end_itr_time - _end_opt_time)

    # states
    xp = x[0:3]
    xv = x[3:6]
    xr1 = x[6:9]
    xr2 = x[9:12]
    xn = x[12:15]
    xw = x[15:18] 

    J = np.array([[modelparams["j11"],0,0],[0,modelparams["j22"],0],[0,0,modelparams["j33"]]]) 
    m = modelparams["m"]
 
    # Energy
    T = 0.5*m*np.dot(xv.T,xv) + 0.5*xw.T @ J @ xw 
    U = rot_control.energy(x) #+ pos_control.energy(x)
    E = T + U
    total_energy.append(E)
    kinetic_energy.append(T)
    potential_energy.append(U)

    # Torques
    torques.append(np.cross(xw.flatten(), (np.diag(u.flatten()) @ xw).flatten()))

    # Zero work term
    zero_work_term.append( np.dot(xw.T, np.cross(xw.flatten(), (J @ xw).flatten())))
 
    # Constraints
    constraints.append((1 - np.dot(xn.flatten(), envparams["xnd"])**2) / (np.linalg.norm(xp[0:2] - envparams["xpd"][0:2])**2 + 1/mpcparams["pos_weight"]))
    
    # Position and orientation 
    positions.append(xp)
    orientations.append(xn) 

    # Orientation error and distance wrt to the target slit
    slit_orientation_errors.append(1-np.linalg.norm(np.dot(xn.T,np.array(envparams["xnd"]))))
    slit_distances.append(np.linalg.norm(xp[0:2]-np.array(envparams["xpd"]).reshape((3,1))[0:2]))
    
    # Target orientation and position errors
    target_orientation_errors.append(0.5*np.trace(np.eye(3) - np.array(simparams["target_rot"]).T @ np.column_stack((xr1, xr2, xn))))
    target_position_errors.append(np.linalg.norm(np.array(simparams["target_pos"]) - np.array(xp)))

end_sim_time = time.time()

####################################################################  
####################################################################  
####################################################################  

# Plot the trajectories# Plot the trajectories
fig1, ax1 = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(8, 8))
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig4, ax4 = plt.subplots(figsize=(8, 8))
fig5, ax5 = plt.subplots(figsize=(8, 8))
fig6, ax6 = plt.subplots(figsize=(8, 8))
fig7, ax7 = plt.subplots(figsize=(8, 8))
fig8, ax8 = plt.subplots(figsize=(8, 8))

positions = np.array(positions).reshape((simparams["n_steps"],3))
orientations = np.array(orientations).reshape((simparams["n_steps"],3)) 
slit_orientation_errors = np.array(slit_orientation_errors).flatten()
total_energy = np.array(total_energy).flatten()
kinetic_energy = np.array(kinetic_energy).flatten()
potential_energy = np.array(potential_energy).flatten()
  
# Plot distance of the drone to the target slit and the orientation error:
# 1) Plot the distance & error vs time
time_array = np.arange(simparams["n_steps"])   # Create the time array
ax1.plot(time_array, slit_distances, label='slit_distances')
ax1.plot(time_array, slit_orientation_errors, label='slit_orientation_errors')
ax1.set_xlabel('time')
ax1.set_ylabel('distance')
ax1.set_title('Distance and error vs time')
ax1.legend()

# 2) Plot the distance vs error
ax2.plot(slit_distances, slit_orientation_errors ) 
ax2.set_xlabel('distance')
ax2.set_ylabel('error')
ax2.set_title('Error vs distance')
ax2.legend()

# Plot total_energy vs time
ax3.plot(time_array, total_energy, label='total_energy' )
ax3.plot(time_array, kinetic_energy, label='kinetic_energy')
ax3.plot(time_array, potential_energy, label='potential_energy')
ax3.set_xlabel('time')
ax3.set_ylabel('total_energy')
ax3.set_title('Energy vs time')
ax3.legend() 

# Plot zero work term vs time
ax4.plot(time_array, zero_work_term )
ax4.set_xlabel('time')
ax4.set_ylabel('zero work term')
ax4.set_title('Zero work term vs time')
ax4.legend()

# plot the torque vs time
torques = np.array(torques).reshape((simparams["n_steps"],3))
ax5.plot(time_array, torques ) 
# ax5.scatter(time_array, torques[:,0], label='torque1')
ax5.set_xlabel('time')
ax5.set_ylabel('torques')
ax5.set_title('Torques vs time')
ax5.legend()

# plot the constraints vs time
ax6.plot(time_array, constraints )
ax6.set_xlabel('time')
ax6.set_ylabel('constraints')
ax6.set_title('Constraints vs time')
ax6.legend()
 
# plot the optimization and iteration times
ax7.plot(time_array, opt_times, label='optimization')
ax7.plot(time_array, itr_times, label='simulation step') 
ax7.set_xlabel('step')
ax7.set_ylabel('time [s]')

# plot the orientation and position slit_orientation_errors
ax8.plot(time_array, target_orientation_errors, label='orientation error')
# ax8.plot(time_array, target_position_errors, label='position error')
ax8.set_xlabel('step')
ax8.set_ylabel('error')
ax8.set_title('Orientation and position slit_orientation_errors')
ax8.legend()
 
plt.tight_layout()
plt.show()
 
# Save the plots
custom_name = str(input("Enter a custom name for the plots folder: "))
plot_path = os.path.join(PLOTS_PATH, "main", custom_name)
os.makedirs(plot_path, exist_ok=True)
fig1.savefig(os.path.join(plot_path, "distance_error_vs_time.png"))
fig2.savefig(os.path.join(plot_path, "distance_vs_error.png"))
fig3.savefig(os.path.join(plot_path, "energy_vs_time.png"))
fig4.savefig(os.path.join(plot_path, "zero_work_term_vs_time.png"))
fig5.savefig(os.path.join(plot_path, "torques_vs_time.png"))
fig6.savefig(os.path.join(plot_path, "constraints_vs_time.png"))
fig7.savefig(os.path.join(plot_path, "opt_itr_times.png"))


####################################################################  

# Graphics
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//20, lowerlimits=[-1,-1,-1], upperlimits=[1,6,1])
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.viz(
    show=True, 
    # save=True, 
    # path=os.path.join(plot_path, "3d_simulation.png"), 
    camera=(30, 30)
)

####################################################################  

# Plotting
plotter = Plotter(mpc, simulator) 
plotter.plot()
 

print(f"Simulation time: {end_sim_time - start_sim_time}")
