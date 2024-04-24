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
from spaceship.policies import void_policy
from spaceship.utils.paths import PARAMS_PATH

# Choose parameter file
model_param_file = 'model1'
mpc_param_file = 'mpc1'
env_param_file = 'env1'
sim_param_file = 'sim1'
sys_param_file = 'sys1'

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


# Create model, controller, system and simulator
system = create_model(systemparams, dt=simparams["t_step"])
simulator = create_simulator(system, void_policy, modelparams, simparams)
model = create_model(modelparams, dt=mpcparams["t_step"])
mpc = create_mpc(model, void_policy, modelparams, mpcparams, envparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,0.1,0] 
simulator.x0['xr1'] = [1,0,0]
simulator.x0['xr2'] = [0,1,0]
simulator.x0['xn'] = [0,0,1]
simulator.x0['xw'] = [0.05,0.05,0.05] 
x = simulator.x0.cat.full()
mpc.x0 = x 
mpc.set_initial_guess() 


# Run MPC main loop 
orientations = []
positions = []
errors = []
distances = []
energy = []
zero_work_term = []
torques = []
constraints = []
opt_times = [] 
itr_times = []
  
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
 
    # energy.append(0.5*m*np.dot(xv.T,xv) + 0.5*xw.T @ J @ xw)   
    energy.append(0.5*xw.T @ J @ xw)   
    torques.append(np.cross(xw.flatten(), (np.diag(u.flatten()) @ xw).flatten()))
    zero_work_term.append( np.dot(xw.T, np.cross(xw.flatten(), (J @ xw).flatten())))

    constraints.append((1 - np.dot(xn.flatten(), envparams["xnd"])**2) / (np.linalg.norm(xp[0:2] - envparams["xpd"][0:2])**2 + 1/mpcparams["pos_weight"]))
   
    positions.append(xp)
    orientations.append(xn) 

    errors.append(1-np.linalg.norm(np.dot(xn.T,np.array(envparams["xnd"]))))
    distances.append(np.linalg.norm(xp[0:2]-np.array(envparams["xpd"]).reshape((3,1))[0:2]))

 
# Plot the trajectories# Plot the trajectories
fig1, ax1 = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(8, 8))
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig4, ax4 = plt.subplots(figsize=(8, 8))
fig5, ax5 = plt.subplots(figsize=(8, 8))
fig6, ax6 = plt.subplots(figsize=(8, 8))
fig7, ax7 = plt.subplots(figsize=(8, 8))

positions = np.array(positions).reshape((simparams["n_steps"],3))
orientations = np.array(orientations).reshape((simparams["n_steps"],3)) 
errors = np.array(errors).flatten()
energy = np.array(energy).flatten()

# Plot distance of the drone to the target slit and the orientation error:
# 1) Plot the distance & error vs time
time_array = np.arange(simparams["n_steps"])
ax1.plot(time_array, distances, label='distances')
ax1.plot(time_array, errors, label='errors')
ax1.set_xlabel('step')
ax1.set_ylabel('distance [m]')
ax1.set_title('Distance and error vs time')
ax1.legend()

# 2) Plot the distance vs error
ax2.plot(distances, errors ) 
ax2.set_xlabel('distance [m]')
ax2.set_ylabel('error')
ax2.set_title('Error vs distance')
ax2.legend()

# Plot energy vs time
ax3.plot(time_array, energy )
ax3.set_xlabel('step')
ax3.set_ylabel('energy')
ax3.set_title('Energy vs time')
ax3.legend() 

# Plot zero work term vs time
ax4.plot(time_array, zero_work_term )
ax4.set_xlabel('step')
ax4.set_ylabel('zero work term')
ax4.set_title('Zero work term vs time')
ax4.legend()

# plot the torque vs time
torques = np.array(torques).reshape((simparams["n_steps"],3))
ax5.plot(time_array, torques ) 
# ax5.scatter(time_array, torques[:,0], label='torque1')
ax5.set_xlabel('step')
ax5.set_ylabel('torques')
ax5.set_title('Torques vs time')
ax5.legend()

# plot the constraints vs time
ax6.plot(time_array, constraints )
ax6.set_xlabel('step')
ax6.set_ylabel('constraints')
ax6.set_title('Constraints vs time')
ax6.legend()

# plot the optimization and iteration times
ax7.plot(time_array, opt_times, label='optimization')
ax7.plot(time_array, itr_times, label='simulation step') 
ax7.set_xlabel('step')
ax7.set_ylabel('time [s]')


end_sim_time = time.time()
print(f"Simulation time: {end_sim_time - start_sim_time}")

plt.tight_layout()
plt.show()

# Graphics
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//20, lowerlimits=-1, upperlimits=5)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()

# # Plotting
plotter = Plotter(mpc, simulator) 
plotter.plot()
 
