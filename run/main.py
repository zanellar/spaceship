import json
import os
import numpy as np
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

# Import parameters 
with open(os.path.join(PARAMS_PATH, model_param_file+".json")) as json_file:
    modelparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, mpc_param_file+".json")) as json_file:
    mpcparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)

with open(os.path.join(PARAMS_PATH, sim_param_file+".json")) as json_file:
    simparams = json.load(json_file)
    

# Create model, MPC and simulator
model = create_model(modelparams)
mpc = create_mpc(model, void_policy, modelparams, mpcparams, envparams)
simulator = create_simulator(model, void_policy, modelparams, simparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,0.1,0] 
simulator.x0['xr1'] = [1,0,0]
simulator.x0['xr2'] = [0,1,0]
simulator.x0['xn'] = [0,0,1]
simulator.x0['xw'] = [0.1,0.1,0.1] 
x = simulator.x0.cat.full()
mpc.x0 = x 
mpc.set_initial_guess() 


# Run MPC main loop 
orientations = []
positions = []
errors = []
distances = []

for k in range(simparams["n_steps"]):  

    print(f"step {k}")
    u = mpc.make_step(x)  
    x = simulator.make_step(u) 

    # states
    xp = x[0:3]
    xv = x[3:6]
    xr1 = x[6:9]
    xr2 = x[9:12]
    xn = x[12:15]
    xw = x[15:18] 
   
    positions.append(xp)
    orientations.append(xn) 

    errors.append(1-np.dot(xn.T,np.array(envparams["xnd"]).reshape((3,1))))
    distances.append(np.linalg.norm(xp[0:2]-np.array(envparams["xpd"]).reshape((3,1))[0:2]))


import matplotlib.pyplot as plt  

# Plot the trajectories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

positions = np.array(positions).reshape((simparams["n_steps"],3))
orientations = np.array(orientations).reshape((simparams["n_steps"],3))
print(orientations )
print(distances )
errors = np.array(errors).flatten()
  
# Plot orientation trajectory (Euler angles)
time_array = np.arange(simparams["n_steps"])   # Create the time array
ax1.plot(time_array, distances, label='distances')
ax1.plot(time_array, errors, label='errors')
ax2.plot(distances, errors ) 

#legend, axes and title
ax1.set_xlabel('time')
ax1.set_ylabel('distance')
ax1.set_title('Distance and error')
ax1.legend()

ax2.set_xlabel('distance')
ax2.set_ylabel('error')
ax2.set_title('Error vs distance')
ax2.legend()
 

plt.tight_layout()
plt.show()

# Graphics
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//20, lowerlimits=-1, upperlimits=5)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()

# # Plotting
# plotter = Plotter(mpc, simulator) 
# plotter.plot()
 
