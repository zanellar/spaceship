import json
import numpy as np
import os
from spaceship.model import create_model
from spaceship.mpc import create_mpc
from spaceship.sim import create_simulator
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion

from spaceship.policies import void_policy

from spaceship.utils.paths import PARAMS_PATH

# Choose parameter file
model_param_file = 'model_test'
mpc_param_file = 'mpc_test'
env_param_file = 'env_test'
sim_param_file = 'sim_test'

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
simulator = create_simulator(model, void_policy, modelparams, simparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,0,0]
simulator.x0['xr1'] = [1,0,0]
simulator.x0['xr2'] = [0,1,0]
simulator.x0['xn'] = [0,0,1] 
simulator.x0['xw'] = [1,1,1]
 
# Run main loop
n_steps = 500
orientations = []
positions = [] 
 
for k in range(n_steps):  

    # Control input
    u = np.array([0, 0, 0])
    u = u.reshape((3,1))

    # Simulate
    x = simulator.make_step(u) 

    print(simulator.x0['xr1'], simulator.x0['xr2'], simulator.x0['xn'])
    print(np.linalg.norm(simulator.x0['xr1']), np.linalg.norm(simulator.x0['xr2']), np.linalg.norm(simulator.x0['xn']))
 
    positions.append(x[0:3])
    orientations.append(x[6:9])


# Graphics
print("@@@@@@@@@@@@@@", np.linalg.norm(orientations[-1]-orientations[0]))
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=0, lowerlimits=-1, upperlimits=3)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()
  