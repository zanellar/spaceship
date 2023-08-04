import json
import os
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

# Import parameters 
with open(os.path.join(PARAMS_PATH, model_param_file+".json")) as json_file:
    modelparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, mpc_param_file+".json")) as json_file:
    mpcparams = json.load(json_file)
    
with open(os.path.join(PARAMS_PATH, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)
    

# Create model, MPC and simulator
model = create_model()
mpc = create_mpc(model, void_policy, modelparams, mpcparams, envparams)
simulator = create_simulator(model, void_policy, modelparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,1,0]
simulator.x0['xn'] = [0,0,1]
x0 = simulator.x0.cat.full()
mpc.x0 = x0 
mpc.set_initial_guess() 


# Run MPC main loop
n_steps = 25
orientations = []
positions = []
for k in range(n_steps): 
    u0 = mpc.make_step(x0) 
    x0 = simulator.make_step(u0) 
    positions.append(x0[0:3])
    orientations.append(x0[6:9])

# Graphics
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=2, lowerlimits=-1, upperlimits=3)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()

# Plotting
plotter = Plotter(mpc, simulator) 
 