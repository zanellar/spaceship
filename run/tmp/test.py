import json
import numpy as np
import os
from spaceship.model import create_model
from spaceship.mpc import create_mpc
from spaceship.sim import create_simulator
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion
import matplotlib.pyplot as plt

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
simulator = create_simulator(model, void_policy, modelparams, simparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,0,0]
simulator.x0['xr1'] = [1,0,0]
simulator.x0['xr2'] = [0,1,0]
simulator.x0['xn'] = [0,0,1] 
simulator.x0['xw'] = [1,0,0]
 
# Run main loop 
orientations = []
positions = [] 

# ax = plt.axes(projection = '3d') #Create axes 
# ax.quiver(*Drone3DStopMotion.origin, *Drone3DStopMotion.x_axis, color='k', label='X-axis', alpha=0.2)
# ax.quiver(*Drone3DStopMotion.origin, *Drone3DStopMotion.y_axis, color='k', label='Y-axis', alpha=0.2)
# ax.quiver(*Drone3DStopMotion.origin, *Drone3DStopMotion.z_axis, color='k', label='Z-axis', alpha=0.2)

for k in range(simparams["n_steps"]):  

    # Control input
    u = np.array([0, 0, 0])
    u = u.reshape((3,1))

    # Simulate
    x = simulator.make_step(u) 

    # states
    xp = x[0:3]
    xv = x[3:6]
    xr1 = x[6:9]
    xr2 = x[9:12]
    xn = x[12:15]
    xw = x[15:18] 
  
    print(k,  xn[0], xn[1], xn[2])
    # print(xr1, xr2, xn)
    print(xw[0], xw[1], xw[2])
    # print(xp,xv)

    # if k > 5:
    #     ax.quiver(*Drone3DStopMotion.origin, *np.array(xr1), color='r')
    #     ax.quiver(*Drone3DStopMotion.origin, *np.array(xr2), color='g' )
    #     ax.quiver(*Drone3DStopMotion.origin, *np.array(xn), color='b' )

    positions.append(xp)
    orientations.append(xn)


# Graphics

positions = np.array(positions).reshape((simparams["n_steps"],3))
orientations = np.array(orientations).reshape((simparams["n_steps"],3))

print("@@@@@@@@@@@@@@", np.linalg.norm(orientations[-1]-orientations[0]))
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//10, lowerlimits=-1, upperlimits=3)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()
  
# ax.set_xlim3d(-5, 5)
# ax.set_ylim3d(-5, 5)
# ax.set_zlim3d(-5, 5)
# plt.show()