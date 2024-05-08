import json
import os
from datetime import datetime
import numpy as np
from spaceship.model import create_model
from spaceship.mpc import create_mpc
from spaceship.sim import create_simulator
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion 
from spaceship.policies import void_policy 
from spaceship.utils.paths import PARAMS_PATH, RESULTS_PATH

flg_baseline = False
# flg_baseline = True

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


# Create model, MPC and simulator
model = create_model(modelparams)
system = create_model(systemparams)
mpc = create_mpc(model, void_policy, modelparams, mpcparams, envparams)
simulator = create_simulator(system, void_policy, modelparams, simparams)

# Set initial state
simulator.x0['xp'] = [0,0,0.5]
simulator.x0['xv'] = [0,0.1,0] 
simulator.x0['xr1'] = [1,0,0]
simulator.x0['xr2'] = [0,1,0]
simulator.x0['xn'] = [0,0,1]

if flg_baseline:
    xw0 = input("Enter the initial angular velocity: ")
    try:
        xw0 = xw0.replace("[","")
        xw0 = xw0.replace("]","")
    except:
        pass
    xw0 = xw0.split(",")
    xw0 = [float(i) for i in xw0]
    simulator.x0['xw'] = xw0 
else:
    simulator.x0['xw'] = np.random.rand(3)*0.05 + 0.05
    print(simulator.x0['xw'])
    input("Press Enter to continue...")

x0 = simulator.x0.cat.full()
mpc.x0 = x0
mpc.set_initial_guess() 
x = x0 


# Run MPC main loop 
orientations = []
positions = []
errors = []
distances = []
energy = []
zero_work_term = []
torques = []
constraints = []
 
for k in range(simparams["n_steps"]):  

    print(f"step {k}")
    u = mpc.make_step(x) 
    
    if flg_baseline:
        u *= 0

    x = simulator.make_step(u) 

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


time_array = np.arange(simparams["n_steps"])   
torques = np.array(torques).reshape((simparams["n_steps"],3))
positions = np.array(positions).reshape((simparams["n_steps"],3))
orientations = np.array(orientations).reshape((simparams["n_steps"],3)) 
errors = np.array(errors).flatten()
energy = np.array(energy).flatten()
slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
   
# Save data as json file
data = {
    "initial_state": x0.tolist(),
    "mpcparams": mpcparams,
    "modelparams": modelparams,
    "simparams": simparams,
    "sysparams": systemparams,
    "envparams": envparams,
    "slits": slits,
    "orientations": orientations.tolist(), 
    "positions": positions.tolist(),
    "errors": errors.tolist(),
    "distances": np.array(distances).tolist(),
    "energy": energy.tolist(),
    "zero_work_term": np.array(zero_work_term).tolist(),
    "torques": torques.tolist(),
    "constraints": np.array(constraints).tolist(),
    "time_array": time_array.tolist() 
}

if flg_baseline:
    file_path = os.path.join(RESULTS_PATH, f"baseline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
else:
    file_path = os.path.join(RESULTS_PATH, f"controlled_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")

with open(file_path, 'w') as outfile:
    json.dump(data, outfile)


# # # Plotting
# plotter = Plotter(mpc, simulator) 
# plotter.plot()