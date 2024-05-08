import json
import os
import time 
from datetime import datetime
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
from spaceship.utils.paths import PARAMS_PATH, RESULTS_PATH

##########################################################################################################################################################  
##########################################################################################################################################################  
##########################################################################################################################################################  
########################################################################################################################################################## 
########################################################################################################################################################## 

def run_and_save(modelparams, mpcparams, envparams, simparams, systemparams, save_file_path):

    # ------- ADDITIONAL CONTROLLERS ---------

    pos_control = globals()[simparams["params_ctrl_pos"]["policy_func"]](target=list(simparams["target_pos"]), params=dict(simparams["params_ctrl_pos"]))
    rot_control = globals()[simparams["params_ctrl_rot"]["policy_func"]](target=list(simparams["target_rot"]), params=dict(simparams["params_ctrl_rot"]))

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
        
    # Save data as json file
    data = { 
        "disable_mpc": simparams["disable_mpc"],
        "disable_nominal_control": simparams["disable_nominal_control"],
        "mpcparams": mpcparams,
        "modelparams": modelparams,
        "simparams": simparams,
        "sysparams": systemparams,
        "envparams": envparams,
        "slits": [dict(height=3, width=0.2, position=envparams["xpd"])],
        "time_array": np.arange(simparams["n_steps"]).tolist(),
        "orientations": np.array(orientations).reshape((simparams["n_steps"],3)).tolist(),
        "positions": np.array(positions).reshape((simparams["n_steps"],3)).tolist(),
        "slit_orientation_errors": np.array(slit_orientation_errors).flatten().tolist(),
        "slit_distances": np.array(slit_distances).flatten().tolist(),
        "total_energy": np.array(total_energy).flatten().tolist(),
        "kinetic_energy": np.array(kinetic_energy).flatten().tolist(),
        "potential_energy": np.array(potential_energy).flatten().tolist(),
        "zero_work_term": np.array(zero_work_term).flatten().tolist(),
        "torques": np.array(torques).reshape((simparams["n_steps"],3)).tolist(),
        "constraints": np.array(constraints).flatten().tolist(),
        "opt_times": np.array(opt_times).flatten().tolist(),
        "itr_times": np.array(itr_times).flatten().tolist(),
        "target_orientation_errors": np.array(target_orientation_errors).flatten().tolist(),
        "target_position_errors": np.array(target_position_errors).flatten().tolist(),
    } 

    with open(save_file_path, 'w') as outfile:
        json.dump(data, outfile)



##########################################################################################################################################################  
##########################################################################################################################################################  
##########################################################################################################################################################  
########################################################################################################################################################## 
########################################################################################################################################################## 

paperid = "paper052024"

# Choose parameter file
model_param_file = 'model_paper'
mpc_param_file = 'mpc_paper'
env_param_file = 'env_paper'
sim_param_file = 'sim_paper'
sys_param_file = 'sys_paper'

# Create folders
params_folder_path = os.path.join(PARAMS_PATH, paperid)
results_folder_path = os.path.join(RESULTS_PATH, paperid)
os.makedirs(params_folder_path, exist_ok=True)
os.makedirs(results_folder_path, exist_ok=True)

# Import parameters 
with open(os.path.join(params_folder_path, model_param_file+".json")) as json_file:
    modelparams = json.load(json_file)
    
with open(os.path.join(params_folder_path, mpc_param_file+".json")) as json_file:
    mpcparams = json.load(json_file)
    
with open(os.path.join(params_folder_path, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)

with open(os.path.join(params_folder_path, sim_param_file+".json")) as json_file:
    simparams = json.load(json_file)

with open(os.path.join(params_folder_path, sys_param_file+".json")) as json_file:
    systemparams = json.load(json_file)
 

for disable_mpc in [True, False]:
    for disable_nominal_control in [True, False]:
        simparams["disable_mpc"] = disable_mpc
        simparams["disable_nominal_control"] = disable_nominal_control
        save_file_name = f"results_paper"
        if not disable_mpc:
            save_file_name += "_mpc"
        if not disable_nominal_control:
            save_file_name += "_pd"
        if disable_mpc and disable_nominal_control:
            save_file_name += "_openloop"
        save_file_name = f"{save_file_name}.json"
        save_file_path = os.path.join(results_folder_path, save_file_name) 
        run_and_save(modelparams, mpcparams, envparams, simparams, systemparams, save_file_path)