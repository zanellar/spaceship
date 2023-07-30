import do_mpc
import casadi as ca
import numpy as np
import os
import json

from spaceship.utils.paths import PARAMS_PATH

def create_simulator(model, policy_fun, param_file):
        
    simulator = do_mpc.simulator.Simulator(model) 
    simulator.set_param(t_step = 0.1)

    # Parameters 
    p_template = simulator.get_p_template() # TODO: n_combinations 
    def sim_p_fun(t_now):   
        ''' Return the values of the model parameters '''
        with open(os.path.join(PARAMS_PATH, param_file+".json")) as json_file:
            param = json.load(json_file)
        p_template['m'] = param['m']
        p_template['j11'] = param['j11']
        p_template['j22'] = param['j22']
        p_template['j33'] = param['j33']
        return p_template 
    simulator.set_p_fun(sim_p_fun)

    # Force and torque policies   
    tvp_template = simulator.get_tvp_template()     
    def sim_tvp_fun(t_now):
        ''' Return the values of the force and torque (as time-varying parameters) at the current time step '''
        tvp_template['f1'] = policy_fun(t_now)['f1']
        tvp_template['tau1'] = policy_fun(t_now)['tau1']
        return tvp_template 
    simulator.set_tvp_fun(sim_tvp_fun)


    simulator.setup()
    return simulator