import do_mpc
import casadi as ca
import numpy as np
import os
import json

from spaceship.utils.paths import PARAMS_PATH

def create_simulator(model, modelparams, simparams):
        
    simulator = do_mpc.simulator.Simulator(model) 
    simulator.set_param(t_step = simparams['t_step'])

    # Parameters 
    p_template = simulator.get_p_template() # TODO: n_combinations 
    def sim_p_fun(t_now):   
        ''' Return the values of the model parameters ''' 
        p_template['m'] = modelparams['m']
        p_template['j11'] = modelparams['j11']
        p_template['j22'] = modelparams['j22']
        p_template['j33'] = modelparams['j33']
        return p_template 
    simulator.set_p_fun(sim_p_fun)
 
    simulator.setup()
    return simulator