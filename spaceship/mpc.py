import do_mpc
import casadi as ca   
import numpy as np
import json 
import os

from spaceship.utils.paths import PARAMS_PATH

def create_mpc(model, policy_fun, param_file):
        
    # Configuring the MPC controller
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters
    setup_mpc = {
        'n_horizon': 20,
        't_step': 0.1,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Parameters 
    p_template = mpc.get_p_template(n_combinations=1) # TODO: n_combinations ?
    def mpc_p_fun(t_now): 
        ''' Return the values of the model parameters '''
        with open(os.path.join(PARAMS_PATH, param_file+".json")) as json_file:
            param = json.load(json_file)
        p_template["_p"] = [param['m'], param['j11'], param['j22'], param['j33']]
        return p_template 
    mpc.set_p_fun(mpc_p_fun)

    # Force and torque policies   
    tvp_template = mpc.get_tvp_template() 
    def mpc_tvp_fun(t_now):
        ''' Return the values of the force and torque (as time-varying parameters) at the current time step '''
        for t in range(len(tvp_template['_tvp'])): 
            tvp_template['_tvp',t,'f1'] = policy_fun(t+t_now)['f1']
            tvp_template['_tvp',t,'tau1'] = policy_fun(t+t_now)['tau1']
        return tvp_template  
    mpc.set_tvp_fun(mpc_tvp_fun)

    ###################### Objective function ######################

    # Real torque that is applied to the system
    J = ca.diag(ca.vertcat(model.p['j11'], model.p['j22'], model.p['j33']))
    tau = ca.mtimes(ca.skew(model.x['xw']), ca.mtimes(J, model.x['xw'])) + model.tvp['tau1']
                            
    # Lagrange term 
    mterm = ca.mtimes(tau.T, tau)

    # Meyer term
    lterm = ca.mtimes(tau.T, tau)

    # Weights of diagonal elements of R matrix  
    mpc.set_rterm(
        uj11=1e-2,
        uj22=1e-2,
        uj33=1e-2
    )

    mpc.set_objective(mterm=mterm, lterm=lterm)

    ###################### Constraints ######################

    # Lower bounds on states:
    # TODO
    # Upper bounds on states
    # TODO
    


    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'uj11'] = -10
    mpc.bounds['lower','_u', 'uj22'] = -10
    mpc.bounds['lower','_u', 'uj33'] = -10
    # Lower bounds on inputs:
    mpc.bounds['upper','_u', 'uj11'] = 10
    mpc.bounds['upper','_u', 'uj22'] = 10
    mpc.bounds['upper','_u', 'uj33'] = 10

    # Scaling
    # TODO
    
    # Uncertain Parameters 
    # TODO

    mpc.setup()
    return mpc