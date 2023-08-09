import do_mpc
import casadi as ca   
import numpy as np
import json 
import os


def create_mpc(model, policy_fun, modelparams, mpcparams, envparams):

    # Configuring the MPC controller
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters
    setup_mpc = {
        'n_horizon': mpcparams["n_horizon"],
        't_step': mpcparams["t_step"],
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Parameters 
    p_template = mpc.get_p_template(n_combinations=1) # TODO: n_combinations ?
    def mpc_p_fun(t_now): 
        ''' Return the values of the model parameters ''' 
        p_template["_p"] = [modelparams['m'], modelparams['j11'], modelparams['j22'], modelparams['j33']]
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
    # J = ca.diag(ca.vertcat(model.u['uj11'], model.u['uj22'], model.u['uj33']))
    # tau = ca.mtimes(ca.skew(model.x['xw']), ca.mtimes(J, model.x['xw'])) + model.tvp['tau1']
                            
    # Lagrange term 
    mterm = ca.mtimes(model.x['xw'].T, model.x['xw'])

    # Meyer term
    lterm = ca.mtimes(model.x['xw'].T, model.x['xw'])

    # Weights of diagonal elements of R matrix  
    mpc.set_rterm(
        uj11=mpcparams["rterm_uj11"],
        uj22=mpcparams["rterm_uj22"],
        uj33=mpcparams["rterm_uj33"]
    )

    mpc.set_objective(mterm=mterm, lterm=lterm)

    ###################### Constraints ######################

    # Lower bounds on states:
    # TODO
    # Upper bounds on states
    # TODO

    # Lower bounds on inputs:
    # mpc.bounds['lower','_u', 'uj11'] = mpcparams["lb_uj11"]
    # mpc.bounds['lower','_u', 'uj22'] = mpcparams["lb_uj22"]
    # mpc.bounds['lower','_u', 'uj33'] = mpcparams["lb_uj33"]
    # # Lower bounds on inputs:
    # mpc.bounds['upper','_u', 'uj11'] = mpcparams["ub_uj11"]
    # mpc.bounds['upper','_u', 'uj22'] = mpcparams["ub_uj22"]
    # mpc.bounds['upper','_u', 'uj33'] = mpcparams["ub_uj33"]

    # Nonlinear constraints
    
    g = (1 - ca.dot(model.x['xn'], envparams["xnd"])**2) / (ca.norm_2(model.x['xp'] - envparams["xpd"])**2 + 1/mpcparams["pos_weight"])
    mpc.set_nl_cons('g', g, ub=mpcparams["ub_err"], soft_constraint=False)
 
    # Scaling
    # TODO
    
    # Uncertain Parameters 
    # TODO

    mpc.setup()
    return mpc