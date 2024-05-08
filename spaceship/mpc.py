import do_mpc
import casadi as ca   
import numpy as np
import json 
import os


def create_mpc(model, modelparams, mpcparams, envparams):

    # Configuring the MPC controller
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters
    setup_mpc = {
        'n_horizon': mpcparams["n_horizon"],
        't_step': mpcparams["t_step"],
        'n_robust': 1,
        'collocation_ni': 2,
        'store_full_solution': True,
        'store_lagr_multiplier': False,
    }
    mpc.set_param(**setup_mpc)

    if mpcparams["silence"]:
        mpc.settings.supress_ipopt_output() 

    # Parameters 
    p_template = mpc.get_p_template(n_combinations=1) # TODO: n_combinations ?
    def mpc_p_fun(t_now): 
        ''' Return the values of the model parameters ''' 
        p_template["_p"] = [modelparams['m'], modelparams['j11'], modelparams['j22'], modelparams['j33']]
        return p_template 
    mpc.set_p_fun(mpc_p_fun)
  
    ###################### Objective function ######################
 
    xw = model.x['xw']
    u1 = model.u['u1']
    u2 = model.u['u2']
    u3 = model.u['u3']
    u = ca.diag(ca.vertcat(u1, u2, u3))
    u = u @ xw  
                       
    # Meyer term     
    mterm = ca.DM(0) 

    # Lagrange term  
    # tau = model.aux['tau']
    # lterm = ca.mtimes(tau.T, tau)
    tau_mpc = ca.cross(u, xw)  
    lterm = ca.mtimes(tau_mpc.T, tau_mpc)
    # lterm = ca.DM(0)
 
    # Weights of diagonal elements of R matrix  
    mpc.set_rterm(
        u1=mpcparams["rterm_u1"],
        u2=mpcparams["rterm_u2"],
        u3=mpcparams["rterm_u3"]
    ) 
 
    mpc.set_objective(mterm=mterm, lterm=lterm) 

    ###################### Constraints ######################

    # Lower bounds on states:
    # TODO
    # Upper bounds on states
    # TODO

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'u1'] = mpcparams["lb_u1"]
    mpc.bounds['lower','_u', 'u2'] = mpcparams["lb_u2"]
    mpc.bounds['lower','_u', 'u3'] = mpcparams["lb_u3"]
    # Lower bounds on inputs:
    mpc.bounds['upper','_u', 'u1'] = mpcparams["ub_u1"]
    mpc.bounds['upper','_u', 'u2'] = mpcparams["ub_u2"]
    mpc.bounds['upper','_u', 'u3'] = mpcparams["ub_u3"]

    # Nonlinear constraints  
    # g = (1 - ca.dot(model.x['xn'], envparams["xnd"])**2) / (ca.norm_2(model.x['xp'][0:2] - envparams["xpd"][0:2])**2 + 1/mpcparams["pos_weight"])
    g = ca.if_else(
        ca.norm_2(model.x['xp'][0:2]) - ca.norm_2(envparams["xpd"][0:2]) > 0, 
        0, 
        (1 - ca.dot(model.x['xn'], envparams["xnd"])**2) / (ca.norm_2(model.x['xp'][0:2] - envparams["xpd"][0:2])**2 + 1/mpcparams["pos_weight"])
    )
     
    mpc.set_nl_cons('g', g, ub=mpcparams["upper_bound_constraint"], soft_constraint=False)
 
    # Scaling
    # TODO
    
    # Uncertain Parameters 
    # TODO

    mpc.setup()
    return mpc