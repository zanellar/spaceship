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

    # Force and torque policies   
    tvp_template = mpc.get_tvp_template() 
    def mpc_tvp_fun(t_now):
        ''' Return the values of the force and torque (as time-varying parameters) at the current time step '''
        for t in range(len(tvp_template['_tvp'])): 
            tvp_template['_tvp',t,'f1'] = policy_fun(t+t_now, model)['f1']
            tvp_template['_tvp',t,'tau1'] = policy_fun(t+t_now, model)['tau1']
        return tvp_template  
    mpc.set_tvp_fun(mpc_tvp_fun)

    ###################### Objective function ######################

    # Real torque that is applied to the system 
    # print(ca.diag(model._u))
    # J = ca.diag(model._u)
    # tau = ca.mtimes(ca.skew(model.x['xw']), ca.mtimes(J, model.x['xw'])) + model.tvp['tau1']
    tau = model.aux['tau']
                       
    # Meyer term     
    mterm = ca.DM(0)
    # mterm = ca.mtimes(model.x['xw'].T, model.x['xw'])

    # Lagrange term 
    lterm = ca.mtimes(tau.T, tau)
    # lterm = ca.mtimes(model.x['xw'].T, model.x['xw'])

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
    g = (1 - ca.dot(model.x['xn'], envparams["xnd"])**2) / (ca.norm_2(model.x['xp'][0:2] - envparams["xpd"][0:2])**2 + 1/mpcparams["pos_weight"])
    # g *= ca.sign(model.x['xp'][0:2] - envparams["xpd"][0:2])

    mpc.set_nl_cons('g', g, ub=mpcparams["ub_err"], soft_constraint=False)
 
    # Scaling
    # TODO
    
    # Uncertain Parameters 
    # TODO

    mpc.setup()
    return mpc