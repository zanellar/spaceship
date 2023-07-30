import numpy as np
import do_mpc
import casadi as ca



def create_model():
    
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    ###################### Variables #########################

    # State variables
    xp = model.set_variable(var_type='_x', var_name='xp', shape=(3,1)) # Position (p)
    xv = model.set_variable(var_type='_x', var_name='xv', shape=(3,1)) # Velocity (dp)
    xn = model.set_variable(var_type='_x', var_name='xn', shape=(3,1)) # Attitude vector (n)
    xw = model.set_variable(var_type='_x', var_name='xw', shape=(3,1)) # Angular velocity (omega)

    # State derivatives 
    dxp = model.set_variable(var_type='_x', var_name='dxp', shape=(3,1))
    dxv = model.set_variable(var_type='_x', var_name='dxv', shape=(3,1))
    dxn = model.set_variable(var_type='_x', var_name='dxn', shape=(3,1))
    dxw = model.set_variable(var_type='_x', var_name='dxw', shape=(3,1)) 

    # Control variables (diagonal of inertia matrix)
    uj11 = model.set_variable(var_type='_u', var_name='uj11', shape=(1,1))
    uj22 = model.set_variable(var_type='_u', var_name='uj22', shape=(1,1))
    uj33 = model.set_variable(var_type='_u', var_name='uj33', shape=(1,1))
    uJ = ca.diag(ca.vertcat(uj11, uj22, uj33))

    # Policies
    f1 = model.set_variable(var_type='_tvp', var_name='f1', shape=(3,1))
    tau1 = model.set_variable(var_type='_tvp', var_name='tau1', shape=(3,1))

    # Parameters
    j11 = model.set_variable(var_type='_p', var_name='j11', shape=(1,1)) 
    j22 = model.set_variable(var_type='_p', var_name='j22', shape=(1,1)) 
    j33 = model.set_variable(var_type='_p', var_name='j33', shape=(1,1)) 
    m = model.set_variable(var_type='_p', var_name='m', shape=(1,1))
    J = ca.diag(ca.vertcat(j11, j22, j33))

    print(model.x.labels()) 
    
    ###################### Equation ######################
    model.set_rhs('xp', dxp) 
    model.set_rhs('xv', dxv)
    model.set_rhs('xn', dxn)
    model.set_rhs('xw', dxw)

    # Nonlinear ODE 
    eq_dxp = xv
    eq_dxv = f1/m 
    eq_dxn = ca.mtimes(xn.T, ca.skew(xw)).T  
    eq_dxw = ca.mtimes(ca.inv(J), (- ca.mtimes(ca.skew(xw), ca.mtimes(J, xw)) - ca.mtimes(ca.skew(xw), ca.mtimes(uJ, xw)) + tau1))

    model.set_rhs('dxp', eq_dxp) 
    model.set_rhs('dxv', eq_dxv)
    model.set_rhs('dxn', eq_dxn)
    model.set_rhs('dxw', eq_dxw) 

    # Expressions for kinetic and potential energy
    # E_kin = # TODO
    # E_kin = # TODO
    # model.set_expression('E_kin', E_kin)
    # model.set_expression('E_pot', E_pot)

    model.setup()
    return model