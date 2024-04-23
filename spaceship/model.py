import numpy as np
import do_mpc
import casadi as ca



def create_model(modelparams, dt):
    
    model_type = 'discrete'  
    model = do_mpc.model.Model(model_type) 

    ###################### Variables #########################

    # Current state
    xp = model.set_variable(var_type='_x', var_name='xp', shape=(3,1)) # Position (p)
    xv = model.set_variable(var_type='_x', var_name='xv', shape=(3,1)) # Velocity (dp)
    xr1 = model.set_variable(var_type='_x', var_name='xr1', shape=(3,1)) # First column of rotation matrix  
    xr2 = model.set_variable(var_type='_x', var_name='xr2', shape=(3,1)) # Second column of rotation matrix
    xn = model.set_variable(var_type='_x', var_name='xn', shape=(3,1)) # Third column of rotation matrix
    xw = model.set_variable(var_type='_x', var_name='xw', shape=(3,1)) # Angular velocity (omega)
  
    # Control variables 
    u1 = model.set_variable(var_type='_u', var_name='u1', shape=(1,1))
    u2 = model.set_variable(var_type='_u', var_name='u2', shape=(1,1))
    u3 = model.set_variable(var_type='_u', var_name='u3', shape=(1,1))
    u = ca.vertcat(u1, u2, u3) 

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

    # Rotation matrix
    R = ca.horzcat(xr1, xr2, xn) 

    # Cayley map
    def cay(x):
        return ca.SX.eye(3) + 0.5*ca.skew(x) @ ca.inv(ca.SX.eye(3) - 0.5*ca.skew(x))
    
    # Torque
    tau = - ca.cross(xw,u) + tau1

    # Intermediate angular velocity 
    _w = xw + 0.5*dt*ca.inv(J) @ (tau + ca.cross(J @ xw, xw))
    C = cay(dt*_w)

    # Right hand side of update equation 
    next_xp = xp + dt*xv
    next_xv = xv + dt*f1/m 
    next_xr1 = R @ C[:, 0 ]
    next_xr2 = R @ C[:, 1 ]
    next_xn = R @ C[:, 2 ]
    next_xw = _w + 0.5*dt*ca.inv(J) @ (tau + ca.cross(J @ _w, _w))

    # Set update equation
    model.set_rhs('xp', next_xp) 
    model.set_rhs('xv', next_xv)
    model.set_rhs('xr1', next_xr1)
    model.set_rhs('xr2', next_xr2)
    model.set_rhs('xn', next_xn)
    model.set_rhs('xw', next_xw)

    # Expressions for kinetic and potential energy
    # E_kin = # TODO
    # E_kin = # TODO
    # model.set_expression('E_kin', E_kin)
    # model.set_expression('E_pot', E_pot)
    model.set_expression('tau', tau)

    model.setup() 

    return model