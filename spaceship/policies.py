import numpy as np
import casadi as ca
 

####################################################################  

class NullControl:
    def __init__(self, dim_control=3, target={}, params={}):
        self.dim_control = dim_control
        self.target = target
        self.params = params

    def __call__(self, model):
        return np.zeros(self.dim_control)
      
    def energy(self, x): 
        return 0 
    
####################################################################  

class PosPDControl:
    def __init__(self, params=dict(Kp=0.1, Kd=1), target=[0,0,0]):
        self.params = params
        self.target = target

    def __call__(self, model):
        xd =  np.array(self.target)
        x = model.x['xp'] 
        v = model.x['xv'] 

        Kp = self.params['Kp']
        Kd = self.params['Kd']

        f_nominal = - Kp*(x - xd) - Kd*v
        return f_nominal

    def energy(self, x):
        # raise NotImplementedError
        return 0 
    
####################################################################  

class RotPDControl:
    def __init__(self, params=dict(Kp=0.1, Kd=1, a1=1, a2=1, a3=1), target=np.eye(3)):
        self.params = params
        self.target = target

    def __call__(self, model):
        Rd =  np.array(self.target)
        w = model.x['xw'] 
        r1 = model.x['xr1'] 
        r2 = model.x['xr2']
        n = model.x['xn']

        a1 = self.params['a1']
        a2 = self.params['a2']
        a3 = self.params['a3']

        e1 = np.array([1,0,0])
        e2 = np.array([0,1,0])
        e3 = np.array([0,0,1])

        rot_err = a1*ca.cross(e1, Rd.T @ r1) + a2*ca.cross(e2, Rd.T @ r2) + a3*ca.cross(e3, Rd.T @ n)
        
        Kp = self.params['Kp']
        Kd = self.params['Kd']

        tau_nominal = - Kp*rot_err - Kd*w
        return tau_nominal

    def energy(self, x):
        raise NotImplementedError
    
####################################################################  

class GeomRotPDControl:

    def __init__(self, params=dict(g1=0.1, g2=0.1, g3=0.1, Kd=1), target=np.eye(3)):
        self.params = params
        self.target = target

    def __call__(self, model):
        Rd =  np.array(self.target)
        w = model.x['xw'] 
        r1 = model.x['xr1'] 
        r2 = model.x['xr2']
        n = model.x['xn']

        R = ca.horzcat(r1, r2, n)

        g1 = self.params['g1']
        g2 = self.params['g2']
        g3 = self.params['g3']

        G = ca.diag(ca.vertcat(g1, g2, g3))
    
        Kd = self.params['Kd']

        A = G @ Rd.T @ R
        skA = (A - A.T)/2
        sktau = -2*skA
        vtau = ca.vertcat(sktau[2,1], sktau[0,2], sktau[1,0])
        tau_nominal = vtau - Kd*w 
        # tau_nominal = -2*va - Kd*w 
        return tau_nominal

    def energy(self, x):
        xp = x[0:3]
        xv = x[3:6]
        xr1 = x[6:9]
        xr2 = x[9:12]
        xn = x[12:15]
        xw = x[15:18] 
  
        G = np.diag([self.params['g1'], self.params['g2'], self.params['g3']])
        R = np.array(self.target).T @ np.column_stack((xr1, xr2, xn))
        E = - np.trace(G @ (R - np.eye(3)))
        
        E = 2*E     # BUG need to multiply by 2 

        return E
    