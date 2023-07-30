import time
from spaceship.model import create_model
from spaceship.mpc import create_mpc
from spaceship.sim import create_simulator
from spaceship.visual.plot import Plotter

from spaceship.policies import void_policy

param_file = 'model1'

model = create_model()
mpc = create_mpc(model, void_policy, param_file)
simulator = create_simulator(model, void_policy, param_file)

# Set initial state
simulator.x0['xp'] = [0,0,0]
simulator.x0['xv'] = [0,1,0]
simulator.x0['xn'] = [0,0,1]
x0 = simulator.x0.cat.full()
mpc.x0 = x0 
mpc.set_initial_guess() 

# Run MPC main loop
plotter = Plotter(mpc, simulator)
n_steps = 10
for k in range(n_steps): 
    u0 = mpc.make_step(x0) 
    x0 = simulator.make_step(u0) 
plotter.plot()
 