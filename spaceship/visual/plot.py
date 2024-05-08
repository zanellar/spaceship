

# Setting up the Graphic
import matplotlib  
import matplotlib.pyplot as plt 
import do_mpc

# Customizing Matplotlib:
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['axes.grid'] = True

class Plotter():
    def __init__(self,mpc, simulator) -> None:
        
        self.mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
        self.sim_graphics = do_mpc.graphics.Graphics(simulator.data)
        
        # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
        # fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
        fig = plt.figure(figsize=(16,9))
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        fig.align_ylabels()

        for g in [self.sim_graphics, self.mpc_graphics]:
            # Plot state
            g.add_line(var_type='_x', var_name='xp', axis=ax1)
            g.add_line(var_type='_x', var_name='xv', axis=ax1)
            g.add_line(var_type='_x', var_name='xn', axis=ax1)
            g.add_line(var_type='_x', var_name='xw', axis=ax1) 
            # Plot inputs
            g.add_line(var_type='_u', var_name='u1', axis=ax2)
            g.add_line(var_type='_u', var_name='u2', axis=ax2)
            g.add_line(var_type='_u', var_name='u3', axis=ax2)


        ax1.set_ylabel('State p, v, n, w')
        ax2.set_ylabel('Inertia u')
        ax2.set_xlabel('time [s]')
        ax1.legend()
        ax2.legend()

    def plot(self):
        self.sim_graphics.plot_results() 
        self.mpc_graphics.plot_predictions()
        plt.show()
