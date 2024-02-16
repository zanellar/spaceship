import os
import json

import numpy as np 
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion
 
from spaceship.utils.paths import RESULTS_PATH, PARAMS_PATH, PLOTS_PATH
 
import matplotlib.pyplot as plt  

flg_show_plots = False
results_file_name = "controlled_2024-02-16_16-48-27"
 
# Parameter file 
env_param_file = 'env1'
sim_param_file = 'sim1' 

#########################################################################################################
#########################################################################################################
#########################################################################################################

# Load the results
with open(os.path.join(RESULTS_PATH, results_file_name+".json")) as json_file:
    results = json.load(json_file)

# Extract the results
time_array = results["time_array"]
distances = results["distances"]
orientations = results["orientations"]
positions = results["positions"]
errors = results["errors"]
energy = results["energy"]
torques = results["torques"]
zero_work_term = results["zero_work_term"]
constraints = results["constraints"]
slits = results["slits"] 
 
# Import parameters  
with open(os.path.join(PARAMS_PATH, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)

with open(os.path.join(PARAMS_PATH, sim_param_file+".json")) as json_file:
    simparams = json.load(json_file)
  
############################################################################################################## 
    
# Graphics
graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//20, lowerlimits=-1, upperlimits=5)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()
 
##############################################################################################################

# Plot the trajectories# Plot the trajectories
fig1, ax1 = plt.subplots(figsize=(8, 8))
fig2, ax2 = plt.subplots(figsize=(8, 8))
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig4, ax4 = plt.subplots(figsize=(8, 8))
fig5, ax5 = plt.subplots(figsize=(8, 8))
fig6, ax6 = plt.subplots(figsize=(8, 8))
 
# Plot distance of the drone to the target slit and the orientation error:
# 1) Plot the distance & error vs time 
ax1.plot(time_array, distances, label='distances')
ax1.plot(time_array, errors, label='errors')
ax1.set_xlabel('time')
ax1.set_ylabel('distance')
ax1.set_title('Distance and error vs time')
ax1.legend()

# 2) Plot the distance vs error
ax2.plot(distances, errors ) 
ax2.set_xlabel('distance')
ax2.set_ylabel('error')
ax2.set_title('Error vs distance')
ax2.legend()

# Plot energy vs time
ax3.plot(time_array, energy )
ax3.set_xlabel('time')
ax3.set_ylabel('energy')
ax3.set_title('Energy vs time')
ax3.legend() 

# Plot zero work term vs time
ax4.plot(time_array, zero_work_term )
ax4.set_xlabel('time')
ax4.set_ylabel('zero work term')
ax4.set_title('Zero work term vs time')
ax4.legend()

# plot the torque vs time
ax5.plot(time_array, torques ) 
# ax5.scatter(time_array, torques[:,0], label='torque1')
ax5.set_xlabel('time')
ax5.set_ylabel('torques')
ax5.set_title('Torques vs time')
ax5.legend()

# plot the constraints vs time
ax6.plot(time_array, constraints )
ax6.set_xlabel('time')
ax6.set_ylabel('constraints')
ax6.set_title('Constraints vs time')
ax6.legend()

plt.tight_layout()

# Save the plots  
fig1.savefig(os.path.join(PLOTS_PATH, results_file_name+"_distances_errors.png"))
fig2.savefig(os.path.join(PLOTS_PATH, results_file_name+"_distances_errors.png"))
fig3.savefig(os.path.join(PLOTS_PATH, results_file_name+"_energy.png"))
fig4.savefig(os.path.join(PLOTS_PATH, results_file_name+"_zero_work_term.png"))
fig5.savefig(os.path.join(PLOTS_PATH, results_file_name+"_torques.png"))
fig6.savefig(os.path.join(PLOTS_PATH, results_file_name+"_constraints.png"))

# show the plots
if flg_show_plots:
    plt.show() 
