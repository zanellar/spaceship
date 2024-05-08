import os
import json

import numpy as np 
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion
 
from spaceship.utils.paths import RESULTS_PATH, PARAMS_PATH, PLOTS_PATH
 
import matplotlib.pyplot as plt  

flg_show_plots = True

res_path = os.path.join(RESULTS_PATH, "") 

# list of file names contained in the results folder excluding the baseline
results_file_name_list = [ name for name in os.listdir(res_path) if name.endswith(".json") and not name.startswith("baseline")]

# take file name of the baseline in the results folder if unique, otherwise show all the files in prompt and ask the user to select the baseline file
if len([ name for name in os.listdir(res_path) if name.startswith("baseline")]) == 1:
    baseline_file_name = [ name for name in os.listdir(res_path) if name.startswith("baseline")][0]
else:
    print("The baseline file is not unique. Please select the baseline file from the list below:")
    for i, name in enumerate([ name for name in os.listdir(res_path) if name.startswith("baseline")]):
        print(str(i) + ") " + name)
    index = input("Enter the baseline file name: ")
    baseline_file_name = [ name for name in os.listdir(res_path) if name.startswith("baseline")][int(index)]
     
# Parameter file 
env_param_file = 'env_test'
sim_param_file = 'sim_test' 
  
#########################################################################################################
#########################################################################################################
#########################################################################################################

results = {}

for i, results_file_name in enumerate(results_file_name_list):

    # Load the results
    with open(os.path.join(res_path, results_file_name)) as json_file:
        results[results_file_name] = json.load(json_file)
  
# Load the baseline
with open(os.path.join(res_path, baseline_file_name)) as json_file:
    baseline = json.load(json_file)
  
############################################################################################################## 

# Import parameters  
with open(os.path.join(PARAMS_PATH, env_param_file+".json")) as json_file:
    envparams = json.load(json_file)

with open(os.path.join(PARAMS_PATH, sim_param_file+".json")) as json_file:
    simparams = json.load(json_file)

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
for i, results_file_name in enumerate(results_file_name_list): 
    ax1.plot(results[results_file_name]["time_array"], results[results_file_name]["errors"], label='orientation error controller '+str(i))
ax1.plot(baseline["time_array"], baseline["distances"], label='distance from the slit')
ax1.plot(baseline["time_array"], baseline["errors"], label='orientation error baseline')
ax1.set_xlabel('time')
ax1.set_ylabel('distance')
ax1.set_title('Distance and error vs time')
ax1.legend()

# 2) Plot the distance vs error
for i, results_file_name in enumerate(results_file_name_list): 
    ax2.plot(results[results_file_name]["distances"], results[results_file_name]["errors"], label='controller '+str(i))
ax2.plot(baseline["distances"], baseline["errors"], label='baseline') 
ax2.set_xlabel('distance')
ax2.set_ylabel('error')
ax2.set_title('Error vs distance')
ax2.legend()

# Plot energy vs time
for i, results_file_name in enumerate(results_file_name_list): 
    ax3.plot(results[results_file_name]["time_array"], results[results_file_name]["energy"], label='controller '+str(i))
ax3.plot(baseline["time_array"], baseline["energy"], label='baseline')
ax3.set_xlabel('time')
ax3.set_ylabel('energy')
ax3.set_title('Energy vs time')
ax3.legend() 

# Plot zero work term vs time
for i, results_file_name in enumerate(results_file_name_list): 
    ax4.plot(results[results_file_name]["time_array"], results[results_file_name]["zero_work_term"], label='controller '+str(i))
ax4.plot(baseline["time_array"], baseline["zero_work_term"], label='baseline')
ax4.set_xlabel('time')
ax4.set_ylabel('zero work term')
ax4.set_title('Zero work term vs time')
ax4.legend()

# plot the torque vs time
for i, results_file_name in enumerate(results_file_name_list):  
    ax5.plot(results[results_file_name]["time_array"], results[results_file_name]["torques"], label='controller '+str(i))
ax5.plot(baseline["time_array"], baseline["torques"], label='baseline') 
# ax5.scatter(baseline["time_array"], torques[:,0], label='torque1')
ax5.set_xlabel('time')
ax5.set_ylabel('torques')
ax5.set_title('Torques vs time')
ax5.legend()

# plot the constraints vs time
for i, results_file_name in enumerate(results_file_name_list): 
    ax6.plot(results[results_file_name]["time_array"], results[results_file_name]["constraints"], label='controller '+str(i))
ax6.plot(baseline["time_array"], baseline["constraints"], label='baseline')
ax6.set_xlabel('time')
ax6.set_ylabel('constraints')
ax6.set_title('Constraints vs time')
ax6.legend()

plt.tight_layout()

# # Save the plots  
fig1.savefig(os.path.join(PLOTS_PATH, results_file_name+"_distances_errors_time.png"))
fig2.savefig(os.path.join(PLOTS_PATH, results_file_name+"_distances_errors.png"))
fig3.savefig(os.path.join(PLOTS_PATH, results_file_name+"_energy.png"))
fig4.savefig(os.path.join(PLOTS_PATH, results_file_name+"_zero_work_term.png"))
fig5.savefig(os.path.join(PLOTS_PATH, results_file_name+"_torques.png"))
fig6.savefig(os.path.join(PLOTS_PATH, results_file_name+"_constraints.png"))

# show the plots
if flg_show_plots:
    plt.show() 
