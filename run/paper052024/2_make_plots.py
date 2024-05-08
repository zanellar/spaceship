import json
import os
import time 
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt  
 
from spaceship.visual.plot import Plotter
from spaceship.visual.draw3d import Drone3DStopMotion
from spaceship.visual.draw3d import Drone3DStopMotion

from spaceship.visual.draw3d import Drone3DStopMotion 

from spaceship.policies import *
from spaceship.utils.paths import PARAMS_PATH, PLOTS_PATH, RESULTS_PATH


paperid = "paper052024"

results_folder_path = os.listdir(os.path.join(RESULTS_PATH, paperid))
fig_ready = None

for results_file_name in results_folder_path:
    
    # Load the results
    with open(os.path.join(RESULTS_PATH, paperid, results_file_name)) as json_file:
        results = json.load(json_file)

    # Extract the results
    mpcparams = results["mpcparams"]
    modelparams = results["modelparams"]
    simparams = results["simparams"]
    systemparams = results["sysparams"]
    envparams = results["envparams"]
    slits = results["slits"]
    time_array = results["time_array"]
    orientations = results["orientations"]
    positions = results["positions"]
    slit_orientation_errors = results["slit_orientation_errors"]
    slit_distances = results["slit_distances"]
    total_energy = results["total_energy"]
    kinetic_energy = results["kinetic_energy"]
    potential_energy = results["potential_energy"]
    zero_work_term = results["zero_work_term"]
    torques = results["torques"]
    constraints = results["constraints"]
    opt_times = results["opt_times"]
    itr_times = results["itr_times"]
    target_orientation_errors = results["target_orientation_errors"]
    target_position_errors = results["target_position_errors"]
    
    ####################################################################  
    ####################################################################  
    ####################################################################  

    # set font size and line width
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title
    plt.rc('lines', linewidth=3)   # line width 


    # Plot the trajectories# Plot the trajectories
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig3, ax3 = plt.subplots(figsize=(8, 4)) 

    positions = np.array(positions).reshape((simparams["n_steps"],3))
    orientations = np.array(orientations).reshape((simparams["n_steps"],3)) 
    slit_orientation_errors = np.array(slit_orientation_errors).flatten()
    total_energy = np.array(total_energy).flatten()
    kinetic_energy = np.array(kinetic_energy).flatten()
    potential_energy = np.array(potential_energy).flatten()
    time_array = np.array(time_array).flatten()*simparams["t_step"]

    # Plot distance of the drone to the target slit and the orientation error: 
    ax1.plot(time_array, slit_distances, label='distance')
    ax1.plot(time_array, slit_orientation_errors, label='orientation error')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('distance [m], error')
    # ax1.set_title('Distance and error vs time')
    ax1.legend()

    # Plot the distance vs error
    ax2.plot(slit_distances, slit_orientation_errors ) 
    ax2.set_xlabel('distance [m]')
    ax2.set_ylabel('error')
    # ax2.set_title('Error vs distance')
    ax2.legend()

    # Plot total_energy vs time
    ax3.plot(time_array, kinetic_energy, label='Kinetic')
    if not simparams["disable_nominal_control"]: 
        ax3.plot(time_array, potential_energy, label='Potential')
        ax3.plot(time_array, total_energy, label='Total' )
        ax3.set_ylim([0, 1.05*max(total_energy)])
    elif simparams["disable_nominal_control"] and fig_ready is not None:
        plot_path = os.path.join(PLOTS_PATH, paperid, fig_ready)
        print("Loading the previous plot ", plot_path)
        fig3 = pl.load(open(os.path.join(plot_path, "energy_vs_time.pickle"),'rb')) 
        ax3 = fig3.axes[0]
        x_old = ax3.lines[0].get_data()[0]
        y_old = ax3.lines[0].get_data()[1]
        ax3.plot(x_old, y_old) 
        ax3.plot(time_array, kinetic_energy, label='Kinetic')
        ax3.set_ylim([0, 3*max(kinetic_energy)])
    else:
        ax3.set_ylim([0, 3*max(kinetic_energy)])
        fig_ready = "openloop" if simparams["disable_mpc"] else "mpc"
    ax3.set_xlabel('time [s]')
    ax3.set_ylabel('Energy [J]')
    
    # ax3.set_title('Energy vs time')
    ax3.legend() 
    
    
    plt.tight_layout()
    # plt.show()
    
    # Save the plots 
    custom_name = results_file_name.split("results_paper_")[-1].split(".json")[0]
    plot_path = os.path.join(PLOTS_PATH, paperid, custom_name)
    os.makedirs(plot_path, exist_ok=True)
    fig1.savefig(os.path.join(plot_path, "distance_error_vs_time.png"))
    # fig2.savefig(os.path.join(plot_path, "distance_vs_error.png"))
    fig3.savefig(os.path.join(plot_path, "energy_vs_time.png")) 

    pl.dump(fig1, open(os.path.join(plot_path, "distance_error_vs_time.pickle"),'wb'))
    # pl.dump(fig2, open(os.path.join(plot_path, "distance_vs_error.pickle"),'wb'))
    pl.dump(fig3, open(os.path.join(plot_path, "energy_vs_time.pickle"),'wb'))



    ####################################################################  
    
    angles = [80, 70, 60, 50, 40, 30, 15, 0, -15, -30, -40, -50, -60, -70, -80] 
    for angle in angles:
        # Graphics
        slits = [dict(height=3, width=0.2, position=envparams["xpd"])]
        graphics = Drone3DStopMotion(skipframes=simparams["n_steps"]//20, lowerlimits=[-1,-1,-1], upperlimits=[1,6,1])
        graphics.add_slits(slits)
        graphics.add_drone(orientations, positions)
        graphics.viz(
            show=False, 
            save=True, 
            path=os.path.join(plot_path, f"3d_simulation_{angle}.png"), 
            camera=(10, angle)
        ) 