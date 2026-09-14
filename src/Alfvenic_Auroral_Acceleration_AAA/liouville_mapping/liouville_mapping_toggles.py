import numpy as np
import spaceToolsLib as stl
import os

class LiouvilleToggles:

    #############################
    # --- PARALLEL PROCESSING ---
    #############################
    processes_count = os.cpu_count()-1  # Number of CPU cores to commit to this operation
    # processes_count = 20  # Number of CPU cores to commit to this operation

    #############################
    # --- RK45 solver toggles ---
    #############################
    RK45_method = 'RK45' # 'LSODA'
    RK45_rtol = 1E-8  # controls the relative accuracy.
    RK45_atol = 1E-9  # controls the absolute accuracy
    # RK45_tspan = [0,-10*data_dict_ray_eqns['time'][0][-1]]  # time range (in seconds). MAKE SURE THIS IS REVERSED IN TIME

    #############################
    # --- OBSERVATION TOGGLES ---
    #############################

    # --- PHYSICAL TOGGLES ---
    mapping_alts = [500, 1000,1500]  # [km] This is the altitude where the Louisville mapping is measured
    upper_termination_altitude = 18000 # [km] upper altitude limit where to stop the Rk45 solver
    lower_termination_altitude = 200 # [km] lower altitude limit where to stop the Rk45

    # --- ENERGY/PITCH COORDINATES ---
    N_energy_space_points = 50
    E_max_obs = 4  # the POWER of 10^E_max for the maximum energy
    E_min_obs = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range_obs = np.linspace(0, 180, 19)
    energy_range_obs = np.logspace(E_min_obs, E_max_obs, N_energy_space_points)

    # --- ESA particle sampling ---
    time_rez = 0.05 # in seconds
    time_obs_start = 0  # in seconds
    time_obs_end = 3 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1
    obs_times = np.linspace(time_obs_start, time_obs_end, N_obs_points)

    # --- Observation Wave-Sampling ---
    time_rez_waves = 0.001 # in seconds
    N_obs_wave_points = int(time_obs_end/time_rez_waves)
    obs_waves_times = np.linspace(0,time_obs_end,N_obs_wave_points)




