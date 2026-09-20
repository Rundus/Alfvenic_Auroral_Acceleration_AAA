import numpy as np
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

    ##############################
    # --- PARTICLE OBSERVATION ---
    ##############################

    # --- PHYSICAL TOGGLES ---
    mapping_alts = [6000]  # [km] This is the altitude where the Louisville mapping is measured
    upper_termination_altitude = 18000 # [km] upper altitude limit where to stop the Rk45 solver
    lower_termination_altitude = 100 # [km] lower altitude limit where to stop the Rk45

    # --- ENERGY/PITCH COORDINATES ---
    N_energy_space_points = 30
    E_max_obs = 4  # the POWER of 10^E_max for the maximum energy
    E_min_obs = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range_obs = np.linspace(0, 180, 9+1)
    # pitch_range_obs = np.linspace(0,90,9+1)
    energy_range_obs = np.logspace(E_min_obs, E_max_obs, N_energy_space_points)

    # --- ESA particle sampling ---
    time_rez = 0.25 # in seconds
    time_obs_start = 0  # in seconds
    time_obs_end = 4 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1 # number of particle observation points

    # --- Loss Cone ---
    use_loss_cone_bool = True

    ###########################
    # --- WAVE OBSERVATIONS ---
    ###########################

    # --- Observation Wave-Sampling ---
    time_rez_waves = 0.001 # [seconds] deltaT sample rate for the waves

    # --- Injected Wave ---
    injected_wave_time_delay = 5 # [seconds] Time delay added to the wave data to cause it to inject later. Defaults to 0 if set to <=0





