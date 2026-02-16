import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
from glob import glob
data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/ray_equations/*.cdf*')[0])

class DistributionToggles:

    #############################
    # --- RK45 solver toggles ---
    #############################
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-8  # controls the relative accuracy. If rtol
    RK45_atol = 1E-9  # controls the absolute accuracy
    RK45_tspan = [0,-10*data_dict_ray_eqns['time'][0][-1]]  # time range (in seconds). MAKE SURE THIS IS REVERSED IN TIME

    #############################
    # --- OBSERVATION TOGGLES ---
    #############################

    # ENERGY/PITCH COORDINATES
    N_energy_space_points = 50
    E_max_obs = 4  # the POWER of 10^E_max for the maximum energy
    E_min_obs = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range_obs = np.linspace(0, 180, 19)
    # pitch_range_obs = [0,90]
    energy_range_obs = np.logspace(E_min_obs, E_max_obs, N_energy_space_points)

    # VELOCITY SPACE COORDINATES
    N_vel_space = 35
    para_space_temp = np.linspace(SimClasses().to_Vel(10**(E_min_obs)), SimClasses().to_Vel(10**(3.48)), N_vel_space)
    v_para_space_obs = np.append(-1 * para_space_temp[::-1], para_space_temp[1:])
    v_perp_space_obs = np.linspace(0, para_space_temp[-1], N_vel_space)

    # Observation Altitude
    Observation_altitudes = [500, 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12500,15000, 17500, 20000]
    # Observation_altitudes = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    # Observation_altitudes = [9500]
    z0_obs = 3000  # Doesn't matter what this is set to, it will be overwritten

    # ESA particle sampling
    time_rez = 0.05 # in seconds
    time_obs_start = 0  # in seconds
    time_obs_end = 10 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1
    obs_times = np.linspace(time_obs_start, time_obs_end, N_obs_points)

    # Observation Wave-Sampling
    time_rez_waves = 0.001 # in seconds
    N_obs_wave_points = int(time_obs_end/time_rez_waves)
    obs_waves_times = np.linspace(0,time_obs_end,N_obs_wave_points)

    ###########################
    # --- SIMULATION EXTENT ---
    ###########################

    # altitude to terminate simulation
    upper_termination_altitude = 50000  # [in km] use the maximum height of the wave reaches as an upper boundary for particles.
    lower_termination_altitude = 300

    #################################
    # --- PLASMA SHEET PARAMETERS ---
    #################################
    n_PS = 0.25E6 # in [m^-3]
    Te_PS = 120 # in [eV]
    Emax_PS = 4000 # in [eV]. maximum energy in the plasma sheet distribution
    Emin_PS = 10  # in [eV]. maximum energy in the plasma sheet distribution
    alpha_min = 2.5 # minimum pitch angle of particles in source distribution

    #################################
    # --- IONOSPHERE PARAMETERS ---
    #################################
    n_iono = 1E10  # in [m^-3]
    Te_iono = 1  # in [eV]
    Emax_iono = 10  # in [eV]. maximum energy in the plasma sheet distribution
    Emin_iono = 0  # in [eV]. maximum energy in the plasma sheet distribution
    # alpha_min = 2.5  # minimum pitch angle of particles in source distribution

    ###########################
    # --- SOME CALCULATIONS ---
    ###########################

    # Calculate the initial observation position in dipole coordinates
    Theta0_obs = RayEquationToggles.Theta0_w
    phi0_obs = RayEquationToggles.phi0_w
    r_obs = 1 + z0_obs / stl.Re
    chi0_obs = np.square(np.sin(np.radians(phi0_obs))) / 1  # get the Chi0 point at the Earth's surface. Don't change Chi at all, only mu.
    u0_obs = (1 - chi0_obs * r_obs) ** (1 / 4) / r_obs  # Solve for mu at the new altitude, given a constant Chi0
    phi0_obs = np.radians(phi0_obs)

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


