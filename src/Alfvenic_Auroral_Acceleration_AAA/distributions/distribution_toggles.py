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
    RK45_tspan = [0,-20*data_dict_ray_eqns['time'][0][-1]]  # time range (in seconds). MAKE SURE THIS IS REVERSED IN TIME

    #############################
    # --- OBSERVATION TOGGLES ---
    #############################

    # Observation Spatial Coordinate
    # Observation_altitudes = [500,1000,2000,2500,3000,4000,5000,6000,7000,7500,8000,9000,10000,11000,12000,12500,13000,14000,15000]
    Observation_altitudes = [1000]
    # Observation_altitudes = [1000,2000,4000, 6000,8000,10000,12000,14000]
    z0_obs = 3000  # in kilometers

    # Calculate the initial observation position in dipole coordinates
    Theta0_obs = RayEquationToggles.Theta0_w
    phi0_obs = RayEquationToggles.phi0_w
    r_obs = 1 + z0_obs / stl.Re
    chi0_obs = np.square(np.sin(np.radians(phi0_obs)))/1 # get the Chi0 point at the Earth's surface. Don't change Chi at all, only mu.
    u0_obs = (1- chi0_obs*r_obs)**(1/4) / r_obs # Solve for mu at the new altitude, given a constant Chi0
    phi0_obs = np.radians(phi0_obs)

    # ESA particle sampling
    time_rez = 0.25 # in seconds
    time_obs_end = 10 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1
    obs_times = np.linspace(0, time_obs_end, N_obs_points)

    # Observation Wave-Sampling
    time_rez_waves = 0.001 # in seconds
    N_obs_wave_points = int(time_obs_end/time_rez_waves)
    obs_waves_times = np.linspace(0,time_obs_end,N_obs_wave_points)

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

    #####################################
    # --- PLASMA DISTRIBUTION TOGGLES ---
    #####################################
    N_energy_space_points = 25

    # ENERGY/PITCH
    E_max = 4  # the POWER of 10^E_max for the maximum energy
    E_min = 1  # the POWER of 10^E_min for the minimum energy
    # pitch_range = np.linspace(0,180,19)
    pitch_range = np.array([0,45,90,110,120,180])
    energy_range = np.logspace(E_min, E_max, N_energy_space_points)

    # VELOCITY SPACE
    N_vel_space = 25
    # para_space_temp = np.linspace(SimClasses().to_Vel(10**(Emin_PS)), SimClasses().to_Vel(10**(Emax_PS)), N_vel_space)
    # v_para_space = np.append(-1 * para_space_temp[::-1], para_space_temp[1:])
    # v_perp_space = np.linspace(0, para_space_temp[-1], N_vel_space)

    ###########################
    # --- SIMULATION EXTENT ---
    ###########################

    # altitude to terminate simulation
    upper_termination_altitude = 40000  # [in km] use the maximum height of the wave reaches as an upper boundary for particles.
    lower_termination_altitude = 0

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


