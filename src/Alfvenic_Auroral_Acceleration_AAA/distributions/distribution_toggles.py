import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles

class DistributionToggles:

    # --- RK45 solver toggles ---
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-6  # controls the relative accuracy. If rtol
    RK45_atol = 1E-7  # controls the absolute accuracy
    RK45_N_eval_points = 30
    RK45_Teval = np.linspace(SimToggles.RK45_tspan[0], SimToggles.RK45_tspan[-1], RK45_N_eval_points)

    ########################################
    # --- OBSERVATION INITIAL CONDITIONS ---
    ########################################
    z0_obs = 500  # in kilometers
    Theta0_obs = RayEquationToggles.Theta0_w
    phi0_obs = RayEquationToggles.phi0_w
    r_obs = 1 + z0_obs / stl.Re
    u0_obs = -1 * np.sqrt(np.cos(np.radians(90 - Theta0_obs))) / r_obs
    chi0_obs = np.power(np.sin(np.radians(90 - Theta0_obs)), 2) / r_obs
    phi0_obs = np.radians(phi0_obs)

    #################################
    # --- PLASMA SHEET PARAMETERS ---
    #################################
    n_PS = 100E6 # in [m^-3]
    Te_PS = 100 # in [eV]

    #####################################
    # --- PLASMA DISTRIBUTION TOGGLES ---
    #####################################
    N_energy_space_points = 30

    # ENERGY/PITCH
    E_max = 4  # the POWER of 10^E_max for the maximum energy
    E_min = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range = np.linspace(0,180,19)
    energy_range = np.logspace(E_min,E_max,N_energy_space_points)


    ###########################
    # --- SIMULATION EXTENT ---
    ###########################

    # altitude to terminate simulation
    upper_termination_altitude = 10000 * stl.m_to_km # in meters
    lower_termination_altitude = 100 * stl.m_to_km

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


