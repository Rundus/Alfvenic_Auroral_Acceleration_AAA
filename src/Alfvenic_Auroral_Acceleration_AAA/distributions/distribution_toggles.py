import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
class DistributionToggles:

    #################################
    # --- PLASMA SHEET PARAMETERS ---
    #################################
    n_PS = 100E6 # in [m^-3]
    Te_PS = 100 # in [eV]

    ##########################
    # --- INITIAL POSITION ---
    ##########################
    z0 = SimToggles.z0  # in kilometers
    Theta0 = SimToggles.Theta0  # in geomagnetic latitude
    phi0 = SimToggles.phi0  # in geomagnetic longitude

    # modified dipole coordinates
    r = 1 + z0 / stl.Re
    u0 = -1 * np.sqrt(np.cos(np.radians(90 - Theta0))) / r
    chi0 = np.power(np.sin(np.radians(90 - Theta0)), 2) / r
    phi0 = np.radians(phi0)

    #####################################
    # --- PLASMA DISTRIBUTION TOGGLES ---
    #####################################
    N_vel_space_points = 10


    # LINEAR
    # E_lim = 10000 # in eV
    # vmin = -np.sqrt(2*(stl.q0*E_lim)/stl.m_e) # define the maximum velocity in terms of energy
    # vmax = np.sqrt(2*(stl.q0*E_lim)/stl.m_e)
    #
    # # mu
    # vel_space_mu_range = np.linspace(vmin, 0, N_points) # only upward particles
    #
    # # perp
    # vel_space_perp_range = np.linspace(np.sqrt(2*(stl.q0*10)/stl.m_e), vmax, N_vel_space_points)

    # LOG
    E_max = 3# the POWER of 10^E_max for the maximum energy
    E_min = 1 # the POWER of 10^E_min for the minimum energy
    vel_space_log = np.sqrt(2*stl.q0*np.logspace(E_min, E_max, N_vel_space_points)/stl.m_e)
    vel_space_mu_range = np.append(-1*np.flip(vel_space_log),vel_space_log)
    vel_space_perp_range = vel_space_log


    ###########################
    # --- SIMULATION EXTENT ---
    ###########################

    # altitude to terminate simulation
    upper_termination_altitude = 10000 * stl.m_to_km # in meters
    lower_termination_altitude = 100 * stl.m_to_km

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


