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
    N_points = 10
    E_lim = 10 # in eV
    vmin = -np.sqrt(2*(stl.q0*E_lim)/stl.m_e) # define the maximum velocity in terms of energy
    vmax = np.sqrt(2*(stl.q0*E_lim)/stl.m_e)

    # mu
    vel_space_mu_range = np.linspace(vmin, vmax, N_points)

    # perp
    vel_space_perp_range = np.linspace(vmin, vmax, N_points)

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


