import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
class DistributionToggles:

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
    vmin = -1E6
    vmax = 1E6

    # mu
    vel_mu = np.linspace(vmin,vmax,N_points)

    # chi
    vel_chi = np.linspace(vmin,vmax,N_points)

    # phi
    v_phi = np.linspace(vmin,vmax,N_points)

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/distributions'


