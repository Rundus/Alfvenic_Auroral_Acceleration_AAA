import numpy as np
import spaceToolsLib as stl

class PlasmaEnvironmentToggles:


    ####################################
    #-----------------------------------
    # --- GRID TOGGLES (DIAGNOSTICS) ---
    # -----------------------------------
    ####################################

    ##################################
    # --- SPATIAL ENVIRONMENT GRID ---
    ##################################
    # DEFINE SIMULATION EXTENT in terms of geophysical parameters
    L_Shell = 8.5
    delta_colat = 0.25
    r_min = (1.08) * stl.Re  # in [km] from Earth's Center
    r_max = 3 * stl.Re  # in [km] from Earth's Center

    #######################
    # --- CHI-Dimension ---
    #######################

    N_chi = 10  # number of points in chi direction

    # determine the APPROXIMATE colat this L-shell corresponds to. Technically McIIwain gives geomagnetic lat, not lat.
    theta_L_choice = 90 - np.degrees(np.arccos(np.sqrt(1 / L_Shell)))

    # construct evenly spaced Chi grid, starting at Earth's surface
    colat_min, colat_max = theta_L_choice - delta_colat, theta_L_choice + delta_colat
    chi_low, chi_high = np.power(np.sin(np.radians(colat_min)), 2) / 1, np.power(np.sin(np.radians(colat_max)), 2) / 1
    chi_range = np.linspace(chi_low, chi_high, N_chi)

    ######################
    # --- MU-Dimension ---
    ######################
    N_mu = 1000  # number of points in mu direction

    # determine minimum/maximum mu value for the TOP colattitude
    mu_min, mu_max = -1 * np.sqrt(np.cos(np.radians(colat_min))) / (r_min / stl.Re), -1 * np.sqrt(np.cos(np.radians(colat_min))) / (r_max / stl.Re)
    mu_range = np.linspace(mu_min, mu_max, N_mu)

    # CONSTRUCT THE GRID
    chi_grid, mu_grid = np.meshgrid(chi_range, mu_range)

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/plasma_environment'


