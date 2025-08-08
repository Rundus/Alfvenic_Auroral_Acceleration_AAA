class SpatialToggles:
    import datetime as dt
    import numpy as np
    import spaceToolsLib as stl
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles

    # --- Spatial Grid Description ---
    # the "modified" dipole coordinates described by [kageyama 2006] are used to define a spatial grid.

    # DEFINE SIMULATION EXTENT in terms of geophysical parameters
    L_Shell = 8.5
    delta_colat = 0.25
    r_min = (1.08)*stl.Re # in [km] from Earth's Center
    r_max = 3*stl.Re # in [km] from Earth's Center

    #######################
    # --- CHI-Dimension ---
    #######################

    N_chi = 10 # number of points in chi direction

    # determine the APPROXIMATE colat this L-shell corresponds to. Technically McIIwain gives geomagnetic lat, not lat.
    theta_L_choice = 90 - np.degrees(np.arccos(np.sqrt(1/L_Shell)))

    # construct evenly spaced Chi grid, starting at Earth's surface
    colat_min, colat_max = theta_L_choice-delta_colat,theta_L_choice+delta_colat
    chi_low, chi_high = np.power(np.sin(np.radians(colat_min)),2)/1, np.power(np.sin(np.radians(colat_max)),2)/1
    chi_range = np.linspace(chi_low, chi_high,N_chi)

    ######################
    # --- MU-Dimension ---
    ######################
    N_mu = 1000 # number of points in mu direction

    # determine minimum/maximum mu value for the TOP colattitude
    mu_min, mu_max = -1*np.sqrt(np.cos(np.radians(colat_min)))/(r_min/stl.Re),-1*np.sqrt(np.cos(np.radians(colat_min)))/(r_max/stl.Re)
    mu_range = np.linspace(mu_min,mu_max,N_mu)

    #######################
    # --- PHI-DIMENSION ---
    #######################
    phi_range = np.array([16]) # just choose Andoya's longitude (in deg). It does not matter what you pick

    # --- File I/O ---
    target_time = dt.datetime(2022,11,20,17,20)
    outputFolder = f'{SimToggles.sim_root_path}\spatial_environment'