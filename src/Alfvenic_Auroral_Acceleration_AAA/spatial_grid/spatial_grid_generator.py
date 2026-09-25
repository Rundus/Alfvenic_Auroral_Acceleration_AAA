from timebudget import timebudget
@timebudget

def spatial_grid_generator():

    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import SpatialGridToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.spatial_grid.spatial_classes import SpatialClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
    import numpy as np
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    import spaceToolsLib as stl
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

    ########################################
    # CONSTRUCT THE GRIDDED SIMULATION SPACE
    ########################################
    # Description: Construct a mu-grid between (s_para_min, s_para_max),
    # which denote the distance along a geomagnetic field line

    # --- Get the initial mu,chi ---
    r0 = (1 + SpatialGridToggles.s_para_min/stl.Re) # [Re] initial radial distance
    colat0 = 90 - np.degrees(np.arccos(np.sqrt(r0 / SpatialGridToggles.L_Shell)))  # [deg] Colatitude
    u0 = -1 * np.sqrt(np.cos(np.radians(colat0))) / r0  # [n/a] Initial mu0 point. Lower end of simulation
    chi0 = np.power(np.sin(np.radians(colat0)), 2) / r0 # [n/a] Initial chi point.

    # --- Get the final mu, chi points ---
    uf = SpatialClasses().mu_from_field_line_distance(u0,chi0, SpatialGridToggles.s_para_max)

    # --- create the simulation grids ---
    # Calculate the modified dipole coordinates
    u_space = np.linspace(u0, uf, SpatialGridToggles.N_mu)
    chi_space = [chi0 for i in range(SpatialGridToggles.N_mu)]
    r_space = SpatialClasses().r_muChi(u_space, chi_space)
    theta_space = SpatialClasses().theta_muChi(u_space, chi_space)
    alt_space = (r_space-1)*stl.m_to_km*stl.Re
    h_mu = envDict['h_mu'](u_space, np.array(chi_space))
    s_para = np.concatenate([[0.0], np.cumsum(0.5*(h_mu[1:] + h_mu[:-1])*np.diff(u_space))])

    # prepare the output
    data_dict_output = {
        'colat': [theta_space, {'UNITS':'deg','LABLAXIS':'Colatitude'}],
        'mu': [u_space, {'UNITS':None,'LABLAXIS':'&mu;'}],
        'chi': [chi_space, {'UNITS':None,'LABLAXIS':'&chi;'}],
        'r':[r_space, {'UNITS':'Re','LABLAXIS':'Distance from Earth Center'}],
        'alt': [alt_space, {'UNITS': 'm', 'LABLAXIS': 'altitude'}],
        'S_para':[s_para,{'UNITS':'m','LABLAXIS':'Field-aligned Distance'}]
    }

    if RunToggles.store_output:
        outputPath = rf'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf'
        stl.outputDataDict(outputPath, data_dict_output)