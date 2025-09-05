def scale_length_RK45_generator():
    # --- imports ---
    import spaceToolsLib as stl
    import numpy as np

    # import the toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.spatial_environment_grid.spatial_environment_toggles import SpatialToggles

    # prepare the output
    spatial_grid_dim = np.zeros(shape=(len(SpatialToggles.mu_range),len(SpatialToggles.chi_range)))
    data_dict_output = {
                        'mu_w': [[], {'UNITS': None, 'LABLAXIS': '&mu;','VAR_TYPE':'data'}],
                        'chi_w': [[], {'UNITS': None, 'LABLAXIS': '&chi;','VAR_TYPE':'data'}],
                        'phi_w': [[], {'UNITS': 'deg', 'LABLAXIS': '&phi;','VAR_TYPE':'data'}],
                        'k_perp': [[], {'UNITS': '1/m', 'LABLAXIS': 'k!B;&perp', 'VAR_TYPE': 'data'}],
                        'k_para': [[], {'UNITS': '1/m', 'LABLAXIS': 'k!B;&parallel', 'VAR_TYPE': 'data'}],
                        }



    ########################################################
    # --- Calculate lat/long/alt from dipole coordinates ---
    ########################################################

    # establish a simulation grid
    chi_grid, mu_grid = np.meshgrid(data_dict_output['chi'][0],data_dict_output['mu'][0])


    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{SpatialToggles.outputFolder}\spatial_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)