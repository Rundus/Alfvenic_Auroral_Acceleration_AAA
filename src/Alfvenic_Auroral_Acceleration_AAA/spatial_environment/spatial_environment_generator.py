def generate_spatial_environment():
    # --- imports ---
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # import the toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.spatial_environment.spatial_environment_toggles import SpatialToggles

    # prepare the output
    spatial_grid_dim = np.zeros(shape=(len(SpatialToggles.mu_range),len(SpatialToggles.chi_range)))
    data_dict_output = {
                        'mu': [SpatialToggles.mu_range, {'UNITS': None, 'LABLAXIS': '&mu;','VAR_TYPE':'data'}],
                        'chi': [SpatialToggles.chi_range, {'UNITS': None, 'LABLAXIS': '&chi;','VAR_TYPE':'data'}],
                        'phi': [SpatialToggles.phi_range, {'UNITS': 'deg', 'LABLAXIS': '&phi;','VAR_TYPE':'data'}],
                        'colat': [np.zeros(shape=np.shape(spatial_grid_dim)), {'DEPEND_0': 'mu','DEPEND_1':'chi', 'UNITS': 'deg', 'LABLAXIS': 'colatitude','VAR_TYPE':'data'}],
                        'alt': [np.zeros(shape=np.shape(spatial_grid_dim)), {'DEPEND_0': 'mu','DEPEND_1':'chi','UNITS': 'm', 'LABLAXIS': 'altitude','VAR_TYPE':'data'}],
                        'radius': [np.zeros(shape=np.shape(spatial_grid_dim)), {'DEPEND_0': 'mu', 'DEPEND_1': 'chi', 'UNITS': 'm', 'LABLAXIS': 'Radius from Earth Center', 'VAR_TYPE': 'data'}],
                        'lat': [np.zeros(shape=np.shape(spatial_grid_dim)), {'DEPEND_0': 'mu','DEPEND_1':'chi', 'UNITS': 'deg', 'LABLAXIS': 'latitude','VAR_TYPE':'data'}],
                        'long': [np.full(np.shape(spatial_grid_dim),SpatialToggles.phi_range[0]), {'DEPEND_0': 'mu','DEPEND_1':'chi', 'UNITS': 'deg', 'LABLAXIS': 'longitude','VAR_TYPE':'data'}],
                        }

    ########################################################
    # --- Calculate lat/long/alt from dipole coordinates ---
    ########################################################

    # establish a simulation grid
    chi_grid, mu_grid = np.meshgrid(data_dict_output['chi'][0],data_dict_output['mu'][0])

    # calculate the zeta, gamma, c1, c2 and w terms
    zeta = np.power(mu_grid/chi_grid,4)
    c1 = (2**(7/3))*(3**(-1/3))
    c2 = (2**(1/3))*(3**(2/3))
    gamma = (9*zeta + np.sqrt(3)*np.sqrt(27*np.power(zeta,2) + 256*np.power(zeta,3)))**(1/3)
    w = -c1/gamma + gamma/(c2*zeta)

    # calculate u term
    u = -0.5*np.sqrt(w) + 0.5*np.sqrt( -w + (2/zeta)*(1/np.sqrt(w))  )

    # calculate colatitude
    theta = np.degrees(np.arcsin(np.sqrt(u)))
    data_dict_output['colat'][0] = deepcopy(theta)

    # calculate distance from Earth's center
    r_alt = u/chi_grid
    data_dict_output['radius'][0] = deepcopy(stl.m_to_km*r_alt*stl.Re)
    data_dict_output['alt'][0] = deepcopy(data_dict_output['radius'][0]) - stl.Re*stl.m_to_km

    # calculate the latitude
    data_dict_output['lat'][0] = 90-deepcopy(data_dict_output['colat'][0])

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{SpatialToggles.outputFolder}\spatial_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)