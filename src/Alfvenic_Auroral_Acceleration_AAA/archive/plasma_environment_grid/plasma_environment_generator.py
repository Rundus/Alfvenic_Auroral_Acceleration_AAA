
def generate_plasma_environment():

    # --- imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from glob import glob

    # import the toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.plasma_environment_grid.plasma_environment_toggles import PlasmaToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.spatial_environment_grid.spatial_environment_toggles import SpatialToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.geomagnetic_field_grid.geomagnetic_field_toggles import GeomagneticToggles

    # import the physics models
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.plasma_environment_grid.plasma_environment_classes import ni,ion_composition,Ti

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{GeomagneticToggles.outputFolder}\*.cdf*')[0])
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SpatialToggles.outputFolder}\*.cdf*')[0])

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################

    # Get the spatial dimensions
    chi = deepcopy(data_dict_spatial['chi'][0])
    mu = deepcopy(data_dict_spatial['mu'][0])
    alt = deepcopy(data_dict_spatial['alt'][0])

    data_dict_output = {
        'Te': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'eV', 'LABLAXIS': 'Te', 'VAR_TYPE':'data'}],
        'Ti': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'eV', 'LABLAXIS': 'Ti', 'VAR_TYPE':'data'}],

        'n_total': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'Plasma Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],

        'n_O+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'O+ Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],
        'n_H+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'H+ Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],

        'C_O+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'O+ Concentration', 'LABLAXIS': None, 'VAR_TYPE':'data'}],
        'C_H+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'H+ Concentration', 'LABLAXIS': None, 'VAR_TYPE':'data'}],

        'Omega_O+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'O+ Larmor Freq.', 'VAR_TYPE':'data'}],
        'Omega_H+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'H+ Larmor Freq.', 'VAR_TYPE':'data'}],
        'Omega_e': [np.zeros(shape=(len(mu), len(chi))),{'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'electron larmor Freq.', 'VAR_TYPE': 'data'}],

        'larmor_O+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'm', 'LABLAXIS': 'O+ Larmor Radius', 'VAR_TYPE':'data'}],
        'larmor_H+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'm', 'LABLAXIS': 'O+ Larmor Radius', 'VAR_TYPE':'data'}],

        'm_eff_i': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i', 'VAR_TYPE':'data'}],
        'Omega_eff_i': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'Omega_eff_i', 'VAR_TYPE': 'data'}],
        'larmor_eff_i': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm', 'LABLAXIS': 'rho_eff_i', 'VAR_TYPE': 'data'}],

        'rho_m': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'kg/m!A3', 'LABLAXIS': 'mass density', 'VAR_TYPE': 'data'}],

        'omega_pe': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'Electron Plasma Freq', 'VAR_TYPE': 'data'}],

        'lambda_e': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm', 'LABLAXIS': 'Electron Skin Depth', 'VAR_TYPE': 'data'}],

        'vth_i': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm/s', 'LABLAXIS': 'ion_thermal_velocity', 'VAR_TYPE': 'data'}],

        'V_A': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm/s', 'LABLAXIS':'MHD Alfven Velocity', 'VAR_TYPE': 'data'}],

        'dLbda_du':[np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/d&mu;', 'VAR_TYPE': 'data'}],
        'dLbda_dX': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/&Chi;', 'VAR_TYPE': 'data'}],

        'dVA_du': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm/s', 'LABLAXIS': 'dV!BA!N/d&mu;', 'VAR_TYPE': 'data'}],
        'dVA_dX': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm/s', 'LABLAXIS': 'dV!BA!N/d&Chi;', 'VAR_TYPE': 'data'}],
    }

    #######################
    # --- DENSITY MODEL ---
    #######################
    model = ni().Chaston2002
    n_O, n_H = model(alt)
    data_dict_output['n_total'][0] = n_O + n_H
    data_dict_output['n_O+'][0] = n_O
    data_dict_output['n_H+'][0] = n_H

    ############################
    # --- ION CONCENTRATIONS ---
    ############################
    model = ion_composition().Chaston2006
    n_O_n_i_ratio = model(alt)
    data_dict_output['C_O+'][0] = n_O_n_i_ratio
    data_dict_output['C_H+'][0] = 1 - n_O_n_i_ratio

    ##################
    # --- ION MASS ---
    ##################
    # get the effective mass based on the IRI
    m_Hp = stl.ion_dict['H+']
    m_Op = stl.ion_dict['O+']

    data_dict_output['m_eff_i'][0] = m_Hp*data_dict_output['C_H+'][0] + m_Op*data_dict_output['C_O+'][0]

    #####################
    # --- TEMPERATURE ---
    #####################
    model = Ti().shroeder2021
    data_dict_output['Ti'][0] = model(alt)
    data_dict_output['Te'][0] = model(alt)

    #####################
    # --- PLASMA BETA ---
    #####################
    plasmaBeta = (2 * stl.u0 *stl.kB)*(data_dict_output['n_total'][0] * data_dict_output['Te'][0] ) / np.power(data_dict_Bgeo['Bgeo'][0],2)
    data_dict_output = {**data_dict_output, **{'beta_e': [plasmaBeta, {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta_e'}]}}

    ##########################
    # --- PLASMA FREQUENCY ---
    ##########################
    plasmaFreq = np.sqrt(data_dict_output['n_total'][0] * (stl.q0 * stl.q0) / (stl.ep0 * data_dict_output['m_eff_i'][0]))
    data_dict_output = {**data_dict_output, **{'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}]}}

    ############################
    # --- ION CYCLOTRON FREQ ---
    ############################
    data_dict_output['Omega_O+'][0] = stl.q0 * deepcopy(data_dict_Bgeo['Bgeo'][0]) / m_Op
    data_dict_output['Omega_H+'][0] = stl.q0 * deepcopy(data_dict_Bgeo['Bgeo'][0]) / m_Hp
    data_dict_output['Omega_eff_i'][0] = stl.q0 * deepcopy(data_dict_Bgeo['Bgeo'][0]) / deepcopy(data_dict_output['m_eff_i'][0])
    data_dict_output['Omega_e'][0] = stl.q0 * deepcopy(data_dict_Bgeo['Bgeo'][0]) / stl.m_e

    ###########################
    # --- ION LARMOR RADIUS ---
    ###########################
    data_dict_output['vth_i'][0] = np.sqrt(2 * stl.q0*deepcopy(data_dict_output['Ti'][0]) / deepcopy(data_dict_output['m_eff_i'][0])) # the np.sqrt(2) comes from the vector sum of two dimensions
    data_dict_output['larmor_O+'][0] = deepcopy(data_dict_output['vth_i'][0]) / deepcopy(data_dict_output['Omega_O+'][0])
    data_dict_output['larmor_H+'][0] = deepcopy(data_dict_output['vth_i'][0]) / deepcopy(data_dict_output['Omega_H+'][0])
    data_dict_output['larmor_eff_i'][0] = deepcopy(data_dict_output['vth_i'][0]) / deepcopy(data_dict_output['Omega_eff_i'][0])

    ######################
    # --- MASS DENSITY ---
    ######################
    data_dict_output['rho_m'][0] = m_Hp * deepcopy(data_dict_output['n_H+'][0]) + m_Op * deepcopy(data_dict_output['n_O+'][0])

    ######################
    # --- ALFVEN SPEED ---
    ######################
    data_dict_output['V_A'][0] = data_dict_Bgeo['Bgeo'][0]/np.sqrt(stl.u0*deepcopy(data_dict_output['rho_m'][0]))

    ####################
    # --- SKIN DEPTH ---
    ####################
    data_dict_output['omega_pe'][0] = np.sqrt((stl.q0*stl.q0*data_dict_output['n_total'][0])/(stl.m_e*stl.ep0))
    data_dict_output['lambda_e'][0] = stl.lightSpeed/deepcopy(data_dict_output['omega_pe'][0])

    ###################
    # --- GRADIENTS ---
    ###################

    # mu gradients
    for i in range(len(chi)):
        data_dict_output['dLbda_du'][0][: ,i] = np.gradient(deepcopy(data_dict_output['lambda_e'][0][:,i]), mu)
        data_dict_output['dVA_du'][0][:, i] = np.gradient(deepcopy(data_dict_output['V_A'][0][:,i]), mu)

    # Chi Gradients
    for i in range(len(mu)):
        data_dict_output['dLbda_dX'][0][i] = np.gradient(deepcopy(data_dict_output['lambda_e'][0][i]),chi)
        data_dict_output['dVA_dX'][0][i] = np.gradient(deepcopy(data_dict_output['V_A'][0][i]), chi)

    #####################
    # --- OUTPUT DATA ---
    #####################
    for key in data_dict_spatial.keys():
        data_dict_spatial[key][1]['VAR_TYPE'] = 'support_data'
    data_dict_output = {**data_dict_spatial, **data_dict_output}

    outputPath = rf'{PlasmaToggles.outputFolder}\plasma_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
