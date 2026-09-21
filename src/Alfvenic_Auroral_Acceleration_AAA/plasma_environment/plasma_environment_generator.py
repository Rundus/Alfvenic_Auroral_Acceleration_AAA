from timebudget import timebudget


@timebudget
def plasma_environment_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import math

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_toggles import PlasmaEnvironmentToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_classes import PlasmaEnvironmentClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles

    # --- Load the wave runners data ---
    data_dict_spatial = stl.loadDictFromFile(glob(rf'{RunToggles.sim_data_output_path}//spatial_grid/*.cdf*')[0])

    # prepare the output
    data_dict_output = {
                        'V_A':[[],{'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'Alfven Speed (MHD)', 'VAR_TYPE': 'data'}],
                        'n_density_cold': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-3', 'LABLAXIS': 'Plasma Density', 'VAR_TYPE': 'data'}],
                        'n_density_PS': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-3', 'LABLAXIS': 'Plasma Density', 'VAR_TYPE': 'data'}],
                        'm_i': [[], {'DEPEND_0': 'alt', 'UNITS': 'kg', 'LABLAXIS': 'Alfven Speed (MHD)', 'VAR_TYPE': 'data'}],
                        'lambda_e': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'Electron Skin Depth', 'VAR_TYPE': 'data'}],
                        'pDD_lambda_e_mu': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/d&mu;', 'VAR_TYPE': 'data'}],
                        'pDD_lambda_e_chi': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/d&chi;', 'VAR_TYPE': 'data'}],
                        'pDD_V_A_mu': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'dV_A/d&mu;', 'VAR_TYPE': 'data'}],
                        'pDD_V_A_alt': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s/m', 'LABLAXIS': 'dV_A/d&z;', 'VAR_TYPE': 'data'}],
                        'pDD_V_A_chi': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'dV_A/d&chi;', 'VAR_TYPE': 'data'}],
                        'dB_dipole_dmu': [[], {'DEPEND_0': 'alt','UNITS': 'T','LABLAXIS':'dB_dipole_dmu', 'VAR_TYPE':'data'}],
                        'h_mu': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'h!B&mu;!N', 'VAR_TYPE': 'data'}],
                        'h_chi': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'h!B&chi;!N', 'VAR_TYPE': 'data'}],
                        'h_phi': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'h!B&phi;!N', 'VAR_TYPE': 'data'}],
                        'B_dipole':[[],{'DEPEND_0': 'alt', 'UNITS': 'nT', 'LABLAXIS': '|B|', 'VAR_TYPE': 'data'}],
                        'meff': [[], {'DEPEND_0': 'alt', 'UNITS': 'kg', 'LABLAXIS': 'mass (avg)', 'VAR_TYPE': 'data'}],
                        'pDD_n_density_mu': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-3', 'LABLAXIS': '(dn/d&mu;)', 'VAR_TYPE': 'data'}],
                        'pDD_n_density_alt': [[], {'DEPEND_0': 'alt','UNITS': 'm^-3 / m','LABLAXIS':'dn/dz', 'VAR_TYPE':'data'}],
                        'WKB_density_scale_length': [[],{'DEPEND_0': 'alt','UNITS': 'm','LABLAXIS':'n/ (dn/dz)', 'VAR_TYPE':'data'}],
                        'WKB_VA_scale_length': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': 'V!BA!N/ (dV!BA!N/dz)', 'VAR_TYPE': 'data'}],
                        'n_Op': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-3', 'LABLAXIS': 'O+ Density', 'VAR_TYPE': 'data'}],
                        'n_Hp': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-3', 'LABLAXIS': 'H+ Density', 'VAR_TYPE': 'data'}],
                        'rho': [[], {'DEPEND_0': 'alt', 'UNITS': 'kg m!A-3', 'LABLAXIS': 'Avg. Mass Density', 'VAR_TYPE': 'data'}],
                        'alt':data_dict_spatial['alt'].copy(),
                        'Te_cold':[[], {'DEPEND_0': 'alt', 'UNITS': 'eV', 'LABLAXIS': 'T!Be!N', 'VAR_TYPE': 'data'}],
                        'Te_PS': [[], {'DEPEND_0': 'alt', 'UNITS': 'eV', 'LABLAXIS': 'T!Be!N', 'VAR_TYPE': 'data'}],
                        'loss_cone': [[], {'DEPEND_0': 'alt', 'UNITS': 'degrees', 'LABLAXIS': '&alpha;!BL!N', 'VAR_TYPE': 'data'}],
                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################
    for key, func in envDict.items():
        data_dict_output[key][0] = func(data_dict_spatial['mu'][0], data_dict_spatial['chi'][0])

    ##########################################
    # CONSTRUCT THE WKB EVALUATION VARIABLES #
    ##########################################
    data_dict_output['pDD_n_density_alt'][0] = np.gradient(data_dict_output['n_density_cold'][0].copy(),stl.m_to_km*data_dict_spatial['alt'][0].copy())
    data_dict_output['WKB_density_scale_length'][0] = data_dict_output['n_density_cold'][0] / data_dict_output['pDD_n_density_alt'][0]
    data_dict_output['pDD_V_A_alt'][0] = np.gradient(data_dict_output['V_A'][0].copy(), stl.m_to_km*data_dict_spatial['alt'][0].copy())
    data_dict_output['WKB_VA_scale_length'][0] = data_dict_output['V_A'][0] / data_dict_output['pDD_V_A_alt'][0]

    ##########################################
    # CALCULATE THE TEMPERATURE VARIABLES #
    ##########################################

    data_dict_output['Te_cold'][0] = np.array([PlasmaEnvironmentToggles.Te_cold for i in range(len(data_dict_output['alt'][0]))])
    data_dict_output['Te_PS'][0] = np.array([PlasmaEnvironmentToggles.Te_PS for i in range(len(data_dict_output['alt'][0]))])

    # HOT PLASMA SHEET DENSITY
    plasma_environ_object = PlasmaEnvironmentClasses()
    data_dict_output['loss_cone'][0] = plasma_environ_object.loss_cone_angle(data_dict_spatial['mu'][0], data_dict_spatial['chi'][0])

    # Use loss cone angle to get hot plasma density from zero-th moment of plasma distribution
    data_dict_output['n_density_PS'][0] = plasma_environ_object.n_density_PS_loss_cone(data_dict_output['loss_cone'][0])

    ################
    # --- OUTPUT ---
    ################

    if RunToggles.store_output:
        outputPath = rf'{RunToggles.sim_data_output_path}/plasma_environment/plasma_environment.cdf'
        stl.outputDataDict(outputPath, data_dict_output)