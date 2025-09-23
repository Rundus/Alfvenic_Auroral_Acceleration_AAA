def plasma_environment_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_toggles import PlasmaEnvironmentToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses

    start_time = time.time()


    # --- Load the wave simulation data ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}\\scale_length\\*.cdf*')[0])


    # prepare the output
    data_dict_output = {
                        'time': deepcopy(data_dict_wavescale['time']),
                        'mu_w': deepcopy(data_dict_wavescale['mu_w']),
                        'chi_w': deepcopy(data_dict_wavescale['chi_w']),
                        'z':deepcopy(data_dict_wavescale['z']),
                        'V_A':[[],{'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'Alfven Speed (MHD)', 'VAR_TYPE': 'data'}],
                        'n': [[], {'DEPEND_0': 'time', 'UNITS': 'm!A-3', 'LABLAXIS': 'Plasma Density', 'VAR_TYPE': 'data'}],
                        'm_i': [[], {'DEPEND_0': 'time', 'UNITS': 'kg', 'LABLAXIS': 'Alfven Speed (MHD)', 'VAR_TYPE': 'data'}],
                        'lambda_e': [[], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': 'Electron Skin Depth', 'VAR_TYPE': 'data'}],

                        'pDD_lambda_e_mu': [[], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/d&mu;', 'VAR_TYPE': 'data'}],
                        'pDD_lambda_e_chi': [[], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': 'd&lambda;!Be!N/d&chi;', 'VAR_TYPE': 'data'}],

                        'pDD_V_A_mu': [[], {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'dV_A/d&mu;', 'VAR_TYPE': 'data'}],
                        'pDD_V_A_chi': [[], {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'dV_A/d&chi;', 'VAR_TYPE': 'data'}],

                        'scale_dkpara': [[], {'DEPEND_0': 'time', 'UNITS': 'm!A-1', 'LABLAXIS': 'h (k!B&parallel;!N)', 'VAR_TYPE': 'data'}],
                        'scale_dkperp': [[], {'DEPEND_0': 'time', 'UNITS': 'm!A-1', 'LABLAXIS': 'h (k!B&perp;!N)', 'VAR_TYPE': 'data'}],
                        'scale_dmu': [[], {'DEPEND_0': 'time', 'UNITS': None, 'LABLAXIS': 'h (d&mu;/dt)', 'VAR_TYPE': 'data'}],
                        'scale_dchi': [[], {'DEPEND_0': 'time', 'UNITS': None, 'LABLAXIS': 'h (d&chi;/dt)', 'VAR_TYPE': 'data'}],
                        'B_dipole':[[],{}],
                        'meff': [[], {}],
                        'n_density': [[], {}],
                        'n_Op': [[], {}],
                        'n_Hp': [[], {}],
                        'rho': [[], {}],
                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################
    for key, func in envDict.items():
        data_dict_output[key][0] = func(data_dict_output['mu_w'][0], data_dict_output['chi_w'][0])

    ########################################
    # CONSTRUCT THE GRIDDED SIMULATION SPACE
    ########################################
    data_dict_output = {**data_dict_output,
                        **{'mu': [toggles.mu_range, {'LABLAXIS':'&mu;'}],
                           'chi': [toggles.chi_range, {'LABLAXIS':'&chi;'}],}
                        }

    for key, func in envDict.items():
        data_dict_output = {**data_dict_output, **{f'grid_{key}': [func(toggles.mu_grid, toggles.chi_grid), {'DEPEND_0':'mu','DEPEND_1':'chi'}]}}

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}\plasma_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)