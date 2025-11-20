

def wave_fields_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses

    # --- Load the wave simulation data ---
    data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
    data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/*.cdf*')[0])

    # prepare the output
    data_dict_output = {
        'time': deepcopy(data_dict_wavescale['time']),
        'mu_w': deepcopy(data_dict_wavescale['mu_w']),
        'chi_w': deepcopy(data_dict_wavescale['chi_w']),
        'z': deepcopy(data_dict_wavescale['z']),

        'E_perp': [[], {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [[], {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [[], {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
    }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()

    #######################################
    # --- GET THE WAVE FIELD MAGNITUDES ---
    #######################################

    def E_para(Eperp):
        k_mu = data_dict_output['k_mu'][0]
        k_perp = data_dict_output['k_perp'][0]
        lmd_e = data_dict_plasma['lmb_e'][0]
        E_para = (k_mu*k_perp*Eperp)/(1 + np.square(k_perp*lmd_e))
        return E_para

    def B_perp(Eperp):
        VA = data_dict_plasma['VA'][0]
        k_perp = data_dict_output['k_perp'][0]
        lmd_e = data_dict_plasma['lmb_e'][0]
        return Eperp/VA*np.sqrt(1 + np.square(k_perp*lmd_e))


    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################
    for key, func in envDict.items():
        data_dict_output[key][0] = func(data_dict_output['mu_w'][0], data_dict_output['chi_w'][0])

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)