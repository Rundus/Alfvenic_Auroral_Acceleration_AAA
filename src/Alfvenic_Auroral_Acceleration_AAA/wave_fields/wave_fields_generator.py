

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

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################

    # create the grid on which to plot everything
    # --- MU-Dimension ---
    # determine minimum/maximum mu value for the TOP colattitude
    N_mu = 200  # number of points in mu direction
    mu_min, mu_max = [-1, -0.3]
    mu_range = np.linspace(mu_min, mu_max, N_mu)
    alt_range = stl.m_to_km*stl.Re*(ScaleLengthClasses.r_muChi(mu_range,[SimToggles.chi0 for i in range(len(mu_range))])-1)


    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)