from timebudget import timebudget

@timebudget
def flux_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.flux.flux_toggles import FluxToggles
    from itertools import product

    # --- Load the wave simulation data ---
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/*.cdf*')[0])

    #####################################
    # --- CALCULATE DIFFERENTIAL FLUX ---
    #####################################
    JE = np.zeros_like(data_dict_distribution['Distribution'][0]) # In S.I units
    JN = np.zeros_like(data_dict_distribution['Distribution'][0])  # In S.I units

    sizes = [len(data_dict_distribution['Distribution'][0]), len(data_dict_distribution['Pitch_Angle'][0]), len(data_dict_distribution['Energy'][0])]
    for tmeIdx, ptchIdx, engyIdx in product(*[range(thing) for thing in sizes]):
        Energy_val = data_dict_distribution['Energy'][0][engyIdx]*stl.q0 # convert from eV to Joules (for now)
        JE[tmeIdx][ptchIdx][engyIdx] = (2*np.square(Energy_val)/np.square(stl.m_e))*(data_dict_distribution['Distribution'][0][tmeIdx][ptchIdx][engyIdx])

    # convert from SI to eV-s^-1-cm^-2-eV^-1
    JE = JE * (1/np.square(stl.cm_to_m))



    ###########################
    # --- OUTPUT EVERYTHING ---
    ###########################

    # --- prepare the output ---
    data_dict_output = {
        'time': [deepcopy(data_dict_distribution['time'][0]), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'time_waves': deepcopy(data_dict_distribution['time_waves']),
        'Energy': deepcopy(data_dict_distribution['Energy']),
        'Pitch_Angle': deepcopy(data_dict_distribution['Pitch_Angle']),
        'Differential_Number_Flux': [np.array(JN), {'DEPEND_0':'time','DEPEND_2':'Energy','DEPEND_1':'Pitch_Angle','UNITS':'cm!U-2!N str!U-1!N s!U-1!N eV!U-1!N','LABLAXIS': 'Differential_Number_Flux','VAR_TYPE':'data'}],
        'Differential_Energy_Flux': [np.array(JE), {'DEPEND_0': 'time', 'DEPEND_2': 'Energy', 'DEPEND_1': 'Pitch_Angle', 'UNITS': 'cm!U-2!N str!U-1!N s!U-1!N eV/eV', 'LABLAXIS': 'Differential_Energy_Flux', 'VAR_TYPE': 'data'}],
        'E_mu_obs':deepcopy(data_dict_distribution['E_mu_obs']),
        'E_perp_obs':deepcopy(data_dict_distribution['E_perp_obs']),
        'B_perp_obs':deepcopy(data_dict_distribution['B_perp_obs'])
    }

    outputPath = rf'{FluxToggles.outputFolder}/flux.cdf'
    stl.outputDataDict(outputPath, data_dict_output)