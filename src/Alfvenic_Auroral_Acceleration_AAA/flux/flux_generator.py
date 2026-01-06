
def flux_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.flux.flux_toggles import FluxToggles

    # --- Load the wave simulation data ---
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])

    # --- prepare the output ---
    data_dict_output = {
        'time': [[], {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'Energy' : [[],{'UNITS':'eV','LABLAXIS':'Energy','VAR_TYPE':'support_data'}],
        'Pitch_Angle':[[], {'UNITS':'deg','LABLAXIS':'Pitch Angle','VAR_TYPE':'support_data'}],
        'Differential_Number_Flux': [[],{'DEPEND_0':'time','DEPEND_1':'Pitch_Angle','DEPEND_2':'Energy','UNITS':''}],
        'Distribution_Function': [[], {'DEPEND_0':'time','DEPEND_1':'Pitch_Angle','DEPEND_2':'Energy','UNITS':'m!A-6!Ns!A-3!N','LABLAXIS':'Distribution Function','VAR_TYPE':'data'}],
    }

    outputPath = rf'{FluxToggles.outputFolder}/flux.cdf'
    stl.outputDataDict(outputPath, data_dict_output)