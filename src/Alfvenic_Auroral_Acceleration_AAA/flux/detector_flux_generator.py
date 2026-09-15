
from timebudget import timebudget

@timebudget
def detector_flux_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import os

    # --- File-specific imports ---
    from glob import glob
    from itertools import product
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    import re
    from tqdm import tqdm

    # --- Delete the old Flux Files ---
    old_files = glob(f'{RunToggles.sim_data_output_path}/detector_flux/*.cdf*')
    for old_file in old_files:
        os.remove(old_file)

    # --- Load the wave runners data ---
    liouville_files = glob(rf'{RunToggles.sim_data_output_path}/liouville_mapping/*.cdf*')

    for i in tqdm(range(len(liouville_files))):

        file_name = liouville_files[i]
        data_dict_distribution = stl.loadDictFromFile(file_name)

        #####################################
        # --- CALCULATE DIFFERENTIAL FLUX ---
        #####################################
        JE = np.zeros_like(data_dict_distribution['distribution_function'][0]) # In S.I units
        JN = np.zeros_like(data_dict_distribution['distribution_function'][0])  # In S.I units

        sizes = [len(data_dict_distribution['distribution_function'][0]), len(data_dict_distribution['pitch_angle'][0]), len(data_dict_distribution['energy'][0])]
        for tmeIdx, ptchIdx, engyIdx in product(*[range(thing) for thing in sizes]):
            Energy_val = data_dict_distribution['energy'][0][engyIdx]*stl.q0 # convert from eV to Joules (for now)
            JE[tmeIdx][ptchIdx][engyIdx] = (2*np.square(Energy_val)/np.square(stl.m_e))*(data_dict_distribution['distribution_function'][0][tmeIdx][ptchIdx][engyIdx])

        # convert from SI to eV-s^-1-cm^-2-eV^-1
        JE = JE * (1/np.square(stl.cm_to_m))

        ###########################
        # --- OUTPUT EVERYTHING ---
        ###########################

        # --- prepare the output ---
        data_dict_output = {
            'time': data_dict_distribution['time'].copy(),
            'time_waves': data_dict_distribution['time_waves'].copy(),
            'energy': data_dict_distribution['energy'].copy(),
            'pitch_angle': data_dict_distribution['pitch_angle'].copy(),
            'Differential_Number_Flux': [np.array(JN), {'DEPEND_0':'time','DEPEND_2':'energy','DEPEND_1':'pitch_angle','UNITS':'cm!U-2!N str!U-1!N s!U-1!N eV!U-1!N','LABLAXIS': 'Differential_Number_Flux','VAR_TYPE':'data'}],
            'Differential_Energy_Flux': [np.array(JE), {'DEPEND_0': 'time', 'DEPEND_2': 'energy', 'DEPEND_1': 'pitch_angle', 'UNITS': 'cm!U-2!N str!U-1!N s!U-1!N eV/eV', 'LABLAXIS': 'Differential_Energy_Flux', 'VAR_TYPE': 'data'}],
            'E_para_obs':data_dict_distribution['E_para_obs'].copy(),
            'E_perp_obs':data_dict_distribution['E_perp_obs'].copy(),
            'B_perp_obs':data_dict_distribution['B_perp_obs'].copy()
        }

        if RunToggles.store_output:
            # save the results
            mapping_alt = int(re.search(r'_(\d+)km', file_name).group(1))
            outputPath = rf'{RunToggles.sim_data_output_path}/detector_flux/detector_flux_{mapping_alt}km.cdf'
            stl.outputDataDict(outputPath, data_dict_output)