from timebudget import timebudget
@timebudget

def liouville_mapping_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from glob import glob
    import os

    # --- File-specific imports ---
    from scipy.interpolate import RegularGridInterpolator
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_toggles import LiouvilleToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_classes import LiouvilleClasses

    # --- Delete any old/previous files ---
    old_files = glob(f'{RunToggles.sim_data_output_path}/liouville_mapping/*.cdf*')
    for old_file in old_files:
        os.remove(old_file)

    # --- Load in some simulated data ---
    data_dict_potentials = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')
    data_dict_spatial = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf')
    mu_grid = data_dict_spatial['mu'][0]

    for z_obs in LiouvilleToggles.mapping_alts:

        print('-----------------------')
        print(stl.color.RED + f'--- Altitude {z_obs} km ---' + stl.color.END)
        print('-----------------------')

        #######################################################
        ### EXECUTE LIOUVILLE MAPPING (Parallel Processing) ###
        #######################################################
        mapping_object = LiouvilleClasses(z_obs)
        distribution_function = mapping_object.liouville_mapper()

        ##############################
        # --- OBSERVED WAVE FIELDS ---
        ##############################
        E_para_obs, E_perp_obs, B_perp_obs, obs_waves_times = mapping_object.observed_fields()

        ################
        # --- OUTPUT ---
        ################
        data_dict_output = {
            'time': [np.array(mapping_object.observation_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
            'time_waves': [np.array(obs_waves_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
            'distribution_function': [np.array(distribution_function), {'DEPEND_0': 'time', 'DEPEND_1': 'pitch_angle', 'DEPEND_2': 'energy', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
            'energy': [np.array(LiouvilleToggles.energy_range_obs), {'UNITS': 'eV', 'LABLAXIS': 'energy'}],
            'pitch_angle': [np.array(LiouvilleToggles.pitch_range_obs), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
            'B_perp_obs': [B_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'T', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
            'E_perp_obs': [E_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
            'E_para_obs': [E_para_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&parallel;!N', 'VAR_TYPE': 'data'}],
            'z_obs':[np.array([z_obs]),{'DEPEND_0': None, 'UNITS': 'km', 'LABLAXIS': 'Observation Altitude', 'VAR_TYPE': 'support_data'}]
        }

        if RunToggles.store_output:
            outputPath = rf'{RunToggles.sim_data_output_path}/liouville_mapping/liouville_mapping_{z_obs}km.cdf'
            stl.outputDataDict(outputPath, data_dict_output)





