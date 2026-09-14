from timebudget import timebudget
@timebudget

def liouville_mapping_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np

    # --- File-specific imports ---
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_toggles import LiouvilleToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_classes import LiouvilleClasses


    for z_obs in LiouvilleToggles.mapping_alts:

        print('-----------------------')
        print(stl.color.RED + f'--- Altitude {z_obs} km ---' + stl.color.END)
        print('-----------------------')

        #######################################################
        ### EXECUTE LIOUVILLE MAPPING (Parallel Processing) ###
        #######################################################
        distribution_function = LiouvilleClasses().liouville_mapper(z_obs)

        ##############################
        # --- OBSERVED WAVE FIELDS ---
        ##############################
        # Eperp
        E_perp_obs = np.zeros(shape=(LiouvilleToggles.N_obs_wave_points))

        # E_mu
        E_mu_obs = np.zeros(shape=(LiouvilleToggles.N_obs_wave_points))

        # B_perp
        B_perp_obs = np.zeros(shape=(LiouvilleToggles.N_obs_wave_points))

        # for tmeIdx in range(LiouvilleToggles.N_obs_wave_points):
        #     eval_pos = [LiouvilleToggles.u0_obs, LiouvilleToggles.chi0_obs]
        #     E_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=LiouvilleToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='eperp')
        #     E_mu_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=LiouvilleToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='eMu')
        #     B_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=LiouvilleToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='bperp')

        ################
        # --- OUTPUT ---
        ################
        data_dict_output = {
            'time': [np.array(LiouvilleToggles.obs_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
            'time_waves': [np.array(LiouvilleToggles.obs_waves_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
            'distribution_function': [np.array(distribution_function), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
            'energy': [np.array(LiouvilleToggles.energy_range_obs), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
            'pitch_angle': [np.array(LiouvilleToggles.pitch_range_obs), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
            'B_perp_obs': [B_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
            'E_perp_obs': [E_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
            'E_para_obs': [E_mu_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
            'z_obs':[np.array([z_obs]),{'DEPEND_0': 'Observed_Altitude', 'UNITS': 'm', 'LABLAXIS': 'Z!Bobs!N', 'VAR_TYPE': 'support_data'}]
        }


        if RunToggles.store_output:
            outputPath = rf'{RunToggles.sim_data_output_path}/liouville_mapping/liouville_mapping_{z_obs}km.cdf'
            stl.outputDataDict(outputPath, data_dict_output)





