import matplotlib.pyplot as plt
from timebudget import timebudget

@timebudget
def field_particle_correlation_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from glob import glob
    import os
    import re

    # --- File-specific imports ---
    from tqdm import tqdm
    from scipy.stats import binned_statistic
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.FileIO import FileIOClass
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.executable_classes import ExecutableClasses
    run_module = ExecutableClasses().load_configuration_module()
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_classes import LiouvilleMapping
    from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_classes_claude import FPCClasses

    # --- Delete the old FPC Files ---
    old_files = glob(f'{FileIOClass.path_sim_output}/field_particle_correlation/*.cdf*')
    for old_file in old_files:
        os.remove(old_file)

    # --- Load the liouville mapped data ---
    dist_files = glob(rf'{FileIOClass.path_sim_output}/liouville_mapping/*.cdf*')

    # --- Load the wave environment data ---
    data_dict_waves = stl.loadDictFromFile(glob(rf'{FileIOClass.path_sim_output}/wave_potentials/*.cdf*')[0])

    # --- load the plasma environment data ---
    data_dict_plasma = stl.loadDictFromFile(glob(rf'{FileIOClass.path_sim_output}/plasma_environment/*.cdf*')[0])
    dist_obj = LiouvilleMapping(1000) # altitude doesn't matter here

    # loop over distribution function files
    for i in tqdm(range(len(dist_files))):

        ##########################
        # --- PREPARE THE DATA ---
        ##########################

        # Load the file-specific data
        data_dict = stl.loadDictFromFile(dist_files[i])

        # determine which altitude is in the file
        match = re.search(r"_(\d+(?:\.\d+)?)km\.cdf$", dist_files[i])
        alt_val = float(match.group(1))  # [km] altitude

        # get the Liouville Mapping Data
        f = data_dict['distribution_function'][0].copy()
        energy = data_dict['energy'][0].copy()
        pitch_angle = data_dict['pitch_angle'][0].copy()
        time_waves = data_dict['time_waves'][0].copy()
        time = data_dict['time'][0].copy()

        E_para_obs = -1 * data_dict['E_mu_obs'][0].copy()  # put a -1 sign since we're using parallel coordinates
        mask = np.where(np.abs(E_para_obs) < 1E-9)
        E_para_obs[mask] = 0
        mid = 0.5 * (time[1:] + time[:-1])  # get bins by calculating the deltaT/2 from the particle timestamps
        edges = np.concatenate(([2 * time[0] - mid[0]], mid, [2 * time[-1] - mid[-1]]))
        E_para_FPC, _, _ = binned_statistic(time_waves, E_para_obs.copy(), bins=edges)

        ##############################################
        # --- CALCULATE FPC (DETECTOR COORDINATES) ---
        ##############################################

        # performs the FPC calculation on f (not f0)
        res = FPCClasses.compute_fpc(f=f, pitch_angle=pitch_angle, energy=energy, Epar=E_para_FPC, t=time)  # or FPCClasses().compute_fpc(...)
        r = res["full"]
        grid = res["grid"]

        FPC_instant = r['C_native']  # C(pitch, Energy, t) - Raw FPC, unintegrated
        FPC_integrate_time = r['C_native_tint'] # C(vpara,vperp) - FPC integrated over time
        FPC_integrate_vperp = r['C'] # C(vpara, t) - FPC integrated over vperp
        FPC_integrate_vperp_vpara = r['C_int_v'] # C(t) - FPC integrated over vperp and vpara
        FPC_integrate_vperp_time = r['C_tint'] # C(vpara) - FPC integrated over vperp and time
        FPC_total = r['W_total'] # C - FPC integrated over vperp, vpara and time
        vparas = res['deposition']['vpar'] # vpara velocity bins

        # --- get the DAW velocity for comparison ---

        # get the instantaneous DAW velocity
        DAW_idx = np.abs(data_dict_waves['alt'][0] - alt_val*stl.m_to_km).argmin()
        DAW_vel_obs = np.array([data_dict_waves['DAW_velocity'][0][DAW_idx]])

        # Average the DAW over ~ 1000 km above the observation altitude for comparison
        DAW_idx_above = np.abs(data_dict_waves['alt'][0] - (alt_val+7000)*stl.m_to_km).argmin()
        DAW_vel_avg = np.array([np.nanmean(data_dict_waves['DAW_velocity'][0][DAW_idx:DAW_idx_above+1])])

        ################
        # --- OUTPUT ---
        ################
        data_dict_output = {
            'FPC_instant': [np.array(FPC_instant), {'DEPEND_0': 'time', 'DEPEND_1': 'pitch_angle', 'DEPEND_2': 'energy', 'VAR_TYPE': 'data', 'LABLAXIS': 'FPC instant'}],
            'FPC_integrate_time': [np.array(FPC_integrate_time), {'VAR_TYPE': 'data','DEPEND_0': 'pitch_angle','DEPEND_1': 'energy'}],
            'FPC_integrate_vperp' : [np.array(FPC_integrate_vperp),{'DEPEND_0':'vparas','DEPEND_1':'time','VAR_TYPE':'data','UNITS':'W m!A-3!N/(m/s)'}],
            'FPC_integrate_vperp_vpara': [np.array(FPC_integrate_vperp_vpara), {'DEPEND_0':'time','VAR_TYPE':'data','UNITS':'W m!A-3!N'}],
            'FPC_integrate_vperp_time': [np.array(FPC_integrate_vperp_time), {'DEPEND_0':'vparas','VAR_TYPE':'data','UNITS':'J m!A-3!N/(m/s)'}],
            'FPC_total': [np.array([FPC_total]), {'VAR_TYPE': 'data','UNITS':'J/m!A-3!N'}],
            'time': data_dict['time'].copy(),
            'vparas':[np.array(vparas),{'UNITS':'m/s'}],
            'E_para_FPC' : [E_para_FPC, {'DEPEND_0':'time','UNITS':'V/m','LABLAXIS':'Epara (FPC)','VAR_TYPE':'data'}],
            'pitch_angle':data_dict['pitch_angle'].copy(),
            'energy': data_dict['energy'].copy(),
            'distribution':[f,{'DEPEND_0': 'time','DEPEND_1': 'pitch_angle','DEPEND_2': 'energy', 'VAR_TYPE': 'data','UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'simulated distribution functon'}],
            'DAW_vel':data_dict_waves['DAW_velocity'].copy(),
            'DAW_vel_obs':[DAW_vel_obs,{'UNITS':'m/s','VAR_TYPE':'data','LABLAXIS':'DAW Velocity Observed'}],
            'DAW_vel_avg': [DAW_vel_avg, {'UNITS': 'm/s', 'VAR_TYPE': 'data', 'LABLAXIS': 'DAW Velocity Averaged'}],
        }

        # save the results
        mapping_alt = int(re.search(r'_(\d+)km', dist_files[i]).group(1))
        outputPath = rf'{FileIOClass.path_sim_output}/field_particle_correlation/FPC_{mapping_alt}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)