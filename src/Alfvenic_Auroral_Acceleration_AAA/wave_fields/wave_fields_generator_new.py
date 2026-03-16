from timebudget import timebudget
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *

@timebudget
def wave_fields_generator():
    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
    from tqdm import tqdm
    from scipy.integrate import simpson
    import multiprocessing as mp

    # --- Load the needed data ---
    data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/ray_equations/ray_equations.cdf')[0])
    data_dict_plasEvrn = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/plasma_environment.cdf')[0])

    # Calculate the Field amplitudes
    alt_idx = np.abs(data_dict_ray_eqns['z'][0] - 500).argmin() # find the index of where Z0 occurs in altitude

    B_perp_amplitude = WaveFieldsToggles.B_perp_0 * np.sqrt(data_dict_ray_eqns['v_group_mu'][0][alt_idx]/data_dict_ray_eqns['v_group_mu'][0])
    E_perp_amplitude = data_dict_plasEvrn['V_A'][0]*data_dict_plasEvrn['inertial_term'][0] * B_perp_amplitude
    E_mu_amplitude = E_perp_amplitude* (( data_dict_ray_eqns['k_mu'][0]*data_dict_ray_eqns['k_perp'][0]*np.square(data_dict_plasEvrn['lambda_e'][0]) )/np.square(data_dict_plasEvrn['inertial_term'][0]))

    # calculate the waveforms using the eikonel
    test_t0 = 0
    E_perp_waveform = np.zeros(shape=(len(data_dict_ray_eqns['time'][0]),len(data_dict_ray_eqns['z'][0])))
    E_mu_waveform = np.zeros(shape=(len(data_dict_ray_eqns['time'][0]), len(data_dict_ray_eqns['z'][0])))
    B_perp_waveform = np.zeros(shape=(len(data_dict_ray_eqns['time'][0]), len(data_dict_ray_eqns['z'][0])))

    for tmeIdx in range(len(data_dict_ray_eqns['time'][0])):
        eikonel_temp = data_dict_ray_eqns['k_mu'][0] * data_dict_ray_eqns['z'][0] * stl.m_to_km - data_dict_ray_eqns['omega_calc'][0] * data_dict_ray_eqns['time'][0][tmeIdx]
        E_perp_waveform[tmeIdx] =  E_perp_amplitude*np.cos(eikonel_temp)
        E_mu_waveform[tmeIdx] = E_mu_amplitude * np.cos(eikonel_temp)
        B_perp_waveform[tmeIdx] = B_perp_amplitude * np.cos(eikonel_temp)

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_ray_eqns['time'][0])),deepcopy(data_dict_ray_eqns['time'][1])],
        'mu_w': deepcopy(data_dict_ray_eqns['mu_w']),
        'chi_w': deepcopy(data_dict_ray_eqns['chi_w']),
        'z': deepcopy(data_dict_ray_eqns['z']),
        # 'eikonel' : [np.array(eikonel), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': None, 'LABLAXIS': 'eikonel', 'VAR_TYPE': 'data'}],
        'E_perp_waveform': [np.array(E_perp_waveform), {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_waveform': [np.array(E_mu_waveform),{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'B_perp_waveform': [np.array(B_perp_waveform)/1E-9,{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_perp_amplitude': [np.array(E_perp_amplitude), {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp_amplitude': [np.array(B_perp_amplitude)/(1E-9), {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_amplitude': [np.array(E_mu_amplitude), {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        # 'potential_perp': [np.array(PotentialPerp), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Perpendicular Potential', 'VAR_TYPE': 'data'}],
        # 'potential_mu': [np.array(PotentialPara), {'DEPEND_0': 'time', 'DEPEND_1': 'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Parallel Potential', 'VAR_TYPE': 'data'}],
        'mu_grid': [WaveFieldsToggles.mu_grid,deepcopy(data_dict_ray_eqns['mu_w'][1])],
        'alt_grid': [WaveFieldsToggles.alt_grid, deepcopy(data_dict_ray_eqns['z'][1])],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity_eV':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        'DAW_velocity': [[], {'DEPEND_0': 'z', 'UNITS': 'm/s', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        # 'mu_ponyting_flux': [np.array(Eperp)*np.array(Bperp)/stl.u0,{'DEPEND_0': 'z', 'UNITS': 'W/m!A2!N', 'LABLAXIS': 'Poynting Flux', 'VAR_TYPE': 'data'}]
    }

    # Calculate the resonance window
    # potential_para_max = np.array([np.max(np.abs(data_dict_output['potential_mu'][0][i])) for i in range(len(data_dict_output['potential_mu'][0]))])
    # DAW_vel = data_dict_ray_eqns['omega'][0]/deepcopy(data_dict_ray_eqns['k_mu'][0])
    # data_dict_output['resonance_high'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel + np.sqrt(2*stl.q0*potential_para_max/stl.m_e))
    # data_dict_output['resonance_low'][0] = 0.5*(stl.m_e/stl.q0)*np.square((DAW_vel - np.sqrt(2 * stl.q0 * potential_para_max / stl.m_e)))
    # data_dict_output['DAW_velocity_eV'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel)
    # data_dict_output['DAW_velocity'][0] = DAW_vel

    ################
    # --- OUTPUT ---
    ################

    # save this particular run
    outputPath = rf'{WaveFieldsToggles.outputFolder}/wave_fields_NEW.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/wave_fields_{DistributionToggles.z0_obs}km_NEW.cdf'
        stl.outputDataDict(outputPath, data_dict_output)



wave_fields_generator()