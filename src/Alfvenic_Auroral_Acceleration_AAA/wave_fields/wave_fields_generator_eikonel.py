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

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_ray_eqns['time'][0])),deepcopy(data_dict_ray_eqns['time'][1])],
        'mu_w': deepcopy(data_dict_ray_eqns['mu_w']),
        'chi_w': deepcopy(data_dict_ray_eqns['chi_w']),
        'z': deepcopy(data_dict_ray_eqns['z']),
        'eikonel' : [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'rad', 'LABLAXIS': 'eikonel', 'VAR_TYPE': 'data'}],
        'eikonel_deg': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'rad', 'LABLAXIS': 'eikonel', 'VAR_TYPE': 'data'}],
        'modFunc': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': None, 'LABLAXIS': 'modFunc', 'VAR_TYPE': 'data'}],
        'cos_eikonel': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': None, 'LABLAXIS': 'cos(eikonel)', 'VAR_TYPE': 'data'}],
        'E_perp_waveform': [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_waveform': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'B_perp_waveform': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_perp_amplitude': [[], {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp_amplitude': [[], {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_amplitude': [[], {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity_eV':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        'DAW_velocity': [[], {'DEPEND_0': 'z', 'UNITS': 'm/s', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
    }

    # Calculate the resonance window
    # potential_para_max = np.array([np.max(np.abs(data_dict_output['potential_mu'][0][i])) for i in range(len(data_dict_output['potential_mu'][0]))])
    DAW_vel = data_dict_ray_eqns['omega'][0]/deepcopy(data_dict_ray_eqns['k_mu'][0])
    # data_dict_output['resonance_high'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel + np.sqrt(2*stl.q0*potential_para_max/stl.m_e))
    # data_dict_output['resonance_low'][0] = 0.5*(stl.m_e/stl.q0)*np.square((DAW_vel - np.sqrt(2 * stl.q0 * potential_para_max / stl.m_e)))
    data_dict_output['DAW_velocity_eV'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel)
    data_dict_output['DAW_velocity'][0] = DAW_vel

    # Calculate the Field amplitudes
    alt_idx = np.abs(data_dict_ray_eqns['z'][0] - 500).argmin() # find the index of where Z0 occurs in altitude

    B_perp_amplitude = WaveFieldsToggles.B_perp_0 * np.sqrt(data_dict_ray_eqns['v_group_mu'][0][alt_idx]/data_dict_ray_eqns['v_group_mu'][0])
    E_perp_amplitude = data_dict_plasEvrn['V_A'][0]*data_dict_plasEvrn['inertial_term'][0] * B_perp_amplitude
    E_mu_amplitude = E_perp_amplitude* (( data_dict_ray_eqns['k_mu'][0]*data_dict_ray_eqns['k_perp'][0]*np.square(data_dict_plasEvrn['lambda_e'][0]) )/np.square(data_dict_plasEvrn['inertial_term'][0]))

    # calculate the waveforms using the eikonel
    eikonel = np.array([data_dict_ray_eqns['k_mu'][0] * data_dict_ray_eqns['z'][0] * stl.m_to_km + data_dict_ray_eqns['omega'][0] * data_dict_ray_eqns['time'][0][tmeIdx] for tmeIdx in range(len(data_dict_ray_eqns['time'][0]))])
    E_perp_waveform = np.multiply(E_perp_amplitude,np.cos(eikonel))
    E_mu_waveform = np.multiply(E_mu_amplitude, np.cos(eikonel))
    B_perp_waveform = np.multiply(B_perp_amplitude,np.cos(eikonel))
    modFunc = np.zeros_like(B_perp_waveform)

    # Isolate the eikonel expressions to a single "wave"
    alt_thresh = 18220 # in kilometers
    # for tmeIdx in range(len(data_dict_ray_eqns['time'][0])):
    #
    #     # --- FRONT OF WAVE ---
    #     # find the first cos(eikonel) zero crossing above alt_thresh. It must be iterative since the altitude of where the wave front is must change
    #     alt_idx = np.abs(data_dict_ray_eqns['z'][0] - alt_thresh).argmin()
    #
    #     # get all the zero crossings of the cos(eikonel)
    #     zero_crossing_idxs = np.where(np.diff(np.sign(np.cos(eikonel[tmeIdx]))))[0]
    #
    #     # get the first index where the eikonel crosses zero ABOVE your altitude threshold
    #     print('\n--- --- --- ---')
    #     print('TmeIdx', tmeIdx)
    #     print('zero crossings',zero_crossing_idxs)
    #     print('alt idx',alt_idx)
    #     lead_idx = zero_crossing_idxs[np.where(zero_crossing_idxs < alt_idx)[0][-1]]
    #     print('lead idx',lead_idx)
    #
    #     # update the altitude threshold by the amount travelled by the wave in the n+1 step
    #     phase_speed = data_dict_output['DAW_velocity'][0][alt_idx]
    #     if tmeIdx == 0:
    #         alt_thresh = data_dict_ray_eqns['z'][0][lead_idx] - phase_speed * data_dict_ray_eqns['time'][0][tmeIdx]/stl.m_to_km
    #     elif tmeIdx == len(data_dict_ray_eqns['time'][0])-1:
    #         deltaT = data_dict_ray_eqns['time'][0][tmeIdx] - data_dict_ray_eqns['time'][0][tmeIdx-1]
    #         alt_thresh -= deltaT * phase_speed/stl.m_to_km
    #     else:
    #         deltaT = data_dict_ray_eqns['time'][0][tmeIdx+1] - data_dict_ray_eqns['time'][0][tmeIdx]
    #         alt_thresh -= deltaT*phase_speed/stl.m_to_km
    #     print('alt thresh',alt_thresh)
    #
    #     # --- BACK OF WAVE ---
    #     # get the index that's 2 lower than the lead index, to form a "sinusoid" wave
    #     try:
    #         follow_idx = zero_crossing_idxs[np.where(zero_crossing_idxs < alt_idx)[0][-3]]
    #     except:
    #         follow_idx = zero_crossing_idxs[np.where(zero_crossing_idxs < alt_idx)[0][-2]]
    #     print('follow idx',follow_idx)
    #
    #     modFunc[tmeIdx,follow_idx+1:lead_idx] = 1

    data_dict_output['eikonel'][0] = eikonel
    data_dict_output['cos_eikonel'][0] = np.cos(eikonel)
    # data_dict_output['E_perp_waveform'][0] = E_perp_waveform*modFunc
    data_dict_output['E_perp_waveform'][0] = E_perp_waveform
    # data_dict_output['E_mu_waveform'][0] = E_mu_waveform * modFunc
    data_dict_output['E_mu_waveform'][0] = E_mu_waveform
    # data_dict_output['B_perp_waveform'][0] = B_perp_waveform * modFunc
    data_dict_output['B_perp_waveform'][0] = B_perp_waveform
    data_dict_output['modFunc'][0] = modFunc

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

