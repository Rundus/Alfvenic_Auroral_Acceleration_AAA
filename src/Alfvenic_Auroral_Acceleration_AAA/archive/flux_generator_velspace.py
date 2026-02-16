import matplotlib.pyplot as plt
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import *
from timebudget import timebudget


@timebudget
def flux_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from tqdm import tqdm
    from scipy.interpolate import LinearNDInterpolator

    # --- File-specific imports ---
    from glob import glob
    from itertools import product

    # --- Load the wave simulation data ---
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/distributions_{DistributionToggles.z0_obs}km.cdf')[0])
    # data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/*.cdf')[0])

    ###########################################################
    # --- INTERPOLATE DISTRIBUTIONS ONTO PITCH/ENERGY SPACE ---
    ###########################################################
    # --- interpolate distribution function onto velocity space ---
    sizes = [len(data_dict_distribution['time'][0]), len(DistributionToggles.pitch_range), len(DistributionToggles.energy_range)]
    vel_sizes = [range(len(DistributionToggles.v_perp_space_obs)),range(len(DistributionToggles.v_para_space_obs))]
    Distribution_interp = np.zeros(shape=tuple(sizes))
    X, Y = np.meshgrid(DistributionToggles.pitch_range, DistributionToggles.energy_range)
    for tmeIdx in tqdm(range(sizes[0])):

        zData = np.array([data_dict_distribution['Distribution_Function'][0][tmeIdx][vperpIdx][vparaIdx] for vperpIdx, vparaIdx in product(*vel_sizes)])
        engy_points = np.array([0.5*(stl.m_e/stl.q0)*(np.square(DistributionToggles.v_perp_space_obs[vperpIdx]) + np.square(DistributionToggles.v_para_space_obs[vparaIdx])) for vperpIdx, vparaIdx in product(*vel_sizes)])
        ptch_points = np.array([np.degrees(np.arctan2(DistributionToggles.v_perp_space_obs[vperpIdx],DistributionToggles.v_para_space_obs[vparaIdx])) for vperpIdx, vparaIdx in product(*vel_sizes)])

        interp = LinearNDInterpolator(list(zip(ptch_points, engy_points)), zData)
        # print('\n',tmeIdx, engy_points, ptch_points, zData)
        Distribution_interp[tmeIdx] = interp(X,Y).T


    #####################################
    # --- CALCULATE DIFFERENTIAL FLUX ---
    #####################################
    JE = np.zeros_like(Distribution_interp) # In S.I units
    JN = np.zeros_like(Distribution_interp)  # In S.I units

    for tmeIdx, ptchIdx, engyIdx in product(*[range(thing) for thing in sizes]):
        Energy_val = DistributionToggles.energy_range[engyIdx]*stl.q0 # convert from eV to Joules (for now)
        JE[tmeIdx][ptchIdx][engyIdx] = (2*np.square(Energy_val)/np.square(stl.m_e))*(Distribution_interp[tmeIdx][ptchIdx][engyIdx])

    # convert from SI to eV-s^-1-cm^-2-eV^-1
    JE = JE * (1/np.square(stl.cm_to_m))

    ###########################
    # --- OUTPUT EVERYTHING ---
    ###########################

    # --- prepare the output ---
    data_dict_output = {
        'time': [deepcopy(data_dict_distribution['time'][0]), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'time_waves': deepcopy(data_dict_distribution['time_waves']),
        'Energy': [np.array(DistributionToggles.energy_range), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
        'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
        'Distribution_interp': [np.array(Distribution_interp), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
        'Differential_Number_Flux': [np.array(JN), {'DEPEND_0':'time','DEPEND_2':'Energy','DEPEND_1':'Pitch_Angle','UNITS':'cm!U-2!N str!U-1!N s!U-1!N eV!U-1!N','LABLAXIS': 'Differential_Number_Flux','VAR_TYPE':'data'}],
        'Differential_Energy_Flux': [np.array(JE), {'DEPEND_0': 'time', 'DEPEND_2': 'Energy', 'DEPEND_1': 'Pitch_Angle', 'UNITS': 'cm!U-2!N str!U-1!N s!U-1!N eV/eV', 'LABLAXIS': 'Differential_Energy_Flux', 'VAR_TYPE': 'data'}],
        'E_mu_obs':deepcopy(data_dict_distribution['E_mu_obs']),
        'E_perp_obs':deepcopy(data_dict_distribution['E_perp_obs']),
        'B_perp_obs':deepcopy(data_dict_distribution['B_perp_obs'])
    }

    # save this particular run
    outputPath = rf'{FluxToggles.outputFolder}/flux.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/flux_{DistributionToggles.z0_obs}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)