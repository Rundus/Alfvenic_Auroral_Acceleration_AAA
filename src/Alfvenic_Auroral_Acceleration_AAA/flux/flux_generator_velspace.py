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
    # data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/distributions_{DistributionToggles.z0_obs}km.cdf')[0])
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/*.cdf')[0])

    ###########################################################
    # --- INTERPOLATE DISTRIBUTIONS ONTO PITCH/ENERGY SPACE ---
    ###########################################################
    # --- interpolate distribution function onto velocity space ---
    sizes = [len(DistributionToggles.pitch_range), len(DistributionToggles.energy_range)]
    Distribution_interp = np.zeros(shape=(len(data_dict_distribution['time'][0]), sizes[0], sizes[1]))

    X, Y = np.meshgrid(DistributionToggles.v_perp_space, DistributionToggles.v_para_space)
    for tmeIdx in tqdm(range(len(Distribution_interp[0]))):
        zData = np.array(deepcopy(data_dict_distribution['Distribution_Function'][0][tmeIdx])).flatten()
        interp = LinearNDInterpolator(list(zip(X.flatten(), Y.flatten())), zData)

        for ptchIdx, engyIdx in product(*[range(sizes[0]), range(sizes[1])]):
            Vbar = SimClasses().to_Vel(DistributionToggles.energy_range[engyIdx])
            Vpara_interp = Vbar * np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
            Vperp_interp = Vbar * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))
            Distribution_interp[tmeIdx][ptchIdx][engyIdx] = interp(Vperp_interp, Vpara_interp)
            print(Vperp_interp, Vpara_interp)

    # Distribution_interp[np.isnan(Distribution_interp)] = 0



    #####################################
    # --- CALCULATE DIFFERENTIAL FLUX ---
    #####################################
    JE = np.zeros_like(Distribution_interp) # In S.I units
    JN = np.zeros_like(Distribution_interp)  # In S.I units

    sizes = [len(data_dict_distribution['time'][0]),
             len(DistributionToggles.pitch_range),
             len(DistributionToggles.energy_range)]

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