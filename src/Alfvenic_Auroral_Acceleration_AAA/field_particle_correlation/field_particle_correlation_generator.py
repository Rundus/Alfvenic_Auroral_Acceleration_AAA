import itertools

import matplotlib.pyplot as plt
from timebudget import timebudget

@timebudget
def field_particle_correlation_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from glob import glob

    # --- File-specific imports ---
    from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_toggles import FPCToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
    from scipy.interpolate import LinearNDInterpolator
    from tqdm import tqdm
    from scipy import signal

    # --- Load the simulation data ---
    data_dict_flux = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/flux/flux.cdf')[0])
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/distributions.cdf')[0])

    ################################
    # --- CALCULATE CORRELATION ----
    ################################

    # --- calculate the velocity grid ---
    Ntimes = len(DistributionToggles.obs_times)
    Nptchs = len(DistributionToggles.pitch_range)
    Nengy = len(DistributionToggles.energy_range)
    sizes = [Ntimes, Nptchs, Nengy]
    parallel_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    perp_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    for tmeIdx, ptchIdx, engyIdx in itertools.product(*[range(item) for item in sizes]):
        parallel_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2*stl.q0*DistributionToggles.energy_range[engyIdx]/stl.m_e)*np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        perp_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))

    # --- interpolate distribution function onto velocity space ---
    sizes_vspace = [len(FPCToggles.v_perp_space), len(FPCToggles.v_para_space)]
    Distribution_interp = np.zeros(shape=(len(data_dict_flux['time'][0]), sizes_vspace[0], sizes_vspace[1]))
    X, Y = np.meshgrid(FPCToggles.v_perp_space, FPCToggles.v_para_space)
    for tmeIdx in tqdm(range(sizes[0])):
        xData = perp_velocity_grid[tmeIdx].flatten()
        yData = parallel_velocity_grid[tmeIdx].flatten()
        zData = np.array(deepcopy(data_dict_distribution['Distribution'][0][tmeIdx])).flatten()
        interp = LinearNDInterpolator(list(zip(xData,yData)), zData)
        Distribution_interp[tmeIdx] = interp(X,Y).T

    Distribution_interp[np.isnan(Distribution_interp)]  = 0

    # --- calculate the distribution function velocity space gradient ---
    df_dvE = np.zeros_like(Distribution_interp)
    print(f'\n {sizes[0]*sizes_vspace[0]} Number of Iterations')
    for tmeIdx, perpIdx in tqdm(itertools.product(*[range(sizes[0]),range(sizes_vspace[0])])):
        df_dvE[tmeIdx, perpIdx] = (stl.q0*np.square(FPCToggles.v_para_space)/2)*np.gradient(Distribution_interp[tmeIdx, perpIdx], FPCToggles.v_para_space)

    # correlate the results
    instant_correlation = np.zeros(shape=(Ntimes ,sizes_vspace[0], sizes_vspace[1]))
    FPC = np.zeros(shape=(sizes_vspace))

    # 1 interpolate the E-Field data onto the particle data timebase
    E_mu_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['E_mu_obs'][0])
    B_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['B_perp_obs'][0])
    E_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0],data_dict_flux['E_perp_obs'][0])
    print(f'\n{sizes_vspace[0] * sizes_vspace[1]} Number of Iterations')
    for perpIdx, paraIdx in tqdm(itertools.product(*[range(thing) for thing in sizes_vspace])):

        # 2 cross-correlate the E-Field timeseries
        corr = signal.correlate(df_dvE[:,perpIdx,paraIdx], -1*E_mu_corr, mode='same')
        instant_correlation[:, perpIdx, paraIdx] = corr
        corr[np.where(np.abs(corr)<1E-33)] = 0
        FPC[perpIdx, paraIdx] = np.mean(corr[np.where(corr != 0)])


    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time':deepcopy(data_dict_distribution['time']),
        'FPC': [np.array(FPC), {'DEPEND_0': 'v_perp', 'DEPEND_1': 'v_para', 'VAR_TYPE': 'data'}],
        'instant_correlation':[np.array(instant_correlation),{'DEPEND_0':'time','DEPEND_1':'v_perp','DEPEND_2':'v_para','VAR_TYPE':'data'}],
        'Distribution_Function' :[np.array(Distribution_interp),deepcopy(data_dict_distribution['Distribution'][1])],
        'df_dvpara' : [np.array(df_dvE), {'DEPEND_0':'time','DEPEND_1':'v_perp','DEPEND_2':'v_para','LABALAXIS':'df/dv_para','UNITS':'m^-3s^-6/m/s','VAR_TYPE':'data'}],
        'v_para': [FPCToggles.v_para_space,{'UNITS': 'm/s', 'LABLAXIS': 'v_para'}],
        'v_perp': [FPCToggles.v_perp_space, {'UNITS': 'm/s', 'LABLAXIS': 'v_perp'}],
        'E_mu_corr':[np.array(E_mu_corr),deepcopy(data_dict_distribution['E_mu_obs'][1])],
        'B_perp_corr': [np.array(B_perp_corr),deepcopy(data_dict_distribution['B_perp_obs'][1])],
        'E_perp_corr': [np.array(E_perp_corr),deepcopy(data_dict_distribution['E_perp_obs'][1])],
        'time_waves' : deepcopy(data_dict_distribution['time_waves'])
    }
    data_dict_output['Distribution_Function'][1]['DEPEND_1'] = 'v_perp'
    data_dict_output['Distribution_Function'][1]['DEPEND_2'] = 'v_para'
    data_dict_output['E_perp_corr'][1]['DEPEND_0'] = 'time'
    data_dict_output['B_perp_corr'][1]['DEPEND_0'] = 'time'
    data_dict_output['E_mu_corr'][1]['DEPEND_0'] = 'time'
    outputPath = rf'{FPCToggles.outputFolder}/field_particle_correlation.cdf'
    stl.outputDataDict(outputPath, data_dict_output)