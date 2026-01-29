
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from timebudget import timebudget

@timebudget
def field_particle_correlation_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from glob import glob


    # --- File-specific imports ---
    from scipy.interpolate import LinearNDInterpolator
    from scipy.interpolate import RegularGridInterpolator
    from tqdm import tqdm
    from scipy.integrate import simpson
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
    import itertools

    # --- Load the simulation data ---
    data_dict_flux = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/flux_{DistributionToggles.z0_obs}km.cdf')[0])
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/distributions_{DistributionToggles.z0_obs}km.cdf')[0])

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

    # Calculate the un-perturbed distribution function in velocity space
    f0 = np.zeros(shape=(len(data_dict_flux['time'][0]), sizes_vspace[0], sizes_vspace[1]))

    for idx1, vperpVal in enumerate(FPCToggles.v_perp_space):
        for idx2, vparaVal in enumerate(FPCToggles.v_para_space):
            f0[0][idx1][idx2] = DistributionClasses().Maxwellian(vperpVal, vparaVal)

    # calculate the gradient in f0
    f0_df_dvE = np.zeros_like(f0)
    for perpIdx in range(sizes_vspace[0]):
        f0_df_dvE[0, perpIdx] = (stl.q0*np.square(FPCToggles.v_para_space)/2)*np.gradient(f0[0][perpIdx], FPCToggles.v_para_space)

    # Fill in the rest of the times with the f0
    for tme in range(1, len(data_dict_flux['time'][0])):
        f0[tme] = f0[0]
        f0_df_dvE[tme] = f0_df_dvE[0]

    # --- correlate the results ---
    instant_correlation = np.zeros(shape=(Ntimes ,sizes_vspace[0], sizes_vspace[1]))
    instant_correlation_f0 = np.zeros(shape=(Ntimes, sizes_vspace[0], sizes_vspace[1]))
    FPC = np.zeros(shape=(sizes_vspace))
    FPC_f0 = np.zeros(shape=(sizes_vspace))
    FPC_residual = np.zeros(shape=(sizes_vspace))

    # 1 interpolate the E-Field data onto the particle data timebase
    E_mu_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['E_mu_obs'][0])
    B_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['B_perp_obs'][0])
    E_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['E_perp_obs'][0])

    # 2 Determine the period of the wave in the data
    grad = np.gradient(E_mu_corr)
    finder = np.where(np.abs(grad) > 0)[0]
    low_idx = finder[0]
    high_idx = finder[-1]
    corr_times = deepcopy(data_dict_distribution['time'][0][low_idx:high_idx + 1])
    tau = corr_times[-1] - corr_times[0]

    print(f'\n{sizes_vspace[0] * sizes_vspace[1]} Number of Iterations')
    for perpIdx, paraIdx in tqdm(itertools.product(*[range(thing) for thing in sizes_vspace])):

        # 2 cross-correlate the E-Field timeseries
        A_term = deepcopy(df_dvE[low_idx:high_idx+1,perpIdx,paraIdx])
        A_term_f0 = deepcopy(f0_df_dvE[low_idx:high_idx + 1, perpIdx, paraIdx])
        A_term_residual = deepcopy(df_dvE[low_idx:high_idx+1,perpIdx,paraIdx] - f0_df_dvE[low_idx:high_idx + 1, perpIdx, paraIdx])
        B_term = -1*E_mu_corr[low_idx:high_idx+1]

        instant_correlation[:, perpIdx, paraIdx] = df_dvE[:,perpIdx,paraIdx] * (-1*E_mu_corr)
        instant_correlation_f0[:,perpIdx, paraIdx] = f0_df_dvE[:, perpIdx, paraIdx] * (-1 * E_mu_corr)

        FPC[perpIdx, paraIdx] = (1/tau)*simpson(y=A_term*B_term,x=corr_times)
        FPC_f0[perpIdx, paraIdx] = (1 / tau) * simpson(y=A_term_f0 * B_term, x=corr_times)
        FPC_residual[perpIdx,paraIdx] = (1 / tau) * simpson(y=A_term_residual * B_term, x=corr_times)

    # integrate over Perpendicular Velocity space to get the J dot E for a given inital electron energy
    FPC_vpara_int = np.zeros(shape=(sizes_vspace[1]))
    FPC_vpara_int_f0 = np.zeros(shape=(sizes_vspace[1]))
    FPC_vpara_int_residual = np.zeros(shape=(sizes_vspace[1]))

    for idx in range(sizes_vspace[1]):
        FPC_vpara_int[idx] = simpson(FPC[:,idx], FPCToggles.v_perp_space)
        FPC_vpara_int_f0[idx] = simpson(FPC_f0[:, idx], FPCToggles.v_perp_space)
        FPC_vpara_int_residual[idx] = simpson(FPC_residual[:, idx], FPCToggles.v_perp_space)

    # Integrate over v_perp to get the total FPC and add 2pi to cover all other v_perp directions (assuming gyrotropy)
    FPC_total = 2*np.pi*simpson(FPC_vpara_int,FPCToggles.v_para_space)
    FPC_total_f0 = 2*np.pi*simpson(FPC_vpara_int_f0, FPCToggles.v_para_space)
    FPC_total_residual = 2*np.pi*simpson(FPC_vpara_int_residual, FPCToggles.v_para_space)

    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time':deepcopy(data_dict_distribution['time']),
        'FPC_total': [np.array([FPC_total]), {'VAR_TYPE': 'data'}],
        'FPC(vpara)': [np.array(FPC_vpara_int), {'DEPEND_0': 'v_para', 'VAR_TYPE': 'data'}],
        'FPC': [np.array(FPC), {'DEPEND_0': 'v_perp', 'DEPEND_1': 'v_para', 'VAR_TYPE': 'data'}],
        'instant_correlation':[np.array(instant_correlation),{'DEPEND_0':'time','DEPEND_1':'v_perp','DEPEND_2':'v_para','VAR_TYPE':'data'}],

        'FPC_total_f0': [np.array([FPC_total_f0]), {'VAR_TYPE': 'data'}],
        'FPC(vpara)_f0': [np.array(FPC_vpara_int_f0), {'DEPEND_0': 'v_para', 'VAR_TYPE': 'data'}],
        'FPC_f0': [np.array(FPC_f0), {'DEPEND_0': 'v_perp', 'DEPEND_1': 'v_para', 'VAR_TYPE': 'data'}],
        'instant_correlation_f0': [np.array(instant_correlation_f0),{'DEPEND_0': 'time', 'DEPEND_1': 'v_perp', 'DEPEND_2': 'v_para', 'VAR_TYPE': 'data'}],

        'FPC_residual': [np.array(FPC_residual), {'DEPEND_0': 'v_perp', 'DEPEND_1': 'v_para', 'VAR_TYPE': 'data'}],
        'FPC_total_residual': [np.array([FPC_total_residual]), {'VAR_TYPE': 'data'}],

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

    # Save the base run
    outputPath = rf'{FPCToggles.outputFolder}/FPC.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/FPC_{DistributionToggles.z0_obs}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)