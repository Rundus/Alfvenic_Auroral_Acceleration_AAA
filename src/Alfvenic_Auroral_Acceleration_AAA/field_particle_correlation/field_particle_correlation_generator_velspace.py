
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
    from tqdm import tqdm
    from scipy.integrate import simpson
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
    import itertools

    # --- Load the simulation data ---
    data_dict_flux = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/flux_{DistributionToggles.z0_obs}km.cdf')[0])
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/distributions_{DistributionToggles.z0_obs}km.cdf')[0])
    # data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/distributions.cdf')[0])

    ################################
    # --- CALCULATE CORRELATION ----
    ################################

    # --- calculate the velocity grid ---
    Ntimes = len(DistributionToggles.obs_times)
    Nvperps = len(DistributionToggles.v_perp_space)
    Nvparas = len(DistributionToggles.v_para_space)
    sizes = [Ntimes, Nvperps, Nvparas]

    # --- calculate the distribution function velocity space gradient ---
    Distribution = deepcopy(data_dict_distribution['Distribution_Function'][0])
    df_dvE = np.zeros_like(Distribution)
    # print(f'\n {sizes[0]*Nvperps} Number of Iterations')
    for tmeIdx, perpIdx in tqdm(itertools.product(*[range(sizes[0]),range(Nvperps)])):
        df_dvE[tmeIdx, perpIdx] = (stl.q0*np.square(DistributionToggles.v_para_space)/2)*np.gradient(Distribution[tmeIdx, perpIdx], DistributionToggles.v_para_space)

    # Calculate the un-perturbed distribution function in velocity space
    f0 = np.zeros(shape=(len(data_dict_flux['time'][0]), Nvperps, Nvparas))
    for idx1, vperpVal in enumerate(DistributionToggles.v_perp_space):
        for idx2, vparaVal in enumerate(DistributionToggles.v_para_space):
            f0[0][idx1][idx2] = DistributionClasses().mapped_distribution(DistributionToggles.u0_obs,DistributionToggles.chi0_obs, vperpVal, vparaVal)

    # calculate the gradient in f0
    f0_df_dvE = np.zeros_like(f0)
    for perpIdx in range(Nvperps):
        f0_df_dvE[0, perpIdx] = (stl.q0*np.square(DistributionToggles.v_para_space)/2)*np.gradient(f0[0][perpIdx], DistributionToggles.v_para_space)

    # Fill in the rest of the times with the f0
    for tme in range(1, len(data_dict_flux['time'][0])):
        f0[tme] = f0[0]
        f0_df_dvE[tme] = f0_df_dvE[0]

    # --- correlate the results ---
    instant_correlation = np.zeros(shape=(Ntimes ,Nvperps, Nvparas))
    instant_correlation_f0 = np.zeros(shape=(Ntimes, Nvperps, Nvparas))
    FPC = np.zeros(shape=(Nvperps,Nvparas))
    FPC_f0 = np.zeros(shape=(Nvperps,Nvparas))
    FPC_residual = np.zeros(shape=(Nvperps,Nvparas))

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

    # print(f'\n{Nvperps*Nvparas} Number of Iterations')
    for perpIdx, paraIdx in tqdm(itertools.product(*[range(thing) for thing in [Nvperps,Nvparas]])):

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
    FPC_vpara_int = np.zeros(shape=(Nvparas))
    FPC_vpara_int_f0 = np.zeros(shape=(Nvparas))
    FPC_vpara_int_residual = np.zeros(shape=(Nvparas))

    for idx in range(Nvparas):
        FPC_vpara_int[idx] = simpson(FPC[:,idx], DistributionToggles.v_perp_space)
        FPC_vpara_int_f0[idx] = simpson(FPC_f0[:, idx], DistributionToggles.v_perp_space)
        FPC_vpara_int_residual[idx] = simpson(FPC_residual[:, idx], DistributionToggles.v_perp_space)

    # Integrate over v_perp to get the total FPC and add 2pi to cover all other v_perp directions (assuming gyrotropy)
    FPC_total = 2*np.pi*simpson(FPC_vpara_int,DistributionToggles.v_para_space)
    FPC_total_f0 = 2*np.pi*simpson(FPC_vpara_int_f0, DistributionToggles.v_para_space)
    FPC_total_residual = 2*np.pi*simpson(FPC_vpara_int_residual, DistributionToggles.v_para_space)

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
        'df_dvpara' : [np.array(df_dvE), {'DEPEND_0':'time','DEPEND_1':'v_perp','DEPEND_2':'v_para','LABALAXIS':'df/dv_para','UNITS':'m^-3s^-6/m/s','VAR_TYPE':'data'}],
        'v_para': [DistributionToggles.v_para_space,{'UNITS': 'm/s', 'LABLAXIS': 'v_para'}],
        'v_perp': [DistributionToggles.v_perp_space, {'UNITS': 'm/s', 'LABLAXIS': 'v_perp'}],
        'E_mu_corr':[np.array(E_mu_corr),deepcopy(data_dict_distribution['E_mu_obs'][1])],
        'B_perp_corr': [np.array(B_perp_corr),deepcopy(data_dict_distribution['B_perp_obs'][1])],
        'E_perp_corr': [np.array(E_perp_corr),deepcopy(data_dict_distribution['E_perp_obs'][1])],
        'time_waves' : deepcopy(data_dict_distribution['time_waves'])
    }

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