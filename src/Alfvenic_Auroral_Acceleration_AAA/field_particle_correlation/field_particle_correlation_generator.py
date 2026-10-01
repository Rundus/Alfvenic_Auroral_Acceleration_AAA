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
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy.stats import binned_statistic
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles

    # --- Delete the old FPC Files ---
    old_files = glob(f'{RunToggles.sim_data_output_path}/field_particle_correlation/*.cdf*')
    for old_file in old_files:
        os.remove(old_file)

    # --- Load the liouville mapped data ---
    dist_files = glob(rf'{RunToggles.sim_data_output_path}/liouville_mapping/*.cdf*')

    # loop over distribution function files
    for i in tqdm(range(len(dist_files))):


        # Load the file-specific data
        data_dict = stl.loadDictFromFile(dist_files[i])

        # --- calculate the velocity grid ---
        Ntimes = len(data_dict['time'][0])
        Nptchs = len(data_dict['pitch_angle'][0])
        Nengy = len(data_dict['energy'][0])
        sizes = [Ntimes, Nptchs, Nengy]

        # --- Calculate the derivatives ---
        distribution = data_dict['distribution_function'][0].copy()
        energy = data_dict['energy'][0].copy()
        pitch_angle = data_dict['pitch_angle'][0].copy()
        time_waves = data_dict['time_waves'][0].copy()
        time = data_dict['time'][0].copy()

        df_dE = np.gradient(distribution,energy,axis=2, edge_order=2)
        df_dalpha = np.gradient(distribution,np.radians(pitch_angle),axis=1, edge_order=2)

        # --- Determine the correlation interval ---
        E_mu_obs = data_dict['E_mu_obs'][0].copy()
        mask = np.where(np.abs(E_mu_obs) < 1E-9)
        E_mu_obs[mask] = 0

        # find the time "blocks" where waves occur.
        # (a) pad a zero on each end
        nz = (E_mu_obs !=0) # Find where the non-zero values are
        d = np.diff(np.concatenate(([0],nz,[0])))
        starts = np.flatnonzero(d==1) # first index of each wave "block"
        ends = np.flatnonzero(d==-1) - 1 # last index of each wave "block"
        correlation_window_wave_timebase = time_waves[starts[0]:ends[0]+1]

        # (b) bin-average the wave data onto the particle data
        mid = 0.5 * (time[1:] + time[:-1]) # get bins by calculating the deltaT/2 from the particle timestamps
        edges = np.concatenate(([2 * time[0] - mid[0]], mid, [2 * time[-1] - mid[-1]]))
        E_mu_obs_particle_timebase,_,_ = binned_statistic(time_waves, data_dict['E_mu_obs'][0].copy(), bins=edges)

        # (c) calculate the correlation interval
        #TODO: Find a way to calculate the correlation window

        # --- Calculate the instantaneous FPC ---
        alpha = np.radians(pitch_angle[None,:, None])
        E = energy[None,None,:]
        E_mu = E_mu_obs_particle_timebase[:,None,None]
        FPC_instant = E_mu*(stl.q0 * 0.5*np.square(np.cos(alpha)) * np.sqrt(2*E/stl.m_e) * (2*E*df_dE - np.sin(alpha)*df_dalpha))


        # --- Calculate the total FPC ---
        low_idx = np.abs(time-1.9).argmin()
        high_idx = np.abs(time-2.7).argmin()
        deltaT = time[high_idx] - time[low_idx]
        FPC = np.asarray((1/deltaT)*np.sum(FPC_instant[low_idx:high_idx+1],axis=0))
        print(low_idx,high_idx)

        ################
        # --- OUTPUT ---
        ################
        data_dict_output = {
            'time': data_dict['time'].copy(),
            'FPC_total': [np.array(FPC), {'VAR_TYPE': 'data','DEPEND_0': 'pitch_angle','DEPEND_1': 'energy'}],
            'FPC_instant': [np.array(FPC_instant), {'DEPEND_0': 'time','DEPEND_1': 'pitch_angle','DEPEND_2': 'energy', 'VAR_TYPE': 'data'}],
            'E_mu_obs_particle_timebase' : [E_mu_obs_particle_timebase,data_dict['E_mu_obs'][1].copy()],
            'df_dE':[df_dE,{'DEPEND_0': 'time','DEPEND_0': 'pitch_angle','DEPEND_2': 'energy',}],
            'df_dalpha':[df_dalpha,{'DEPEND_0': 'time','DEPEND_0': 'pitch_angle','DEPEND_2': 'energy',}],
            'pitch_angle':data_dict['pitch_angle'].copy(),
            'energy': data_dict['energy'].copy(),
        }

        data_dict_output['E_mu_obs_particle_timebase'][1]['DEPEND_0'] = 'time'

        if RunToggles.store_output:

            # save the results
            mapping_alt = int(re.search(r'_(\d+)km', dist_files[i]).group(1))
            outputPath = rf'{RunToggles.sim_data_output_path}/field_particle_correlation/FPC_{mapping_alt}km.cdf'
            stl.outputDataDict(outputPath, data_dict_output)

    # ################################
    # # --- CALCULATE CORRELATION ----
    # ################################
    # parallel_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    # perp_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    # for tmeIdx, ptchIdx, engyIdx in itertools.product(*[range(item) for item in sizes]):
    #     parallel_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2*stl.q0*DistributionToggles.energy_range_obs[engyIdx]/stl.m_e)*np.cos(np.radians(DistributionToggles.pitch_range_obs[ptchIdx]))
    #     perp_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range_obs[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range_obs[ptchIdx]))
    #
    # # --- interpolate distribution function onto velocity space ---
    # sizes_vspace = [len(FPCToggles.v_perp_space), len(FPCToggles.v_para_space)]
    # Distribution_interp = np.zeros(shape=(len(data_dict_flux['time'][0]), sizes_vspace[0], sizes_vspace[1]))
    #
    # X, Y = np.meshgrid(FPCToggles.v_perp_space, FPCToggles.v_para_space)
    # for tmeIdx in tqdm(range(sizes[0])):
    #     xData = perp_velocity_grid[tmeIdx].flatten()
    #     yData = parallel_velocity_grid[tmeIdx].flatten()
    #     zData = np.array(deepcopy(data_dict_distribution['Distribution_Function'][0][tmeIdx])).flatten()
    #     interp = LinearNDInterpolator(list(zip(xData,yData)), zData)
    #     Distribution_interp[tmeIdx] = interp(X,Y).T
    #
    # Distribution_interp[np.isnan(Distribution_interp)]  = 0
    #
    # # --- calculate the distribution function velocity space gradient ---
    # df_dvE = np.zeros_like(Distribution_interp)
    # print(f'\n {sizes[0]*sizes_vspace[0]} Number of Iterations')
    # for tmeIdx, perpIdx in tqdm(itertools.product(*[range(sizes[0]),range(sizes_vspace[0])])):
    #     df_dvE[tmeIdx, perpIdx] = (stl.q0*np.square(FPCToggles.v_para_space)/2)*np.gradient(Distribution_interp[tmeIdx, perpIdx], FPCToggles.v_para_space)
    #
    # # Calculate the un-perturbed distribution function in velocity space
    # f0 = np.zeros(shape=(len(data_dict_flux['time'][0]), sizes_vspace[0], sizes_vspace[1]))
    #
    # for idx1, vperpVal in enumerate(FPCToggles.v_perp_space):
    #     for idx2, vparaVal in enumerate(FPCToggles.v_para_space):
    #         f0[0][idx1][idx2] = DistributionClasses().mapped_distribution(DistributionToggles.u0_obs,DistributionToggles.chi0_obs, vperpVal, vparaVal)
    #
    # # calculate the gradient in f0
    # f0_df_dvE = np.zeros_like(f0)
    # for perpIdx in range(sizes_vspace[0]):
    #     f0_df_dvE[0, perpIdx] = (stl.q0*np.square(FPCToggles.v_para_space)/2)*np.gradient(f0[0][perpIdx], FPCToggles.v_para_space)
    #
    # # Fill in the rest of the times with the f0
    # for tme in range(1, len(data_dict_flux['time'][0])):
    #     f0[tme] = f0[0]
    #     f0_df_dvE[tme] = f0_df_dvE[0]
    #
    # # --- correlate the results ---
    # instant_correlation = np.zeros(shape=(Ntimes ,sizes_vspace[0], sizes_vspace[1]))
    # instant_correlation_f0 = np.zeros(shape=(Ntimes, sizes_vspace[0], sizes_vspace[1]))
    # FPC = np.zeros(shape=(sizes_vspace))
    # FPC_f0 = np.zeros(shape=(sizes_vspace))
    # FPC_residual = np.zeros(shape=(sizes_vspace))
    #
    # # 1 interpolate the E-Field data onto the particle data timebase
    # E_mu_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['E_mu_obs'][0])
    # B_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['B_perp_obs'][0])
    # E_perp_corr = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0], data_dict_flux['E_perp_obs'][0])
    #
    # # 2 Determine the period of the wave in the data
    # grad = np.gradient(E_mu_corr)
    # finder = np.where(np.abs(grad) > 0)[0]
    # low_idx = finder[0]
    # high_idx = finder[-1]
    # corr_times = deepcopy(data_dict_distribution['time'][0][low_idx:high_idx + 1])
    # tau = corr_times[-1] - corr_times[0]
    #
    # print(f'\n{sizes_vspace[0] * sizes_vspace[1]} Number of Iterations')
    # for perpIdx, paraIdx in tqdm(itertools.product(*[range(thing) for thing in sizes_vspace])):
    #
    #     # 2 cross-correlate the E-Field timeseries
    #     A_term = deepcopy(df_dvE[low_idx:high_idx+1,perpIdx,paraIdx])
    #     A_term_f0 = deepcopy(f0_df_dvE[low_idx:high_idx + 1, perpIdx, paraIdx])
    #     A_term_residual = deepcopy(df_dvE[low_idx:high_idx+1,perpIdx,paraIdx] - f0_df_dvE[low_idx:high_idx + 1, perpIdx, paraIdx])
    #     B_term = -1*E_mu_corr[low_idx:high_idx+1]
    #
    #     instant_correlation[:, perpIdx, paraIdx] = df_dvE[:,perpIdx,paraIdx] * (-1*E_mu_corr)
    #     instant_correlation_f0[:,perpIdx, paraIdx] = f0_df_dvE[:, perpIdx, paraIdx] * (-1 * E_mu_corr)
    #
    #     FPC[perpIdx, paraIdx] = (1/tau)*simpson(y=A_term*B_term,x=corr_times)
    #     FPC_f0[perpIdx, paraIdx] = (1 / tau) * simpson(y=A_term_f0 * B_term, x=corr_times)
    #     FPC_residual[perpIdx,paraIdx] = (1 / tau) * simpson(y=A_term_residual * B_term, x=corr_times)
    #
    # # integrate over Perpendicular Velocity space to get the J dot E for a given inital electron energy
    # FPC_vpara_int = np.zeros(shape=(sizes_vspace[1]))
    # FPC_vpara_int_f0 = np.zeros(shape=(sizes_vspace[1]))
    # FPC_vpara_int_residual = np.zeros(shape=(sizes_vspace[1]))
    #
    # for idx in range(sizes_vspace[1]):
    #     FPC_vpara_int[idx] = simpson(FPC[:,idx], FPCToggles.v_perp_space)
    #     FPC_vpara_int_f0[idx] = simpson(FPC_f0[:, idx], FPCToggles.v_perp_space)
    #     FPC_vpara_int_residual[idx] = simpson(FPC_residual[:, idx], FPCToggles.v_perp_space)
    #
    # # Integrate over v_perp to get the total FPC and add 2pi to cover all other v_perp directions (assuming gyrotropy)
    # FPC_total = 2*np.pi*simpson(FPC_vpara_int,FPCToggles.v_para_space)
    # FPC_total_f0 = 2*np.pi*simpson(FPC_vpara_int_f0, FPCToggles.v_para_space)
    # FPC_total_residual = 2*np.pi*simpson(FPC_vpara_int_residual, FPCToggles.v_para_space)