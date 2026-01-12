import itertools

from timebudget import timebudget

@timebudget
def field_particle_correlation_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from glob import glob

    # --- File-specific imports ---
    from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_toggles import FieldParticleCorrelationToggles
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
    Distribution_interp = np.zeros(shape=(len(data_dict_flux['time'][0]),
                                          len(FieldParticleCorrelationToggles.v_perp_space),
                                          len(FieldParticleCorrelationToggles.v_para_space)))
    X, Y = np.meshgrid(FieldParticleCorrelationToggles.v_perp_space,FieldParticleCorrelationToggles.v_para_space)
    for tmeIdx in tqdm(range(sizes[0])):
        xData = perp_velocity_grid[tmeIdx].flatten()
        yData = parallel_velocity_grid[tmeIdx].flatten()
        zData = np.array(deepcopy(data_dict_distribution['Distribution'][0][tmeIdx]))
        interp = LinearNDInterpolator(list(zip(xData,yData)), zData.flatten())
        Distribution_interp[tmeIdx] = interp(X,Y).T

    # --- calculate the distribution function velocity space gradient ---
    df_dvE = np.zeros_like(Distribution_interp)
    for tmeIdx in tqdm(range(sizes[0])):
        for perpIdx in range(len(FieldParticleCorrelationToggles.v_perp_space)):
            df_dvE[tmeIdx][perpIdx] = (stl.q0*np.square(FieldParticleCorrelationToggles.v_para_space)/2)*np.gradient(Distribution_interp[tmeIdx][perpIdx], FieldParticleCorrelationToggles.v_para_space)


    # correlate the results
    instant_correlation = np.zeros(shape=(len(FieldParticleCorrelationToggles.v_perp_space),len(FieldParticleCorrelationToggles.v_perp_space)))
    for perpIdx, paraIdx in itertools.product(*[range(sizes[1]),range(sizes[2])]):

        # 1 interpolate the E-Field data onto the particle data timebase
        B_term = np.interp(deepcopy(data_dict_distribution['time'][0]), data_dict_flux['time_waves'][0],data_dict_flux['E_mu_obs'][0])

        # 2 cross-correlate the E-Field timeseries and the weird shit thingy
        A_term = df_dvE[:][paraIdx][paraIdx]

        instant_correlation[perpIdx][paraIdx] = np.mean(signal.correlate(A_term, B_term))




    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time':deepcopy(data_dict_distribution['time']),
        'instant_correlation':[np.array(instant_correlation),{'DEPEND_1':'v_perp','DEPEND_0':'v_para','VAR_TYPE':'data'}],
        'Distribution_Function' :[np.array(Distribution_interp),deepcopy(data_dict_distribution['Distribution'][1])],
        'df_dvpara' : [np.array(df_dvE), {'DEPEND_0':'time','DEPEND_1':'v_perp','DEPEND_2':'v_para','LABALAXIS':'df/dv_para','UNITS':'m^-3s^-6/m/s','VAR_TYPE':'data'}],
        'v_para': [FieldParticleCorrelationToggles.v_para_space,{'UNITS': 'm/s', 'LABLAXIS': 'v_para'}],
        'v_perp': [FieldParticleCorrelationToggles.v_perp_space, {'UNITS': 'm/s', 'LABLAXIS': 'v_perp'}],
        'E_mu_obs':deepcopy(data_dict_distribution['E_mu_obs']),
        'B_perp_obs': deepcopy(data_dict_distribution['B_perp_obs']),
        'E_perp_obs': deepcopy(data_dict_distribution['E_perp_obs']),
        'time_waves' : deepcopy(data_dict_distribution['time_waves'])
    }
    data_dict_output['Distribution_Function'][1]['DEPEND_1'] = 'v_perp'
    data_dict_output['Distribution_Function'][1]['DEPEND_2'] = 'v_para'
    outputPath = rf'{FieldParticleCorrelationToggles.outputFolder}/field_particle_correlation.cdf'
    stl.outputDataDict(outputPath, data_dict_output)