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

    # --- Load the simulation data ---
    data_dict_flux = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/flux/flux.cdf')[0])

    ################################
    # --- CALCULATE CORRELATION ----
    ################################

    # --- calculate the parallel velocity grid ---
    Ntimes = len(DistributionToggles.obs_times)
    Nptchs = len(DistributionToggles.pitch_range)
    Nengy = len(DistributionToggles.energy_range)
    sizes = [Ntimes, Nptchs, Nengy]
    parallel_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    perp_velocity_grid = np.zeros_like(data_dict_flux['Differential_Energy_Flux'][0])
    for tmeIdx, ptchIdx, engyIdx in itertools.product(*sizes):
        parallel_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2*stl.q0*DistributionToggles.energy_range[engyIdx]/stl.m_e)*np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        perp_velocity_grid[tmeIdx][ptchIdx][engyIdx] = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))

    # --- calculate the distribution function velocity space gradient ---



    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'fpc': [],
    }
    outputPath = rf'{FieldParticleCorrelationToggles.outputFolder}/field_particle_correlation.cdf'
    stl.outputDataDict(outputPath, data_dict_output)