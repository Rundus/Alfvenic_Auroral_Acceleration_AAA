from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses


def distribution_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    import math
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from itertools import product
    from tqdm import tqdm

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()
    B_dipole = envDict['B_dipole']
    dB_dipole_dmu = envDict['dB_dipole_dmu']
    h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]

    # --- PREPARE OUTPUTS ---
    Distribution = np.zeros(shape=(len(SimToggles.RK45_Teval),
                                   len(DistributionToggles.pitch_range),
                                   len(DistributionToggles.energy_range)
                                   ))

    ########################################
    # --- LOOP OVER VELOCITY PHASE SPACE ---
    ########################################
    sizes = [len(SimToggles.RK45_Teval),len(DistributionToggles.pitch_range),len(DistributionToggles.energy_range)]

    print(f'Total Iterations: {sizes[0]*sizes[1]*sizes[2]}')
    for tmeIdx, ptchIdx, engyIdx in tqdm(product(*[range(val) for val in sizes])):

        # get the initial state vector
        v_perp0 = np.sqrt(2*stl.q0*DistributionToggles.energy_range[engyIdx]/stl.m_e) * np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_para0 = np.sqrt(2*stl.q0*DistributionToggles.energy_range[engyIdx]/stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_mu0 = -1*v_para0
        B0 = B_dipole(SimToggles.u0, SimToggles.chi0)
        s0 = [SimToggles.u0,
              SimToggles.chi0,
              v_mu0,
              v_perp0]

        # get the solver arguments
        deltaT = SimToggles.RK45_Teval[tmeIdx]
        uB = (0.5 * stl.m_e * np.power(v_perp0, 2)) / B0

        # Perform the RK45 Solver
        [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = DistributionClasses().louivilleMapper(SimToggles.RK45_tspan, s0, deltaT, uB)

        ################################
        # --- PERPENDICULAR DYNAMICS ---
        ################################
        # geomagnetic field experienced by particle
        B_mag_particle = B_dipole(deepcopy(particle_mu), deepcopy(particle_chi))
        mapped_v_perp = v_perp0 * np.sqrt(B_mag_particle / np.array([B0 for i in range(len(B_mag_particle))]))

        ####################################################
        # --- UPDATE DISTRIBUTION GRID AT simulation END ---
        ####################################################
        Distribution[tmeIdx][ptchIdx][engyIdx] = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                                  Te=DistributionToggles.Te_PS,
                                                                                  vel_perp=deepcopy(mapped_v_perp[-1]),
                                                                                  vel_para=deepcopy(particle_vel_Mu[-1]))


    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time_eval' : [np.array(SimToggles.RK45_Teval),{'UNITS': 's', 'LABLAXIS': 'Time Eval','VAR_TYPE':'data'}],
        'Distribution': [np.array(Distribution), {'DEPEND_0':'time_eval','DEPEND_1':'Pitch_Angle','DEPEND_2':'Energy','UNITS':'m!A-6!Ns!A-3!N','LABLAXIS':'Distribution Function','VAR_TYPE':'data'}],
        'Energy': [np.array(DistributionToggles.energy_range), {'UNITS':'eV', 'LABLAXIS':'Energy'}],
        'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)