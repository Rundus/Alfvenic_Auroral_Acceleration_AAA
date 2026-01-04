
def distribution_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from scipy.integrate import solve_ivp
    from tqdm import tqdm

    # --- Load the wave simulation data ---
    data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
    data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/*.cdf*')[0])

    # prepare the output
    data_dict_output = {
        'time': deepcopy(data_dict_wavescale['time']),
        'edist': [[], {'DEPEND_0': 'time', 'UNITS': 'm!A-3!Ns!A-6!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
        'vel_mu': [[], {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'v!B&mu;!N', 'VAR_TYPE': 'support_data'}],
        'vel_chi': [[], {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'v!B&chi;!N', 'VAR_TYPE': 'support_data'}],
        'vel_phi': [[], {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'v!B&phi;!N', 'VAR_TYPE': 'support_data'}],
    }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()
    B_dipole = envDict['B_dipole']
    dB_dipole_dmu = envDict['dB_dipole_dmu']
    h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]

    # The
    def equations_of_motion(t, S):
        # State Vector - [mu, v_mu]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[3] / h_factors[0](S[0], S[1])

        # dchi/dt
        DchiDt = S[4] / h_factors[1](S[0], S[1])

        # dphi/dt
        DphiDt = S[5] / h_factors[2](S[0], S[1])

        # --- Velocity ---
        # dv_mu/dt
        # DvmuDt = (-1*stl.q0/stl.m_e) * EField_mu(t,) - (uB/stl.m_e) * (dB_dipole_dmu(S[0],S[1])/h_factors[0](S[0],S[1]))
        DvmuDt = - (uB / stl.m_e) * (dB_dipole_dmu(S[0], S[1]) / h_factors[0](S[0], S[1]))

        # dv_chi/dt
        DvchiDt = 0

        # dv_phi/dt
        DvphiDt = 0

        dS = [DmuDt, DchiDt, DphiDt, DvmuDt, DvchiDt, DvphiDt]

        return dS

    #####################
    # --- RK45 SOLVER ---
    #####################

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0):
        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=equations_of_motion,
                         t_span=t_span,
                         y0=s0,
                         method=SimToggles.RK45_method,
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol,
                         t_eval=SimToggles.RK45_Teval,
                         )
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        particle_phi = soln.y[2, :]
        vel_Mu = soln.y[3, :]
        vel_chi = soln.y[4, :]
        vel_phi = soln.y[5, :]
        return [T, particle_mu, particle_chi, particle_phi, vel_Mu, vel_chi, vel_phi]

    ########################################
    # --- LOOP OVER VELOCITY PHASE SPACE ---
    ########################################

    # output
    Distribution = np.zeros(shape=(len(SimToggles.RK45_Teval),len(DistributionToggles.vel_space_mu_range), len(DistributionToggles.vel_space_perp_range)))
    Energy = np.zeros(shape=(len(SimToggles.RK45_Teval), len(DistributionToggles.vel_space_mu_range), len(DistributionToggles.vel_space_perp_range)))

    for paraIdx in tqdm(range(len(DistributionToggles.vel_space_mu_range))):
        for perpIdx in range(len(DistributionToggles.vel_space_perp_range)):
            v_perp0 = np.sqrt((DistributionToggles.vel_space_mu_range[paraIdx])**2 + (DistributionToggles.vel_space_perp_range[perpIdx])**2)
            B0 = B_dipole(DistributionToggles.u0, DistributionToggles.chi0)
            uB = (0.5*stl.m_e*np.power(v_perp0,2))/B0
            s0 = [DistributionToggles.u0,
                  DistributionToggles.chi0,
                  DistributionToggles.phi0,
                  DistributionToggles.vel_space_mu_range[paraIdx],
                  DistributionToggles.vel_space_perp_range[perpIdx],
                  DistributionToggles.vel_space_perp_range[perpIdx]]

            [T, particle_mu, particle_chi, particle_phi, particle_vel_Mu, particle_vel_chi, particle_vel_phi] = my_RK45_solver(SimToggles.RK45_tspan, s0)

            ################################
            # --- PERPENDICULAR DYNAMICS ---
            ################################
            # geomagnetic field experienced by particle
            B_mag_particle = B_dipole(particle_mu,np.array([DistributionToggles.chi0 for i in range(len(particle_mu))]))
            particle_vel_perp = v_perp0*np.sqrt(B_mag_particle/np.array([B0 for i in range(len(B_mag_particle))]))

            ################
            # --- ENERGY ---
            ################
            Energy[0][paraIdx][perpIdx] = (0.5*stl.m_e*(np.square(particle_vel_perp[0]) + np.square(particle_vel_Mu[0])))/stl.q0 - (0.5*stl.m_e*(np.square(particle_vel_perp[-1]) + np.square(particle_vel_Mu[-1])))/stl.q0


            ####################################################
            # --- UPDATE DISTRIBUTION GRID AT simulation END ---
            ####################################################
            Distribution[0][paraIdx][perpIdx] = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                   Te=DistributionToggles.Te_PS,
                                                                   vel_perp=particle_vel_perp[-1],
                                                                   vel_para=particle_vel_Mu[-1])


    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time': [np.array(T), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'particle_mu': [np.array(particle_mu), {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&mu;', 'VAR_TYPE':'data'}],
        # 'particle_chi': [np.array(particle_chi), {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&chi;', 'VAR_TYPE':'data'}],
        # 'particle_phi': [np.array(particle_phi), {'DEPEND_0':'time','UNITS': 'rad', 'LABLAXIS': '&phi;', 'VAR_TYPE':'data'}],
        'particle_vel_mu' : [np.array(particle_vel_Mu), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&mu;!N', 'VAR_TYPE':'data'}],
        'particle_vel_perp': [np.array(particle_vel_perp), {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'V!B&perp;!N', 'VAR_TYPE': 'data'}],
        'Energy_diff' : [np.array(Energy), {'DEPEND_0':'time','DEPEND_1':'vperp_range','DEPEND_2':'vpara_range', 'UNITS': 'eV', 'LABLAXIS': 'Energy', 'VAR_TYPE': 'data'}],
        'Distribution': [np.array(Distribution), {'DEPEND_0':'time','DEPEND_1':'vperp_range','DEPEND_2':'vpara_range','UNITS':'m!A-6!N!A-3!N','LABLAXIS':'Distribution Function','VAR_TYPE':'data'}],
        'vperp_range': [np.array(DistributionToggles.vel_space_perp_range), {'UNITS':'m/s','LABLAXIS':'V!B&perp;!N'}],
        'vpara_range': [np.array(DistributionToggles.vel_space_mu_range), {'UNITS':'m/s','LABLAXIS':'V!B&parallel;!N'}],
        # 'vel_chi': [np.array(vel_Chi), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&chi;!N', 'VAR_TYPE':'data'}],
        # 'vel_phi': [np.array(vel_Phi), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&phi;!N', 'VAR_TYPE':'data'}],
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)