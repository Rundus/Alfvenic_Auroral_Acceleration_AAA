
def distribution_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from scipy.integrate import solve_ivp

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

    ########################################
    # --- LOOP OVER VELOCITY PHASE SPACE ---
    ########################################

    # Initial State
    i, j, k = 0, 0, 0
    v_perp = np.sqrt((DistributionToggles.vel_chi[j])**2 + (DistributionToggles.vel_phi[k])**2)
    uB = (0.5*stl.m_e*np.power(v_perp,2))/B_dipole(DistributionToggles.u0, DistributionToggles.chi0)
    s0 = [DistributionToggles.u0,
          DistributionToggles.chi0,
          DistributionToggles.phi0,
          DistributionToggles.vel_mu[i],
          DistributionToggles.vel_chi[j],
          DistributionToggles.vel_phi[k]]
    #####################
    # --- RK45 SOLVER ---
    #####################

    # The
    def equations_of_motion(t, S):
        # State Vector - [mu, chi, phi, v_mu, v_chi, v_phi]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[3]/h_factors[0](S[0],S[1])

        # dchi/dt
        DchiDt = S[4]/h_factors[1](S[0],S[1])

        # dphi/dt
        DphiDt = S[5]/h_factors[2](S[0],S[1])

        # --- Velocity ---

        # dv_phi/dt
        # DvmuDt = (-1*stl.q0/stl.m_e) * EField_mu(t,) - (uB/stl.m_e) * (dB_dipole_dmu(S[0],S[1])/h_factors[0](S[0],S[1]))
        DvmuDt = - (uB / stl.m_e) * (dB_dipole_dmu(S[0], S[1]) / h_factors[0](S[0], S[1]))

        # dv_chi/dt
        DvchiDt = 0

        # dv_phi/dt
        DvphiDt = 0

        dS = [DmuDt, DchiDt, DphiDt, DvmuDt, DvchiDt, DchiDt]

        return dS

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0):

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=equations_of_motion,
                         t_span=t_span,
                         y0=s0,
                         method=SimToggles.RK45_method,
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol)
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        particle_phi = soln.y[2, :]
        vel_Mu = soln.y[3, :]
        vel_Chi = soln.y[4, :]
        vel_Phi = soln.y[5, :]
        return [T, particle_mu, particle_chi, particle_phi, vel_Mu, vel_Chi, vel_Phi]

    [T, particle_mu, particle_chi, particle_phi, vel_Mu, vel_Chi, vel_Phi] = my_RK45_solver(SimToggles.RK45_tspan, s0)

    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time': [np.array(T), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'particle_mu': [np.array(particle_mu), {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&mu;', 'VAR_TYPE':'data'}],
        'particle_chi': [np.array(particle_chi), {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&chi;', 'VAR_TYPE':'data'}],
        'particle_phi': [np.array(particle_phi), {'DEPEND_0':'time','UNITS': 'rad', 'LABLAXIS': '&phi;', 'VAR_TYPE':'data'}],
        'vel_mu' : [np.array(vel_Mu), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&mu;!N', 'VAR_TYPE':'data'}],
        'vel_chi': [np.array(vel_Chi), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&chi;!N', 'VAR_TYPE':'data'}],
        'vel_phi': [np.array(vel_Phi), {'DEPEND_0':'time','UNITS': 'm/s', 'LABLAXIS': 'V!B&phi;!N', 'VAR_TYPE':'data'}],
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)