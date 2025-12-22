
def distribution_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses

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

    ########################################
    # --- LOOP OVER VELOCITY PHASE SPACE ---
    ########################################

    #####################
    # --- RK45 SOLVER ---
    #####################

    # The
    def equations_of_motion(t, S):
        # State Vector - [mu, chi, phi, v_mu, v_chi, v_phi]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[3]

        # dchi/dt
        DchiDt = S[4]

        # dphi/dt
        DphiDt = S[5]

        # --- Velocity ---

        # dv_phi/dt
        DvmuDt = (-1*stl.q0/stl.m_e) * EField_mu(t,)

        # dv_chi/dt
        DvchiDt = 0

        # dv_phi/dt
        DvphiDt = 0

        dS = [DmuDt, DchiDt, DphiDt, D]

        return dS

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0):

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=ray_equations,
                         t_span=t_span,
                         y0=s0,
                         method=SimToggles.RK45_method,
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol)
        T = soln.t
        K_mu = soln.y[0, :]
        K_chi = soln.y[1, :]
        K_phi = soln.y[2, :]
        Mu = soln.y[3, :]
        Chi = soln.y[4, :]
        Phi = soln.y[5,:]
        return [T, K_mu, K_chi, K_phi, Mu, Chi, Phi]

    [T, K_mu, K_chi, K_phi, Mu, Chi, Phi] = my_RK45_solver(SimToggles.RK45_tspan, s0)

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################
    for key, func in envDict.items():
        data_dict_output[key][0] = func(data_dict_output['mu_w'][0], data_dict_output['chi_w'][0])

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)