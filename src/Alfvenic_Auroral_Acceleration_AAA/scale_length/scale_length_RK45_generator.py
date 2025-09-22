def scale_length_RK45_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from copy import deepcopy

    # --- File-specific imports ---
    from scipy.integrate import solve_ivp
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_toggles import ScaleLengthToggles as toggles

    start_time = time.time()

    # prepare the output
    data_dict_output = {
                        'time': [[], {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
                        'mu_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&mu; wave','VAR_TYPE':'data'}],
                        'chi_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&chi; wave','VAR_TYPE':'data'}],
                        'phi_w': [[], {'DEPEND_0':'time','UNITS': 'deg', 'LABLAXIS': '&phi; wave','VAR_TYPE':'data'}],
                        'k_perp': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&perp;!N', 'VAR_TYPE': 'data'}],
                        'k_para': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&parallel;!N', 'VAR_TYPE': 'data'}],
                        'r': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': 'r', 'VAR_TYPE': 'data'}],
                        'colat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Colatitude &theta;', 'VAR_TYPE': 'data'}],
                        'lat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Geomagnetic Latitude 90-&theta;', 'VAR_TYPE': 'data'}],
                        'z': [[], {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'altitude', 'VAR_TYPE': 'data'}],
                        'lambda_para0': [np.array([toggles.Lambda_para0/stl.m_to_km]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                        'lambda_perp0': [np.array([toggles.Lambda_perp0/stl.m_to_km]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    stl.prgMsg('Importing Plasma Environment Functions')
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import envFunc
    stl.Done(start_time)

    ###################################
    # --- IMPLEMENT THE RK45 Solver ---
    ###################################
    stl.prgMsg('Solving Scale Length IVP')

    def ray_equations(t, S):

        # Initial Conditions
        kpara, kperp, mu, chi = S[0], S[1], S[2], S[3]

        # Ray equation 1 - k_parallel
        dkpara = (envFunc.func_scale_dkpara(mu, chi) * (kpara*envFunc.func_V_A(mu, chi))/np.sqrt(1 + np.power(kperp*envFunc.func_lbda(mu, chi),2))) * ((kperp**2*envFunc.func_lbda(mu, chi)/( (1 + np.power(kperp*envFunc.func_lbda(mu, chi),2))**(3/2)))*envFunc.func_pDD_mu_lbda(mu, chi) - (1/envFunc.func_V_A(mu, chi)) *(envFunc.func_pDD_mu_V_A(mu, chi)) )

        dkperp = (envFunc.func_scale_dkperp(mu, chi) * (kpara*envFunc.func_V_A(mu, chi))/np.sqrt(1 + np.power(kperp*envFunc.func_lbda(mu, chi),2))) * ((kperp**2*envFunc.func_lbda(mu, chi)/( (1 + np.power(kperp*envFunc.func_lbda(mu, chi),2))**(3/2)))*envFunc.func_pDD_chi_lbda(mu, chi) - (1/envFunc.func_V_A(mu, chi)) *(envFunc.func_pDD_chi_V_A(mu, chi)) )

        dmu = envFunc.func_scale_dmu(mu,chi) * (envFunc.func_V_A(mu, chi))/np.sqrt(1 + np.power(kperp*envFunc.func_lbda(mu, chi),2))

        dchi = envFunc.func_scale_dchi(mu, chi) * (kpara*kperp*envFunc.func_V_A(mu, chi)*np.power(envFunc.func_lbda(mu,chi),2))/((1 + np.power(kperp*envFunc.func_lbda(mu, chi),2)**(3/2)))

        dS = [dkpara, dkperp, dmu, dchi]

        return dS

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0):

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=ray_equations,
                         t_span=t_span,
                         y0=s0,
                         # t_eval=toggles.RK45_eval,
                         method=toggles.RK45_method,
                         rtol=toggles.RK45_rtol,
                         atol=toggles.RK45_atol)
        T = soln.t
        Kpara = soln.y[0, :]
        Kperp = soln.y[1, :]
        Mu = soln.y[2, :]
        Chi = soln.y[3, :]
        return [T, Kpara, Kperp,Mu,Chi]

    [T, Kpara, Kperp,Mu,Chi] = my_RK45_solver(toggles.RK45_tspan, toggles.s0)
    stl.Done(start_time)

    # --- store the output ---
    data_dict_output['time'][0] = np.array(T)
    data_dict_output['k_para'][0] = np.array(Kpara)
    data_dict_output['k_perp'][0] = np.array(Kperp)
    data_dict_output['mu_w'][0] = np.array(Mu)
    data_dict_output['chi_w'][0] = np.array(Chi)

    # Convert output to geophysical coordinates
    def r_muChi(mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            distance along geomagnetic field line, measure from earth's surface in [km]
        '''

        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.power(zeta, 2) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)

        r = u / chi  # in R_E from earth's center

        return r

    def theta_muChi(mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            distance along geomagnetic field line in [m]
        '''
        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.power(zeta, 2) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)
        return np.degrees(np.arcsin(np.sqrt(u)))

    data_dict_output['colat'][0] = theta_muChi(deepcopy(data_dict_output['mu_w'][0]),deepcopy(data_dict_output['chi_w'][0]))
    data_dict_output['lat'][0] = 90 - deepcopy(data_dict_output['colat'][0])
    data_dict_output['r'][0] = r_muChi(deepcopy(data_dict_output['mu_w'][0]), deepcopy(data_dict_output['chi_w'][0]))
    data_dict_output['z'][0] = (deepcopy(data_dict_output['r'][0])-1)*stl.Re

    # add the parallel/perp wavelength
    data_dict_output = {**data_dict_output,
                        **{
                            'lambda_para': [2*np.pi/data_dict_output['k_para'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                            'lambda_perp': [2*np.pi/data_dict_output['k_perp'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                        }}

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}\scale_length.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)