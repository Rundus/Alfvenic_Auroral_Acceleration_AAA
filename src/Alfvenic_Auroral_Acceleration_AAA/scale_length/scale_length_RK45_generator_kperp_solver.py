def scale_length_RK45_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from copy import deepcopy

    # --- File-specific imports ---
    from scipy.integrate import solve_ivp
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_toggles import ScaleLengthToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    import dill

    start_time = time.time()

    # prepare the output
    data_dict_output = {
                        'time': [[], {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
                        'mu_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&mu; wave','VAR_TYPE':'data'}],
                        'chi_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&chi; wave','VAR_TYPE':'data'}],
                        'phi_w': [[], {'DEPEND_0':'time','UNITS': 'deg', 'LABLAXIS': '&phi; wave','VAR_TYPE':'data'}],
                        'k_perp': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&perp;!N', 'VAR_TYPE': 'data'}],
                        'k_para': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&parallel;!N', 'VAR_TYPE': 'data'}],
                        'omega': [[], {'DEPEND_0': 'time', 'UNITS': 'rad/s', 'LABLAXIS': ' &omega;', 'VAR_TYPE': 'data'}],
                        'r': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': 'r', 'VAR_TYPE': 'data'}],
                        'colat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Colatitude &theta;', 'VAR_TYPE': 'data'}],
                        'lat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Geomagnetic Latitude 90-&theta;', 'VAR_TYPE': 'data'}],
                        'z': [[], {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'altitude', 'VAR_TYPE': 'data'}],
                        'lambda_para0': [[], {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                        'lambda_perp0': [np.array([SimToggles.Lambda_perp0/stl.m_to_km]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&perp;', 'VAR_TYPE': 'data'}],

                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()
    V_A = envDict['V_A']
    scale_dkpara = envDict['scale_dkpara']
    scale_dkperp = envDict['scale_dkperp']
    scale_dmu = envDict['scale_dmu']
    scale_dchi = envDict['scale_dchi']
    lmb_e = envDict['lambda_e']
    pDD_mu_lmb_e = envDict['pDD_lambda_e_mu']
    pDD_chi_lmb_e = envDict['pDD_lambda_e_chi']
    pDD_mu_V_A =envDict['pDD_V_A_mu']
    pDD_chi_V_A = envDict['pDD_V_A_chi']

    ###################################
    # --- IMPLEMENT THE RK45 Solver ---
    ###################################
    stl.prgMsg('Solving Scale Length IVP')

    def ray_equations(t, S):

        # Initial Conditions
        kpara, kperp, mu, chi, omega = S[0], S[1], S[2], S[3], S[4]

        # Ray equation 1 - k_parallel
        # k_parallel
        kpara_term1 = (np.power(kperp,2) * lmb_e(mu,chi)/(1 + np.power(kperp*lmb_e(mu, chi),2))) * pDD_mu_lmb_e(mu,chi)
        kpara_term2 = (1/V_A(mu,chi)) * pDD_mu_V_A(mu,chi)
        dkpara = scale_dkpara(mu, chi) * omega * (kpara_term1 - kpara_term2)

        # k_perp
        kperp_term1 = (np.power(kperp,2) * lmb_e(mu,chi)/(1 + np.power(kperp*lmb_e(mu, chi),2))) * pDD_chi_lmb_e(mu,chi)
        kperp_term2 = (1/V_A(mu,chi)) * pDD_chi_V_A(mu,chi)
        dkperp = scale_dkperp(mu, chi) * omega * (kperp_term1 - kperp_term2)

        # dmu/dt
        dmu = scale_dmu(mu,chi) * V_A(mu, chi)/np.sqrt(1 + np.power(kperp*lmb_e(mu, chi),2) )

        # dchi/dt
        # dchi = -1*scale_dchi(mu, chi) * (kpara*kperp*V_A(mu, chi)*np.power(lmb_e(mu,chi),2))/( (1 + np.power(kperp*lmb_e(mu, chi), 2))**(3/2)  ) # DONT enforce domega/dt

        dchi = -1*omega*scale_dchi(mu, chi) * (kperp * np.power(lmb_e(mu, chi), 2)) / (1 + np.power(kperp * lmb_e(mu, chi), 2))

        domega = 0

        dS = [dkpara, dkperp, dmu, dchi, domega]

        return dS

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0):

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=ray_equations,
                         t_span=t_span,
                         y0=s0,
                         method=SimToggles.methods[SimToggles.wMethod],
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol)
        T = soln.t
        Kpara = soln.y[0, :]
        Kperp = soln.y[1, :]
        Mu = soln.y[2, :]
        Chi = soln.y[3, :]
        omega = soln.y[4,:]
        return [T, Kpara, Kperp, Mu, Chi, omega]

    # Initial Wave position
    r = 1 + SimToggles.z0 / stl.Re
    u0 = -1 * np.sqrt(np.cos(np.radians(90 - SimToggles.Theta0))) / r
    chi0 = np.power(np.sin(np.radians(90 - SimToggles.Theta0)), 2) / r

    # Initial Wave k_Parallel
    Lambda_para0 = (V_A(u0, chi0)/SimToggles.f_0) * (1/np.sqrt(1 + np.power((2 * np.pi / SimToggles.Lambda_perp0) * (lmb_e(u0, chi0)),2)))
    data_dict_output['lambda_para0'][0] = np.array([Lambda_para0])

    # Initial wave state
    s0 = [2 * np.pi / Lambda_para0, 2 * np.pi / SimToggles.Lambda_perp0, u0, chi0, 2*np.pi*SimToggles.f_0]  # initial conditions [k_para0, k_perp0, mu0, chi0]
    stl.Done(start_time)

    # --- Run the Solver ---
    [T, Kpara, Kperp, Mu, Chi, Omega] = my_RK45_solver(SimToggles.RK45_tspan, s0)

    # --- store the output ---
    data_dict_output['time'][0] = np.array(T)
    data_dict_output['k_para'][0] = np.array(Kpara)
    data_dict_output['k_perp'][0] = np.array(Kperp)
    data_dict_output['mu_w'][0] = np.array(Mu)
    data_dict_output['chi_w'][0] = np.array(Chi)
    data_dict_output['omega'][0] = np.array(Omega)

    # Convert to geophysical coordinates
    data_dict_output['colat'][0] = ScaleLengthClasses().theta_muChi(deepcopy(data_dict_output['mu_w'][0]),deepcopy(data_dict_output['chi_w'][0]))
    data_dict_output['lat'][0] = 90 - deepcopy(data_dict_output['colat'][0])
    data_dict_output['r'][0] = ScaleLengthClasses().r_muChi(deepcopy(data_dict_output['mu_w'][0]), deepcopy(data_dict_output['chi_w'][0]))
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