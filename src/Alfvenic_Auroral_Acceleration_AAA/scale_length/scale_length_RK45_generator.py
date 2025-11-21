from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses


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
                        'k_chi': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&Chi;!N', 'VAR_TYPE': 'data'}],
                        'k_phi': [[], {'DEPEND_0': 'time', 'UNITS': '1/m', 'LABLAXIS': ' k!B&phi;!N', 'VAR_TYPE': 'data'}],
                        'k_perp': [[], {'DEPEND_0': 'time', 'UNITS': '1/m', 'LABLAXIS': ' k!B&perp;!N', 'VAR_TYPE': 'data'}],
                        'k_mu': [[], {'DEPEND_0':'time','UNITS': '1/m', 'LABLAXIS': ' k!B&mu;!N', 'VAR_TYPE': 'data'}],
                        'r': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': 'r', 'VAR_TYPE': 'data'}],
                        'colat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Colatitude &theta;', 'VAR_TYPE': 'data'}],
                        'lat': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': 'Geomagnetic Latitude 90-&theta;', 'VAR_TYPE': 'data'}],
                        'z': [[], {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'altitude', 'VAR_TYPE': 'data'}],
                        'lambda_mu_0': [np.array([]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&mu;', 'VAR_TYPE': 'data'}],
                        'lambda_chi_0': [np.array([]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&chi;', 'VAR_TYPE': 'data'}],
                        'lambda_phi_0': [np.array([]), {'DEPEND_0': 'time', 'UNITS': 'km', 'LABLAXIS': 'Initial &lambda;!B&phi;', 'VAR_TYPE': 'data'}],
                        'omega':[np.array([]), {'DEPEND_0': 'time', 'UNITS': 'rad/s', 'LABLAXIS': '&omega;', 'VAR_TYPE': 'data'}],
                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()
    V_A = envDict['V_A']
    h_mu = envDict['h_mu']
    h_chi = envDict['h_chi']
    h_phi = envDict['h_phi']
    lmb_e = envDict['lambda_e']
    pDD_mu_lmb_e = envDict['pDD_lambda_e_mu']
    pDD_chi_lmb_e = envDict['pDD_lambda_e_chi']
    pDD_mu_V_A =envDict['pDD_V_A_mu']
    pDD_chi_V_A = envDict['pDD_V_A_chi']
    B_dipole = envDict['B_dipole']

    #######################################
    # --- CREATE THE INITIAL CONDITIONS ---
    #######################################

    # Calculate the initial lambda_mu0 from the dispersion relation and initial conditions

    k_perp_0 = 2*np.pi/SimToggles.Lambda_perp0
    k_mu_0 = (SimToggles.omega0*np.sqrt(1+np.square(k_perp_0*lmb_e(SimToggles.u0,SimToggles.chi0))))/V_A(SimToggles.u0,SimToggles.chi0)
    lambda_phi_0 = SimToggles.perp_ratio*SimToggles.Lambda_perp0
    k_phi_0 = 2*np.pi/lambda_phi_0
    lambda_chi_0 = SimToggles.Lambda_perp0*lambda_phi_0/np.sqrt(lambda_phi_0**2 - SimToggles.Lambda_perp0**2)
    k_chi_0 = 2*np.pi/lambda_chi_0

    data_dict_output['lambda_phi_0'][0] = np.array([lambda_phi_0])
    data_dict_output['lambda_chi_0'][0] = np.array([lambda_chi_0])
    data_dict_output['lambda_mu_0'][0] = np.array([2 * np.pi / k_mu_0])

    # print('\nParameters')
    # print('V_A ',V_A(SimToggles.u0,SimToggles.chi0))
    # print('lmb_e ',lmb_e(SimToggles.u0,SimToggles.chi0))
    # print('sqrt(1+k_perp^2lmb_e^2) ',np.sqrt(1+np.square(k_perp_0*lmb_e(SimToggles.u0,SimToggles.chi0))))
    # print('omega_0 ',2*np.pi*V_A(SimToggles.u0,SimToggles.chi0)/SimToggles.omega0)
    # print('\nWaveLengths')
    # print('lambda_phi_0 [km]',0.001*2*np.pi/k_phi_0)
    # print('lambda_chi_0 [km]',0.001*2*np.pi/k_chi_0)
    # print('lambda_mu_0 [km]',0.001*2 * np.pi / k_mu_0)
    # print('lambda_perp [km]', 0.001*2*np.pi*np.sqrt(1/(k_chi_0**2 + k_phi_0**2)))

    # Calculate the initial B0 magnitude
    B0 = B_dipole(SimToggles.u0,SimToggles.chi0)

    # initial conditions [k_mu0,k_chi0,k_phi0, mu0, chi0, phi0]
    s0 = [k_mu_0, k_chi_0, k_phi_0, SimToggles.u0, SimToggles.chi0, SimToggles.phi0]

    ###################################
    # --- IMPLEMENT THE RK45 Solver ---
    ###################################
    stl.prgMsg('Solving Scale Length IVP')

    def calc_k_phi(mu,chi, k_mu,k_chi):

        # Calculate the current k_phi
        # k_phi_current = np.sqrt((np.square(V_A(mu, chi) * k_mu / SimToggles.omega0) - 1) / (lmb_e(mu, chi) ** 2) - k_chi ** 2)

        k_phi_current = np.sqrt((B_dipole(mu,chi)/B0) *(np.square(k_chi_0) + np.square(k_phi_0)) - np.square(k_chi))

        return k_phi_current

    def ray_equations(t, S):

        # Initial Conditions
        k_mu, k_chi, k_phi, mu, chi, phi = S[0], S[1], S[2], S[3], S[4], S[5]

        k_phi_current = calc_k_phi(mu, chi, k_mu, k_chi)

        # Calculate the current k_perp
        k_perp = np.sqrt(np.square(k_phi_current) + np.square(k_chi))

        ########################
        # --- Ray equation 1 ---
        ########################

        # k_mu
        term1 = (np.square(k_perp) * lmb_e(mu,chi)/(1 + np.square(k_perp*lmb_e(mu, chi)))) * pDD_mu_lmb_e(mu,chi)
        term2 = (1/V_A(mu,chi)) * pDD_mu_V_A(mu,chi)
        dk_mu = (1/h_mu(mu, chi)) * SimToggles.omega0 * (term1 - term2)

        # k_chi
        term1 = (np.square(k_perp) * lmb_e(mu,chi)/(1 + np.square(k_perp*lmb_e(mu, chi)))) * pDD_chi_lmb_e(mu,chi)
        term2 = (1/V_A(mu,chi)) * pDD_chi_V_A(mu,chi)
        dk_chi = (1/h_chi(mu, chi)) * SimToggles.omega0 * (term1 - term2)

        # k_phi
        dk_phi = 0

        ########################
        # --- Ray equation 2 ---
        ########################
        # dmu/dt
        dmu = (1/h_mu(mu,chi))*(SimToggles.omega0/k_mu)

        # dchi/dt
        dchi = -1* (1/h_chi(mu, chi))*(SimToggles.omega0) * ((k_chi*np.square(lmb_e(mu,chi)))/(1+np.square(k_perp*lmb_e(mu,chi))))

        # dphi/dt
        dphi = -1 * (1/h_phi(mu, chi)) * (SimToggles.omega0) * ((k_phi * np.square(lmb_e(mu, chi))) / (1 + np.square(k_perp * lmb_e(mu, chi))))

        dS = [dk_mu, dk_chi, dk_phi, dmu, dchi, dphi]

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
    stl.Done(start_time)

    # --- store the output ---
    data_dict_output['time'][0] = np.array(T)
    data_dict_output['k_mu'][0] = np.array(K_mu)
    data_dict_output['k_chi'][0] = np.array(K_chi)
    data_dict_output['mu_w'][0] = np.array(Mu)
    data_dict_output['chi_w'][0] = np.array(Chi)
    data_dict_output['phi_w'][0] = np.array(np.degrees(Phi))
    data_dict_output['k_phi'][0] = np.array([calc_k_phi(data_dict_output['mu_w'][0][i], data_dict_output['chi_w'][0][i], data_dict_output['k_mu'][0][i], data_dict_output['k_chi'][0][i]) for i in range(len(data_dict_output['time'][0]))])
    data_dict_output['k_perp'][0] = np.sqrt(K_chi ** 2 + np.square(deepcopy(data_dict_output['k_phi'][0])))

    data_dict_output['colat'][0] = ScaleLengthClasses.theta_muChi(Mu,Chi)
    data_dict_output['lat'][0] = 90 - deepcopy(data_dict_output['colat'][0])
    data_dict_output['r'][0] = ScaleLengthClasses.r_muChi(Mu,Chi)
    data_dict_output['z'][0] = (deepcopy(data_dict_output['r'][0])-1)*stl.Re
    data_dict_output['omega'][0] = data_dict_output['k_mu'][0] * V_A(Mu,Chi)/np.sqrt(1 + np.square(data_dict_output['k_perp'][0]*lmb_e(Mu,Chi)))

    # add the parallel/perp wavelength
    data_dict_output = {**data_dict_output,
                        **{
                            'lambda_mu': [2*np.pi/data_dict_output['k_mu'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&parallel;', 'VAR_TYPE': 'data'}],
                            'lambda_perp': [2*np.pi/data_dict_output['k_perp'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                            'lambda_phi': [2 * np.pi / data_dict_output['k_phi'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&phi;', 'VAR_TYPE': 'data'}],
                            'lambda_chi': [2 * np.pi / data_dict_output['k_chi'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&chi;', 'VAR_TYPE': 'data'}],
                        }}

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}/scale_length.cdf'
    stl.outputDataDict(outputPath, data_dict_output)