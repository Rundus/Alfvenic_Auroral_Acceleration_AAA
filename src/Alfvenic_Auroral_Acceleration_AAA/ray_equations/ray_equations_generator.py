from timebudget import timebudget

@timebudget
def ray_equations_RK45_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from copy import deepcopy

    # --- File-specific imports ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_classes import SimClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_classes import RayEquationsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles

    start_time = time.time()

    # prepare the output
    data_dict_output = {
                        'time': [[], {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
                        'mu_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&mu; wave','VAR_TYPE':'data'}],
                        'chi_w': [[], {'DEPEND_0':'time','UNITS': None, 'LABLAXIS': '&chi; wave','VAR_TYPE':'data'}],
                        'phi_w': [[], {'DEPEND_0':'time','UNITS': 'rad', 'LABLAXIS': '&phi; wave','VAR_TYPE':'data'}],
                        'phi_w_deg': [[], {'DEPEND_0': 'time', 'UNITS': 'deg', 'LABLAXIS': '&phi; wave', 'VAR_TYPE': 'data'}],
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
                        'omega_calc': [np.array([]), {'DEPEND_0': 'time', 'UNITS': 'rad/s', 'LABLAXIS': '&omega;', 'VAR_TYPE': 'data'}],
                        }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
    V_A = envDict['V_A']
    lmb_e = envDict['lambda_e']

    #######################################
    # --- CREATE THE INITIAL CONDITIONS ---
    #######################################

    # Calculate the initial lambda_mu0 from the dispersion relation and initial conditions
    k_perp_0 = 2*np.pi/RayEquationToggles.Lambda_perp0
    k_mu_0 = (RayEquationToggles.omega0*np.sqrt(1+np.square(k_perp_0*lmb_e(RayEquationToggles.u0_w,RayEquationToggles.chi0_w))))/V_A(RayEquationToggles.u0_w,RayEquationToggles.chi0_w)
    lambda_phi_0 = RayEquationToggles.perp_ratio*RayEquationToggles.Lambda_perp0
    k_phi_0 = 2*np.pi/lambda_phi_0
    lambda_chi_0 = RayEquationToggles.Lambda_perp0*lambda_phi_0/np.sqrt(lambda_phi_0**2 - RayEquationToggles.Lambda_perp0**2)
    k_chi_0 = 2*np.pi/lambda_chi_0

    # initial conditions [k_mu0,k_chi0,k_phi0, mu0, chi0_w, phi0_w]
    s0 = [k_mu_0, k_chi_0, k_phi_0, RayEquationToggles.u0_w, RayEquationToggles.chi0_w, RayEquationToggles.phi0_w, RayEquationToggles.omega0]

    ###################################
    # --- IMPLEMENT THE RK45 Solver ---
    ###################################

    # Format: [T, K_mu, K_chi, K_phi, Mu, Chi, Phi, Omega]
    Solution = [[] for i in range(8)]
    UpSolution = RayEquationsClasses().ray_equation_RK45_solver(t_span=RayEquationToggles.RK45_tspan_up[0:2], s0=s0, k_perp_0=k_perp_0)
    DownSolution = RayEquationsClasses().ray_equation_RK45_solver(t_span=RayEquationToggles.RK45_tspan_down[0:2], s0=s0, k_perp_0=k_perp_0)

    # Combine the two solutions into one
    for i in range(len(Solution)):
        if i == 0: # For the TimeRange
            Down_actual = DownSolution[i] - DownSolution[i][-1]
            Down = Down_actual[::-1]
            Up = np.array(UpSolution[i][1:]) - DownSolution[i][-1]
        else:
            Up = UpSolution[i][1:]
            Down = DownSolution[i][::-1]

        Solution[i] = np.append(Down, Up)

    ##########################
    # --- STORE THE OUTPUT ---
    ##########################
    # Format: [T, K_mu, K_chi, K_phi, Mu, Chi, Phi, Omega]
    data_dict_output['lambda_phi_0'][0] = np.array([lambda_phi_0])
    data_dict_output['lambda_chi_0'][0] = np.array([lambda_chi_0])
    data_dict_output['lambda_mu_0'][0] = np.array([2 * np.pi / k_mu_0])
    data_dict_output['time'][0] = np.array(Solution[0]) # REVERSE THE TIME SO THE WAVE PROPAGATES EARTHWARD
    data_dict_output['k_mu'][0] = np.array(Solution[1])
    data_dict_output['k_chi'][0] = np.array(Solution[2])
    data_dict_output['k_phi'][0] = np.array(Solution[3])
    data_dict_output['mu_w'][0] = np.array(Solution[4])
    data_dict_output['chi_w'][0] = np.array(Solution[5])
    data_dict_output['phi_w'][0] = np.array(Solution[6])
    data_dict_output['phi_w_deg'][0] = np.array(np.degrees(deepcopy(Solution[6])))
    Mu, Chi, Omega = Solution[4], Solution[5], Solution[7]
    data_dict_output['k_perp'][0] = np.array([RayEquationsClasses().calc_k_perp(Mu[i],Chi[i], k_perp_0) for i in range(len(Mu))])
    data_dict_output['colat'][0] = SimClasses.theta_muChi(Mu,Chi)
    data_dict_output['lat'][0] = 90 - deepcopy(data_dict_output['colat'][0])
    data_dict_output['r'][0] = SimClasses.r_muChi(Mu,Chi)
    data_dict_output['z'][0] = (deepcopy(data_dict_output['r'][0])-1)*stl.Re
    data_dict_output['omega'][0] = np.array(Omega)
    data_dict_output['omega_calc'][0] = data_dict_output['k_mu'][0] * V_A(Mu,Chi)/np.sqrt(1 + np.square(data_dict_output['k_perp'][0]*lmb_e(Mu,Chi)))

    # add the parallel/perp wavelength
    data_dict_output = {**data_dict_output,
                        **{
                            'lambda_mu': [2 * np.pi/data_dict_output['k_mu'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&parallel;', 'VAR_TYPE': 'data'}],
                            'lambda_perp': [2 * np.pi/data_dict_output['k_perp'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&perp;', 'VAR_TYPE': 'data'}],
                            'lambda_phi': [2 * np.pi / data_dict_output['k_phi'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&phi;', 'VAR_TYPE': 'data'}],
                            'lambda_chi': [2 * np.pi / data_dict_output['k_chi'][0], {'DEPEND_0': 'time', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&chi;', 'VAR_TYPE': 'data'}],
                        }}

    # "Reverse time" on all the variables by flipping everything and time shifting time
    for key in data_dict_output.keys():
        if key == 'time':
            data_dict_output[key][0] = data_dict_output[key][0][-1] - data_dict_output[key][0][::-1]
        else:
            data_dict_output[key][0] = data_dict_output[key][0][::-1]

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{RayEquationToggles.outputFolder}/ray_equations.cdf'
    stl.outputDataDict(outputPath, data_dict_output)