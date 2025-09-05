def scale_length_RK45_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_toggles import ScaleLengthToggles

    # --- File-specific imports ---
    from scipy.integrate import solve_ivp

    start_time = time.time()

    # prepare the output
    data_dict_output = {
                        'mu_w': [[], {'UNITS': None, 'LABLAXIS': '&mu;','VAR_TYPE':'data'}],
                        'chi_w': [[], {'UNITS': None, 'LABLAXIS': '&chi;','VAR_TYPE':'data'}],
                        'phi_w': [[], {'UNITS': 'deg', 'LABLAXIS': '&phi;','VAR_TYPE':'data'}],
                        'k_perp': [[], {'UNITS': '1/m', 'LABLAXIS': 'k!B;&perp', 'VAR_TYPE': 'data'}],
                        'k_para': [[], {'UNITS': '1/m', 'LABLAXIS': 'k!B;&parallel', 'VAR_TYPE': 'data'}],
                        }


    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    stl.prgMsg('Importing Plasma Environment Functions')
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import PlasmaEnvironment
    stl.Done(start_time)


    ###################################
    # --- IMPLEMENT THE RK45 Solver ---
    ###################################
    stl.prgMsg('Importing Plasma Environment Functions')


    def my_lorenz(t, S, sigma, rho, beta):
        # put your code here
        x, y, z = S[0], S[1], S[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        dS = [dx, dy, dz]
        return dS

    s = np.array([1, 2, 3])
    dS = my_lorenz(0, s, 10, 28, 8 / 3)
    print(dS)  # Should be [10, 23, -6]

    # --- Run the Solver and Plot it ---
    def my_lorenz_solver(t_span, s0, sigma, rho, beta):
        p = (sigma, rho, beta)

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=my_lorenz,
                         t_span=t_span,
                         y0=s0,
                         method='RK45',
                         args=p,
                         rtol=1E-7,
                         atol=1E-7)
        T = soln.t
        X = soln.y[0, :]
        Y = soln.y[1, :]
        Z = soln.y[2, :]
        return [T, X, Y, Z]

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{ScaleLengthToggles.sim_root_path}\scale_length.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)