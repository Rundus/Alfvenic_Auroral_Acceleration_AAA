
class SimToggles:
    import numpy as np
    import spaceToolsLib as stl

    # --- RK45 solver toggles ---
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-10 # controls the relative accuracy. If rtol
    RK45_atol = 1E-6 # controls the absolute accuracy
    RK45_tspan = [0, 3]  # time range (in seconds)
    RK45_N_eval_points = 30
    RK45_Teval = np.linspace(RK45_tspan[0], RK45_tspan[-1], RK45_N_eval_points)

    # --- INITIAL CONDITIONS ---

    # Initial Wave Frequency
    f_0 = 5
    omega0 = 2*np.pi*f_0 # in Hz

    # Initial Wave Position
    z0 = 500  # in kilometers
    Theta0 = 70  # in geomagnetic latitude
    phi0 = 0 # in geomagnetic longitude

    # modified dipole coordinates
    r = 1 + z0 / stl.Re
    u0 = -1 * np.sqrt(np.cos(np.radians(90 - Theta0))) / r
    chi0 = np.power(np.sin(np.radians(90 - Theta0)), 2) / r
    phi0 = np.radians(phi0)

    # Initial Wavelength
    Lambda_perp0 = 4*1000 # perpendicular wavelength (in meters) AT THE IONOSPHERE
    perp_ratio = 1.1 # what % of the initial lambda_perp is lambda_phi

    # --- File I/O ---
    sim_root_path = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    sim_data_output_path = r'/home/connor/Data/physicsModels/alfvenic_auroral_acceleration_AAA'

    # sim_root_path = r'C:/Users/cfelt/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = r'C:/Data/physicsModels/alfvenic_auroral_acceleration_AAA'
