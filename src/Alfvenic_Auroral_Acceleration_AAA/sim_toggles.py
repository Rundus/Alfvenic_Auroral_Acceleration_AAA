
class SimToggles:
    import numpy as np
    import spaceToolsLib as stl

    # --- RK45 solver toggles ---
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-6 # controls the relative accuracy. If rtol
    RK45_atol = 1E-7 # controls the absolute accuracy
    RK45_tspan = [0, 1.5]  # time range (in seconds)
    RK45_N_eval_points = 10
    RK45_Teval = np.linspace(RK45_tspan[0], RK45_tspan[-1], RK45_N_eval_points)

    # --- WAVE INITIAL CONDITIONS ---

    # Initial Wave Frequency
    f_0 = 5
    omega0 = 2*np.pi*f_0 # in Hz

    # Initial Wave Position
    z0_w = 20000  # in kilometers
    Theta0_w = 70  # in geomagnetic latitude
    phi0_w = 0 # in geomagnetic longitude

    # modified dipole coordinates
    r_w = 1 + z0_w / stl.Re
    u0_w = -1 * np.sqrt(np.cos(np.radians(90 - Theta0_w))) / r
    chi0_w = np.power(np.sin(np.radians(90 - Theta0_w)), 2) / r
    phi0_w = np.radians(phi0_w)

    # Initial Wavelength
    Lambda_perp0 = 4*1000 # perpendicular wavelength (in meters) AT THE IONOSPHERE
    perp_ratio = 1.1 # what % of the initial lambda_perp is lambda_phi

    # --- OBSERVATION INITIAL CONDITIONS ---
    z0_obs = 500 # in kilometers
    Theta0_obs = Theta0_w
    phi0_obs = phi0_w
    r_obs = 1 + z0_obs / stl.Re
    u0_obs = -1 * np.sqrt(np.cos(np.radians(90 - Theta0_obs))) / r_obs
    chi0_obs = np.power(np.sin(np.radians(90 - Theta0_obs)), 2) / r_obs
    phi0_obs = np.radians(phi0_obs)

    # --- File I/O ---
    sim_root_path = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    sim_data_output_path = r'/home/connor/Data/physicsModels/alfvenic_auroral_acceleration_AAA'

    # sim_root_path = r'C:/Users/cfelt/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = r'C:/Data/physicsModels/alfvenic_auroral_acceleration_AAA'
