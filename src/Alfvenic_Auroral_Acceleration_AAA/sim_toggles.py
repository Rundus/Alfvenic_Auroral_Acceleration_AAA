
class SimToggles:
    import numpy as np
    import spaceToolsLib as stl

    # --- RK45 solver toggles ---
    # RK45_method = 'RK45' # 'LSODA'
    RK45_method = 'LSODA'
    RK45_rtol = 1E-16 # controls the relative accury. If rtol
    RK45_atol = 1E-18 # controls the absolute accuracy
    RK45_tspan = [0, 20]  # time range (in seconds)

    # Initial Wave conditions
    Lambda_para0 = 100 * 1000  # in meters
    Lambda_perp0 = 5 * 1000  # in meters

    # Initial Wave Position
    z0 = 500  # in kilometers
    Theta0 = 70  # in latitude

    r = 1 + z0 / stl.Re
    u0 = -1 * np.sqrt(np.cos(np.radians(90 - Theta0))) / r
    chi0 = np.power(np.sin(np.radians(90 - Theta0)), 2) / r
    s0 = [2 * (np.pi) / Lambda_para0, 2 * (np.pi) / Lambda_perp0, u0, chi0]  # initial conditions [k_para0, k_perp0, mu0, chi0]

    # --- File I/O ---
    sim_root_path = r'C:\Users\cfelt\PycharmProjects\Alfvenic_Auroral_Acceleration_AAA\src\Alfvenic_Auroral_Acceleration_AAA'
    sim_data_output_path = r'C:\Data\physicsModels\alfvenic_auroral_acceleration_AAA'
