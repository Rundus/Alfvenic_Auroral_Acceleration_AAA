import numpy as np
import spaceToolsLib as stl

class ScaleLengthToggles:

    # --- RK45 solver toggles ---
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-12
    RK45_atol = 1E-12
    RK45_tspan = [0,20] # time range (in seconds)
    RK45_eval = np.linspace(0,20,1000+1)

    # Initial Wave conditions
    Lambda_para0 = 100*1000 # in meters
    Lambda_perp0 = 4*1000 # in meters

    # z0 = 500 # in kilometers
    # Theta0 = 70 # in latitude
    z0 = 20000  # in kilometers
    Theta0 = 70  # in latitude [deg]
    r = 1 + z0/stl.Re
    u0 = -1*np.sqrt(np.cos(np.radians(90-Theta0))) / r
    chi0 = np.power(np.sin(np.radians(90-Theta0)),2)/ r
    s0 = [2*(np.pi)/Lambda_para0, 2*(np.pi)/Lambda_perp0, u0, chi0] # initial conditions [k_para0, k_perp0, mu0, chi0]

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}\scale_length'