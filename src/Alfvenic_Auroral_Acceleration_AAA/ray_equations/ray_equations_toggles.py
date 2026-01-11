import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles

class RayEquationToggles:

    # --- RK45 solver toggles ---
    RK45_method = 'RK45'
    # RK45_method = 'LSODA'
    RK45_rtol = 1E-10  # controls the relative accuracy. If rtol
    RK45_atol = 1E-15  # controls the absolute accuracy

    # --- Up the Field Line ---
    RK45_N_eval_points_up = 500
    RK45_tspan_up = [0, 3, RK45_N_eval_points_up]  # time range (in seconds)
    RK45_Teval_up = np.linspace(*RK45_tspan_up)

    # --- Down the Field Line ---
    RK45_N_eval_points_down = 200
    RK45_tspan_down = [0, -1, RK45_N_eval_points_down]  # time range (in seconds)
    RK45_Teval_down = np.linspace(*RK45_tspan_down)

    # --- Ray Equation Simulation Boundaries ---
    lower_boundary = 100 # in kilometers

    # --- WAVE INITIAL CONDITIONS ---
    # Initial Wave Frequency
    f_0 = 5
    omega0 = 2 * np.pi * f_0  # in Hz

    # Initial Wave Position
    z0_w = 500  # in kilometers
    Theta0_w = 70  # in geomagnetic latitude
    phi0_w = 0  # in geomagnetic longitude

    # modified dipole coordinates
    r_w = 1 + z0_w / stl.Re
    u0_w = -1 * np.sqrt(np.cos(np.radians(90 - Theta0_w))) / r_w
    chi0_w = np.power(np.sin(np.radians(90 - Theta0_w)), 2) / r_w
    phi0_w = np.radians(phi0_w)

    # Initial Wavelength
    Lambda_perp0 = 4 * 1000  # perpendicular wavelength (in meters) AT THE IONOSPHERE
    perp_ratio = 1.1  # what % of the initial lambda_perp is lambda_phi

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/ray_equations'