import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
import spaceToolsLib as stl

class WaveFieldsToggles:

    # --- MU-Plotting Grid ---
    # determine minimum/maximum mu value for the TOP colattitude
    N_mu = 1000  # number of points in mu direction
    mu_min, mu_max = [-0.9, -0.1]
    mu_grid = np.linspace(mu_min, mu_max, N_mu)
    alt_grid = np.array(stl.Re * (SimClasses.r_muChi(mu_grid, [RayEquationToggles.chi0_w for i in range(len(mu_grid))]) - 1))

    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = 250  # Amplitude of the potential pulse in the perpendicular direction [in Volts]. Note: The

    # --- Inverted-V potential ---
    inV_Volts = 5000 # in Volts
    inV_Zmin = 5000  # in km
    inV_Zmax = 10000 # in km

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/wave_fields'