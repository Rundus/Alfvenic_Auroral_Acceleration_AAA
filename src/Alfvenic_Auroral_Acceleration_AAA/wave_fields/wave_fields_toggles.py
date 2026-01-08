import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.sim_classes import SimClasses
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
import spaceToolsLib as stl

class WaveFieldsToggles:

    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = 10000 # In Volts. maximum potential in the perpendicular direction

    # --- MU-Plotting Grid ---
    # determine minimum/maximum mu value for the TOP colattitude
    N_mu = 500  # number of points in mu direction
    mu_min, mu_max = [-1, 0]
    mu_grid = np.linspace(mu_min, mu_max, N_mu)
    alt_grid = np.array(stl.Re * (SimClasses.r_muChi(mu_grid, [SimToggles.chi0_w for i in range(len(mu_grid))]) - 1))

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/wave_fields'

    # --- Inverted-V potential ---
    inV_Volts = 5000 # in Volts
    inV_Zmin = 5000  # in km
    inV_Zmax = 10000 # in km