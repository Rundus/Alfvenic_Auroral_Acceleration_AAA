import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
import spaceToolsLib as stl

class FPCToggles:

    # --- velocity space ---
    N_vel_space = 500
    # para_space_temp = np.linspace(np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_min) / stl.m_e), np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_max) / stl.m_e), N_vel_space)
    para_space_temp = np.linspace(0, np.sqrt(2 * stl.q0 * np.power(10, DistributionToggles.E_max) / stl.m_e), N_vel_space)
    v_para_space = np.append(-1*para_space_temp[::-1], para_space_temp[1:])
    v_perp_space = np.linspace(0, np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_max) / stl.m_e), N_vel_space)

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/field_particle_correlation'