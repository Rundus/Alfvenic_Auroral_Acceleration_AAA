"""
    This is where the run-level toggles are stored
"""

import spaceToolsLib as stl
import numpy as np
import os

class RunToggles:

    # --- Run Identification ---
    run_number = 1

    # --- SubRoutine Options ---
    dict_executable = {
        'regen_EVERYTHING': 0,
        'regen_environment_expressions': 1,
        'regen_spatial_grid': 1,
        'regen_plasma_environment': 1,
        'regen_wave_potentials': 1,
        'animate_wave_potentials': 1,
        'regen_liouville_mapping': 1,
        'regen_detector_flux': 1,
        'plot_detector_flux':1,
        'regen_field_particle_correlation': 0
    }

    # --- FILE I/O ---
    store_output = True
    sim_root_path = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    sim_data_output_path = rf'/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/run_{run_number}'

    # sim_root_path = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = rf'C:/data/alfvenic_auroral_acceleration_AAA/run_{run_number}'

    # sim_root_path = r'C:/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = rf'C:/Data/MODELS/alfvenic_auroral_Acceleration_AAA/run_{run_number}'

class EnvironmentExpressionsToggles:

    def __init__(self):
        self.environment_density_dict ={
                'chaston2006':False,
                'shroeder2021':True, # Note this is EXACTLY the Kletzing & Torbert Model
                'chaston2003_nightside':False,
                'chaston2003_cusp': False,
            }

        # FILE I/O
        self.wDenModel_key = [key for key in self.environment_density_dict.keys() if self.environment_density_dict[key]][0]

class SpatialGridToggles:

    ##################################
    # --- SPATIAL ENVIRONMENT GRID ---
    ##################################
    # DEFINE SIMULATION EXTENT in terms of geophysical parameters
    L_Shell = 8.5
    z_para_min = 100 # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude
    z_para_max = 3.4*stl.Re # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude

    ######################
    # --- MU-Dimension ---
    ######################
    N_mu = 20000  # number of points in mu direction

class PlasmaEnvironmentToggles:

    # Plasma Sheet (Hot)
    Te_PS = 100  # [eV] Temperature of the isotropic Plasma Sheet Distribution
    n0_PS = 0.5  # [cm^-3] Density of the plasma sheet population at the dipole geomagnetic equator
    Emin_PS = 0 # [eV]
    Emax_PS = 1E5  # [eV]

    # Ionosphere/Plasmasphere/Exosphere (Cold)
    Te_cold = 1 # [eV] Temperature of the cold ionospheric plasma up to 20,000 km
    Emin_cold = 0  # [eV]
    Emax_cold = 1E5  # [eV]

    # --- Loss Cone Information ---
    alt_lost = 550  # [km] altitude which any particles which reach this have distribution=0. The exobase is where particles are essentially collisionless

class WavePotentialsToggles:

    # ====================
    # === WAVE TOGGLES ===
    # ====================
    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = -1*2*500  # Amplitude of the potential pulse in the perpendicular direction [in Volts]. Note: The 2* comes
    # from the conversion between a CHARACTERSITIC and ACTUAL potential. On RHS boundary: Φ = Z (W⁺ − W⁻)/2
    # which we specify W⁺ =0, W⁻ = W⁻ = −Φ₀/(v_A \sqrt{1+\lambda k_{\perp}}^{2}), so Φ = Z · (0 + Φ₀/Z)/2 = Φ₀/2
    # Note: The -1* out front is to flip from parallel electric field to modified dipole coordinate electric field

    # Perpendicular Scale at The Ionosphere
    Lambda_perp0 = 3 # [km] Perpendicular scale of wave at Z_min (ionosphere). Mapping using flux tube scaling.

    # Wave Frequeuency
    f_0 = 2 # [Hz] Frequency of injected wave

    driver_dict = {
        'gaussian_pulse':0,
        'gaussian_wavepacket':0,
        'tapered_half_sine_pulse':1,
        'tapered_sine_pulse':0
    }

    # ===========================
    # === BOUNDARY CONDITIONS ===
    # ===========================
    SIGMA_P = 1 # [S] Pedersen Conductance in Ionosphere

    # =============================
    # === RK45 Time Integration ===
    # =============================
    t_start, t_end = 0.0, 8 #[seconds] Time from z_max the wave is allowed to propogate
    n_out = 1000  # number of time-points to store for output
    cfl = 0.4


class LiouvilleToggles:

    #############################
    # --- PARALLEL PROCESSING ---
    #############################
    processes_count = os.cpu_count()-1  # Number of CPU cores to commit to this operation
    # processes_count = 20  # Number of CPU cores to commit to this operation

    #############################
    # --- RK45 solver toggles ---
    #############################
    RK45_method = 'RK45' # 'LSODA'
    RK45_rtol = 1E-8  # controls the relative accuracy.
    RK45_atol = 1E-9  # controls the absolute accuracy
    # RK45_tspan = [0,-10*data_dict_ray_eqns['time'][0][-1]]  # time range (in seconds). MAKE SURE THIS IS REVERSED IN TIME

    ##############################
    # --- PARTICLE OBSERVATION ---
    ##############################

    # --- PHYSICAL TOGGLES ---
    mapping_alts = [700]  # [km] This is the altitude where the Louisville mapping is measured
    upper_termination_altitude = SpatialGridToggles.z_para_max # [km] upper altitude limit where to stop the Rk45 solver
    lower_termination_altitude = SpatialGridToggles.z_para_min # [km] lower altitude limit where to stop the Rk45

    # --- ESA ENERGY/PITCH COORDINATES ---
    N_energy_space_points = 40
    E_max_obs = 4  # the POWER of 10^E_max for the maximum energy
    E_min_obs = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range_obs = np.linspace(0, 180, 12+1)
    # pitch_range_obs = np.linspace(0, 180, 9 + 1)
    energy_range_obs = np.logspace(E_min_obs, E_max_obs, N_energy_space_points)

    # --- ESA particle sampling ---
    time_rez = 0.05 # in seconds
    time_obs_start = 0  # in seconds
    time_obs_end = 5 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1 # number of particle observation points

    # --- Loss Cone ---
    use_loss_cone_bool = False

    ###########################
    # --- WAVE OBSERVATIONS ---
    ###########################

    # --- Observation Wave-Sampling ---
    time_rez_waves = 0.001 # [seconds] deltaT sample rate for the waves

    # --- Injected Wave ---
    injected_wave_time_delay = 10 # [seconds] Time delay added to the wave data to cause it to inject later. Defaults to 0 if set to <=0

# class FPCToggles:
#
#     # --- velocity space ---
#     N_vel_space = 500
#     # para_space_temp = np.linspace(np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_min) / stl.m_e), np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_max) / stl.m_e), N_vel_space)
#     para_space_temp = np.linspace(0, np.sqrt(2 * stl.q0 * np.power(10, DistributionToggles.E_max_obs) / stl.m_e), N_vel_space)
#     v_para_space = np.append(-1*para_space_temp[::-1], para_space_temp[1:])
#     v_perp_space = np.linspace(0, np.sqrt(2 * stl.q0 * np.power(10,DistributionToggles.E_max_obs) / stl.m_e), N_vel_space)
#
#     # --- File I/O ---
#     from src.Alfvenic_Auroral_Acceleration_AAA.runners.sim_toggles import SimToggles
#     outputFolder = f'{SimToggles.sim_data_output_path}/field_particle_correlation'