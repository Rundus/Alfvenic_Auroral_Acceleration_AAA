# kletzingHu2001.py
"""
    This is where the run-level toggles are stored
"""

import spaceToolsLib as stl
import numpy as np
import os


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
    s_para_min = 100 # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude
    s_para_max = 3.4*stl.Re # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude

    ######################
    # --- MU-Dimension ---
    ######################
    N_mu = 20000  # number of points in mu direction

class PlasmaEnvironmentToggles:

    # Plasma Sheet (Hot)
    Te_PS = 200  # [eV] Temperature of the isotropic Plasma Sheet Distribution
    n0_PS = 0.5  # [cm^-3] Density of the plasma sheet population at the dipole geomagnetic equator
    Emin_PS = 0 # [eV]
    Emax_PS = 1E5  # [eV]

    # Ionosphere/Plasmasphere/Exosphere (Cold)
    Te_cold = 2.5 # [eV] Temperature of the cold ionospheric plasma up to 20,000 km
    Emin_cold = 0  # [eV]
    Emax_cold = 1E5  # [eV]

    # --- Loss Cone Information ---
    use_loss_cone_bool = False
    alt_lost = 100  # [km] altitude which any particles which reach this have distribution=0. The exobase is where particles are essentially collisionless

class WavePotentialsToggles:

    # ====================
    # === WAVE TOGGLES ===
    # ====================
    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = -1*2*100  # Amplitude of the potential pulse in the perpendicular direction [in Volts]. Note: The 2* comes
    # from the conversion between a CHARACTERISTIC and ACTUAL potential. On RHS boundary: Φ = Z (W⁺ − W⁻)/2
    # which we specify W⁺ =0, W⁻ = W⁻ = −Φ₀/(v_A \sqrt{1+\lambda k_{\perp}}^{2}), so Φ = Z · (0 + Φ₀/Z)/2 = Φ₀/2
    # Note: The -1* out front is to flip from parallel electric field to modified dipole coordinate electric field

    # Perpendicular Scale at The Ionosphere
    Lambda_perp0 = 3 # [km] Perpendicular scale of wave at Z_min (ionosphere). Mapping using flux tube scaling.

    # Wave Frequeuency
    f_0 = 4 # [Hz] Frequency of injected wave

    driver_dict = {
        'gaussian_pulse':0,
        'gaussian_wavepacket':0,
        'tapered_half_sine_pulse':1,
        'tapered_sine_pulse':0
    }

    # ===========================
    # === BOUNDARY CONDITIONS ===
    # ===========================
    SIGMA_P = 0.13 # [S] Pedersen Conductance in Ionosphere
    absorbing_ionosphere_bool = False

    # =============================
    # === RK45 Time Integration ===
    # =============================
    t_start, t_end = 0.0, 4 #[seconds] Time from z_max the wave is allowed to propogate
    n_out = 1500  # number of time-points to store for output
    cfl = 0.4

class LiouvilleToggles:

    #############################
    # --- PARALLEL PROCESSING ---
    #############################
    processes_count = os.cpu_count()-1  # Number of CPU cores to commit to this operation

    #############################
    # --- RK45 solver toggles ---
    #############################
    RK45_method = 'RK45' # 'LSODA'
    RK45_rtol = 1E-8  # controls the relative accuracy.
    RK45_atol = 1E-9  # controls the absolute accuracy

    ##############################
    # --- PARTICLE OBSERVATION ---
    ##############################

    # --- PHYSICAL TOGGLES ---
    mapping_alts = [700]  # [km] This is the altitude where the Louisville mapping is measured
    upper_termination_altitude = SpatialGridToggles.s_para_max # [km] upper altitude limit where to stop the Rk45 solver
    lower_termination_altitude = SpatialGridToggles.s_para_min # [km] lower altitude limit where to stop the Rk45

    # --- ESA ENERGY/PITCH COORDINATES ---
    N_energy_space_points = 100
    E_max_obs = 3.6  # the POWER of 10^E_max for the maximum energy
    E_min_obs = 1  # the POWER of 10^E_min for the minimum energy
    pitch_range_obs = np.linspace(5, 175, 10+1)
    # pitch_range_obs = np.linspace(0, 180, 18 + 1)
    energy_range_obs = np.logspace(E_min_obs, E_max_obs, N_energy_space_points)

    # --- ESA particle sampling ---
    time_rez = 0.025 # in seconds
    time_obs_start = 0  # in seconds
    time_obs_end = 4 # in seconds
    N_obs_points = int(time_obs_end/time_rez)+1 # number of particle observation points

    ###########################
    # --- WAVE OBSERVATIONS ---
    ###########################

    # --- Observation Wave-Sampling ---
    time_rez_waves = 0.001 # [seconds] deltaT sample rate for the waves

    # --- Injected Wave ---
    injected_wave_time_delay = 0 # [seconds] Time delay added to the wave data to cause it to inject later. Defaults to 0 if set to <=0

class DetectorFluxToggles:

    use_esa_specs_bool = False
    count_threshold = 1 # count level required for flux to output non-zero value
    esa_geometric_factor = 1.74E-4
    esa_deadtime = 674E-9 # [seconds]
    esa_acqusition_time = 0.9E-3 # [seconds]

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