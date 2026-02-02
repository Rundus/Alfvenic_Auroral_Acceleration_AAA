# ignore warings
import warnings

warnings.filterwarnings("ignore")

# --- general imports ---
import spaceToolsLib as stl
import numpy as np
from copy import deepcopy
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from tqdm import tqdm
from timebudget import timebudget

# --- File-specific imports ---
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import WaveFieldsClasses
from itertools import product
import multiprocessing as mp

#################################################
# --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
#################################################
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
B_dipole = envDict['B_dipole']

# --- PREPARE PARALLELIZED OUTPUTS ---
Ntimes = len(DistributionToggles.obs_times)
Nvperps = len(DistributionToggles.v_perp_space)
Nvparas = len(DistributionToggles.v_para_space)
sizes = [Ntimes, Nvperps, Nvparas]

# distribution
mp_array_1 = mp.Array('d', Ntimes * Nvperps * Nvparas)
arr_1 = np.frombuffer(mp_array_1.get_obj())
Distribution = arr_1.reshape((Ntimes, Nvperps, Nvparas))

# Particle Trajectories


def louisville_mapping(tmeIdx):

    B0 = B_dipole(DistributionToggles.u0_obs, DistributionToggles.chi0_obs)

    for perpIdx, paraIdx in product(*[range(Nvperps), range(Nvparas)]):
        # get the initial state vector
        # v_perp0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_perp0 = DistributionToggles.v_perp_space[perpIdx]
        v_para0 = DistributionToggles.v_para_space[paraIdx]
        # v_para0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_mu0 = -1 * v_para0

        s0 = [DistributionToggles.u0_obs, DistributionToggles.chi0_obs, v_mu0, v_perp0]

        # get the solver arguments
        deltaT = DistributionToggles.obs_times[tmeIdx]
        uB = (0.5 * stl.m_e * np.square(v_perp0)) / B0

        # Perform the RK45 Solver
        [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = DistributionClasses().louivilleMapper(DistributionToggles.RK45_tspan, s0, deltaT, uB)

        ################################
        # --- PERPENDICULAR DYNAMICS ---
        ################################
        # geomagnetic field experienced by particle
        B_mag_particle = B_dipole(deepcopy(particle_mu), deepcopy(particle_chi))
        mapped_v_perp = v_perp0 * np.sqrt(B_mag_particle / np.array([B0 for i in range(len(B_mag_particle))]))

        ####################################################
        # --- UPDATE DISTRIBUTION GRID AT simulation END ---
        ####################################################
        Distribution[tmeIdx][perpIdx][paraIdx] = DistributionClasses().Maxwellian(vel_perp=deepcopy(mapped_v_perp[-1]),
                                                                                  vel_para=deepcopy(-1*particle_vel_Mu[-1]))

# Parallelize the Code
@timebudget
def distribution_generator():
    # Execute the Louiville Mapping (Parallel Processing)
    processes_count = 32  # Number of CPU cores to commit to this operation
    pool_object = mp.Pool(processes_count)
    inputs = range(Ntimes)
    for _ in tqdm(pool_object.imap_unordered(louisville_mapping, inputs), total=Ntimes):
        pass

    ##############################
    # --- OBSERVED WAVE FIELDS ---
    ##############################
    # Eperp
    E_perp_obs = np.zeros(shape=(DistributionToggles.N_obs_wave_points))

    # E_mu
    E_mu_obs = np.zeros(shape=(DistributionToggles.N_obs_wave_points))

    # B_perp
    B_perp_obs = np.zeros(shape=(DistributionToggles.N_obs_wave_points))

    for tmeIdx in range(DistributionToggles.N_obs_wave_points):
        eval_pos = [DistributionToggles.u0_obs, DistributionToggles.chi0_obs]
        E_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='eperp')
        E_mu_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='eMu')
        B_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.obs_waves_times[tmeIdx], eval_pos=eval_pos, type='bperp')

    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time': [np.array(DistributionToggles.obs_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
        'time_waves': [np.array(DistributionToggles.obs_waves_times), {'UNITS': 's', 'LABLAXIS': 'Time', 'VAR_TYPE': 'data'}],
        'Distribution_Function': [np.array(Distribution), {'DEPEND_0': 'time', 'DEPEND_1': 'vperp', 'DEPEND_2': 'vpara', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
        'vperp' : [np.array(DistributionToggles.v_perp_space), {'VAR_TYPE': 'data', 'LABLAXIS':'vperp','UNITS':'m/s'}],
        'vpara': [np.array(DistributionToggles.v_para_space), {'VAR_TYPE': 'data', 'LABLAXIS':'vpara','UNITS':'m/s'}],
        # 'Energy': [np.array(DistributionToggles.energy_range), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
        # 'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
        'B_perp_obs': [B_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_perp_obs': [E_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_obs': [E_mu_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}]
    }

    # Save the base run
    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/distributions_{DistributionToggles.z0_obs}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)





