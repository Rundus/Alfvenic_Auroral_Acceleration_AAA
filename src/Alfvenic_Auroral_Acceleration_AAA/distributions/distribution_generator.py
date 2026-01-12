# ignore warings
import warnings

from src.Alfvenic_Auroral_Acceleration_AAA.sim_classes import SimClasses

warnings.filterwarnings("ignore")

# --- general imports ---
import spaceToolsLib as stl
import numpy as np
from copy import deepcopy

# --- File-specific imports ---
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import WaveFieldsClasses
from itertools import product
from tqdm import tqdm
from timebudget import timebudget
import multiprocessing as mp



#################################################
# --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
#################################################
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
B_dipole = envDict['B_dipole']

# --- PREPARE PARALLELIZED OUTPUTS ---
Ntimes = len(DistributionToggles.obs_times)
Nptchs = len(DistributionToggles.pitch_range)
Nengy = len(DistributionToggles.energy_range)
sizes = [Ntimes, Nptchs, Nengy]

# distribution
mp_array_1 = mp.Array('d', Ntimes * Nptchs * Nengy)
arr_1 = np.frombuffer(mp_array_1.get_obj())
Distribution = arr_1.reshape((Ntimes, Nptchs, Nengy))


def louisville_mapping(tmeIdx):
    for ptchIdx, engyIdx in product(*[range(Nptchs), range(Nengy)]):

        # get the initial state vector
        v_perp0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_para0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_mu0 = -1 * v_para0
        B0 = B_dipole(DistributionToggles.u0_obs, DistributionToggles.chi0_obs)
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

        # Check if the Last velocity is a valid velocity
        # Note: If the last velocity value in a mapping is invalid b/c teh particle would trigger an event,
        # the simulation just reports a np.nan value. In these cases, just take the next available value
        # print(stl.Re*(SimClasses.r_muChi(particle_mu[-1],DistributionToggles.chi0_obs)-1))
        if stl.Re*(SimClasses.r_muChi(particle_mu[-1],DistributionToggles.chi0_obs)-1) >= 10000:
            Distribution[tmeIdx][ptchIdx][engyIdx] = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                                  Te=DistributionToggles.Te_PS,
                                                                                  vel_perp=deepcopy(mapped_v_perp[-1]),
                                                                                  vel_para=deepcopy(particle_vel_Mu[-1]))
        else:
            Distribution[tmeIdx][ptchIdx][engyIdx] = 0

# Parallelize the Code
@timebudget
def distribution_generator():
    # Execute the Louiville Parallel Process Mapping
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
        'Distribution': [np.array(Distribution), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
        'Energy': [np.array(DistributionToggles.energy_range), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
        'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
        'B_perp_obs': [B_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_perp_obs': [E_perp_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_obs': [E_mu_obs, {'DEPEND_0': 'time_waves', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}]
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)





