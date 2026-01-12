# ignore warings
import warnings
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
Ntimes = len(DistributionToggles.RK45_Teval)
Nptchs = len(DistributionToggles.pitch_range)
Nengy = len(DistributionToggles.energy_range)
sizes = [Ntimes, Nptchs, Nengy]

# distribution
mp_array_1 = mp.Array('d', Ntimes * Nptchs * Nengy)
arr_1 = np.frombuffer(mp_array_1.get_obj())
Distribution = arr_1.reshape((Ntimes, Nptchs, Nengy))

# Eperp
mp_array_2 = mp.Array('d', Ntimes)
E_perp_obs = np.frombuffer(mp_array_2.get_obj())

# E_mu
mp_array_3 = mp.Array('d', Ntimes)
E_mu_obs = np.frombuffer(mp_array_3.get_obj())

# B_perp
mp_array_4 = mp.Array('d', Ntimes)
B_perp_obs = np.frombuffer(mp_array_4.get_obj())


def louisville_mapping(tmeIdx):
    for ptchIdx, engyIdx in product(*[range(Nptchs), range(Nengy)]):

        # get the initial state vector
        v_perp0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.cos(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_para0 = np.sqrt(2 * stl.q0 * DistributionToggles.energy_range[engyIdx] / stl.m_e) * np.sin(np.radians(DistributionToggles.pitch_range[ptchIdx]))
        v_mu0 = -1 * v_para0
        B0 = B_dipole(DistributionToggles.u0_obs, DistributionToggles.chi0_obs)
        s0 = [DistributionToggles.u0_obs, DistributionToggles.chi0_obs, v_mu0, v_perp0]

        # get the solver arguments
        deltaT = DistributionToggles.RK45_Teval[tmeIdx]
        uB = (0.5 * stl.m_e * np.square(v_perp0)) / B0

        # Perform the RK45 Solver
        [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = DistributionClasses().louivilleMapper(DistributionToggles.RK45_tspan, s0, deltaT, uB)

        ################################
        # --- PERPENDICULAR DYNAMICS ---
        ################################
        # geomagnetic field experienced by particle
        B_mag_particle = B_dipole(deepcopy(particle_mu), deepcopy(particle_chi))
        # mapped_v_perp = v_perp0 * np.sqrt(B_mag_particle / np.array([B0 for i in range(len(B_mag_particle))]))

        # try conservation of energy (wont work for Any kind of acceleration)
        mapped_v_perp = np.sqrt(np.array([(np.square(v_perp0) + np.square(v_mu0)) for i in range(len(particle_vel_Mu))]) - np.square(particle_vel_Mu))

        ####################################################
        # --- UPDATE DISTRIBUTION GRID AT simulation END ---
        ####################################################

        Energy_checker = 0.5*(stl.m_e/stl.q0)*((np.square(v_perp0) + np.square(v_mu0)) - (np.square(mapped_v_perp[-1]) + np.square(particle_vel_Mu[-1])))

        Distribution[tmeIdx][ptchIdx][engyIdx] = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                                  Te=DistributionToggles.Te_PS,
                                                                                  vel_perp=deepcopy(mapped_v_perp[-1]),
                                                                                  vel_para=deepcopy(particle_vel_Mu[-1]))
        a = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                                  Te=DistributionToggles.Te_PS,
                                                                                  vel_perp=deepcopy(mapped_v_perp[-1]),
                                                                                  vel_para=deepcopy(particle_vel_Mu[-1]))
        b = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                                  Te=DistributionToggles.Te_PS,
                                                                                  vel_perp=deepcopy(v_perp0),
                                                                                  vel_para=deepcopy(v_mu0))
        # print(f'{DistributionToggles.pitch_range[ptchIdx]} deg     ',
        #       f'{DistributionToggles.energy_range[engyIdx]} eV     ',
        #       f'{Energy_checker} dE    ',
        #       f'{(a - b)*1E14} df\n\n')

        ##############################
        # --- OBSERVED WAVE FIELDS ---
        ##############################
        eval_pos = [DistributionToggles.u0_obs, DistributionToggles.chi0_obs]
        E_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.RK45_Teval[tmeIdx], eval_pos=eval_pos, type='eperp')
        E_mu_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.RK45_Teval[tmeIdx], eval_pos=eval_pos, type='eMu')
        B_perp_obs[tmeIdx] = WaveFieldsClasses().field_generator(time=DistributionToggles.RK45_Teval[tmeIdx], eval_pos=eval_pos, type='eperp')

# Parallelize the Code
@timebudget
def run_louisville_mapping():
    # Execute the Louiville Parallel Process Mapping
    processes_count = 32  # Number of CPU cores to commit to this operation
    pool_object = mp.Pool(processes_count)
    inputs = range(Ntimes)
    for _ in tqdm(pool_object.imap_unordered(louisville_mapping, inputs), total=Ntimes):
        pass

    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time': [DistributionToggles.RK45_tspan[1] - np.array(DistributionToggles.RK45_Teval),
                 {'UNITS': 's', 'LABLAXIS': 'Time Eval', 'VAR_TYPE': 'data'}],
        'Distribution': [np.array(Distribution), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy',
                                                  'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function',
                                                  'VAR_TYPE': 'data'}],
        'Energy': [np.array(DistributionToggles.energy_range), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
        'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
        'B_perp_obs': [B_perp_obs, {'DEPEND_0': 'time', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_perp_obs': [E_perp_obs, {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu_obs': [E_mu_obs, {'DEPEND_0': 'time', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}]
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)





