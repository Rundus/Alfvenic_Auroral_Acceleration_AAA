# ignore warings
import warnings

import matplotlib.pyplot as plt

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
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
from itertools import product
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

    B0 = B_dipole(DistributionToggles.u0_obs, DistributionToggles.chi0_obs)

    for ptchIdx, engyIdx in product(*[range(Nptchs), range(Nengy)]):

        # get the initial state vector
        engyVal = DistributionToggles.energy_range[engyIdx]
        ptchVal = np.radians(DistributionToggles.pitch_range[ptchIdx])
        v_perp0 = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.sin(ptchVal)
        v_para0 = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.cos(ptchVal)
        v_mu0 = -1 * v_para0

        s0 = [DistributionToggles.u0_obs, DistributionToggles.chi0_obs, v_mu0, v_perp0]

        # get the solver arguments
        deltaT = DistributionToggles.obs_times[tmeIdx]
        uB = (0.5 * stl.m_e * np.square(v_perp0)) / B0

        # Perform the RK45 Solver
        [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = DistributionClasses().louivilleMapper([0,-deltaT], s0, deltaT, uB)

        ################################
        # --- PERPENDICULAR DYNAMICS ---
        ################################
        # geomagnetic field experienced by particle
        B_mag_particle = B_dipole(deepcopy(particle_mu), deepcopy(particle_chi))
        mapped_v_perp = v_perp0 * np.sqrt(B_mag_particle / np.array([B0 for i in range(len(B_mag_particle))]))

        ####################################################
        # --- UPDATE DISTRIBUTION GRID AT simulation END ---
        ####################################################

        # if deltaT == 4:
        #     # print(f'Energy: {engyVal} ', f' Pitch: {np.degrees(ptchVal)} ',list(particle_mu), list(particle_vel_Mu))
        #     fig, ax =plt.subplots(6, sharex=True)
        #     fig.suptitle(f'T={deltaT} s, Energy: {engyVal} eV, Pitch: {np.degrees(ptchVal)} deg')
        #     particle_alt = stl.Re * (SimClasses.r_muChi(particle_mu, particle_chi) - 1)
        #     ax[0].plot(T+deltaT,particle_alt )
        #     ax[0].set_ylabel('Alt [km]')
        #
        #     ax[1].plot(T+deltaT,particle_vel_Mu/stl.m_to_km)
        #     ax[1].set_ylabel('$v_{\mu}$ [km/s]')
        #
        #     ax[2].plot(T+deltaT, np.degrees(np.arctan2(mapped_v_perp,-1*particle_vel_Mu)) )
        #     ax[2].set_ylabel(r'$\alpha$ [deg]')
        #     ax[2].axhline(y=90,color='tab:red',alpha=0.5,linestyle='--')
        #
        #     ax[3].plot(T+deltaT,0.5*(stl.m_e/stl.q0)*(np.square(particle_vel_Mu) + np.square(mapped_v_perp)))
        #     ax[3].set_ylabel('Energy [eV]')
        #     ax[3].set_xlabel('Time (t) [s]')
        #
        #     particle_pos = np.array([particle_mu,particle_chi,[RayEquationToggles.phi0_w for i in range(len(T))]]).T
        #     E_mu_particle = np.array([WaveFieldsClasses().field_generator(deltaT+tme, pos, type='emu') for tme,pos in zip(T,particle_pos)])
        #     ax[4].plot(T+deltaT, E_mu_particle)
        #     ax[4].set_xlabel('Time (t) [s]')
        #
        #     wave_value = np.zeros(shape=(len(T),len(WaveFieldsToggles.mu_grid)))
        #     eval_pos = [[WaveFieldsToggles.mu_grid[idx], RayEquationToggles.chi0_w, RayEquationToggles.phi0_w] for idx in range(len(WaveFieldsToggles.mu_grid))]
        #     for idx,tmeVal in enumerate(T):
        #         wave_value[idx] = np.array([WaveFieldsClasses().field_generator(deltaT+tmeVal, pos, type='emu') for pos in eval_pos])
        #
        #     alts = np.array([stl.Re*(SimClasses.r_muChi(mu,RayEquationToggles.chi0_w)-1) for mu in WaveFieldsToggles.mu_grid])
        #     ax[5].pcolormesh(deltaT+T, alts, wave_value.T, cmap='bwr')
        #     ax[5].set_xlabel('time')
        #     ax[5].set_ylabel('alt')
        #     ax[5].plot(deltaT + T, particle_alt, 'ro')
        #
        #
        #     for i in range(6):
        #         ax[i].invert_xaxis()
        #     plt.show()

        Distribution[tmeIdx][ptchIdx][engyIdx] = DistributionClasses().mapped_distribution(mu=particle_mu[-1],
                                                                                           chi=particle_chi[-1],
                                                                                           vel_perp=mapped_v_perp[-1],
                                                                                           vel_para=-1 * particle_vel_Mu[-1])



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
        'Distribution_Function': [np.array(Distribution), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy', 'UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'Distribution Function', 'VAR_TYPE': 'data'}],
        'Energy': [np.array(DistributionToggles.energy_range), {'UNITS': 'eV', 'LABLAXIS': 'Energy'}],
        'Pitch_Angle': [np.array(DistributionToggles.pitch_range), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle'}],
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





