# --- general imports ---
import spaceToolsLib as stl
import numpy as np
from copy import deepcopy
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from timebudget import timebudget

# --- File-specific imports ---
from glob import glob
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
from tqdm import tqdm
from scipy.integrate import simpson
import multiprocessing as mp

# --- Load the needed data ---
data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/results/{DistributionToggles.z0_obs}km/ray_equations_{DistributionToggles.z0_obs}km.cdf')[0])

# --- PREPARE PARALLELIZED OUTUTS ---
# prepare some variables
Ntimes = len(data_dict_ray_eqns['time'][0])
Nmus = len(WaveFieldsToggles.mu_grid)

# --- prepare the parallelized outputs ---
mp_arr_1 = mp.Array('d',Ntimes*Nmus)
arr_1 = np.frombuffer(mp_arr_1.get_obj())
Eperp = arr_1.reshape((Ntimes,Nmus))

mp_arr_2 = mp.Array('d',Ntimes*Nmus)
arr_2 = np.frombuffer(mp_arr_2.get_obj())
PotentialPerp = arr_2.reshape((Ntimes,Nmus))

mp_arr_3 = mp.Array('d',Ntimes*Nmus)
arr_3 = np.frombuffer(mp_arr_3.get_obj())
Emu = arr_3.reshape((Ntimes,Nmus))

mp_arr_4 = mp.Array('d',Ntimes*Nmus)
arr_4 = np.frombuffer(mp_arr_4.get_obj())
Bperp = arr_4.reshape((Ntimes,Nmus))

mp_arr_5 = mp.Array('d',Ntimes*Nmus)
arr_5 = np.frombuffer(mp_arr_5.get_obj())
PotentialPara = arr_5.reshape((Ntimes,Nmus))


def waveFields_calculator(tmeIdx):

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################
    for idx in range(Nmus):
        # perform the loop
        time = data_dict_ray_eqns['time'][0][tmeIdx]
        eval_pos = [WaveFieldsToggles.mu_grid[idx], RayEquationToggles.chi0_w, RayEquationToggles.phi0_w]
        PotentialPerp[tmeIdx][idx] = WaveFieldsClasses().field_generator(time, eval_pos, type='potential')
        Eperp[tmeIdx][idx] = WaveFieldsClasses().field_generator(time, eval_pos, type='eperp')
        Emu[tmeIdx][idx] = WaveFieldsClasses().field_generator(time, eval_pos, type='emu')
        Bperp[tmeIdx][idx] = WaveFieldsClasses().field_generator(time, eval_pos, type='bperp')

def ParallelPotential_calculator(tmeIdx):
    # calculate the parallel potential via line integration
    PotentialPara[tmeIdx] = np.array([-1 * simpson(y=Emu[tmeIdx][0:i], x=stl.m_to_km * WaveFieldsToggles.alt_grid[0:i]) if i != 0 else 0 for i in range(len(WaveFieldsToggles.mu_grid))])

@timebudget
def wave_fields_generator():
    # Execute the Louiville Mapping (Parallel Processing)
    processes_count = 32  # Number of CPU cores to commit to this operation
    pool_object = mp.Pool(processes_count)
    inputs = range(Ntimes)
    for _ in tqdm(pool_object.imap_unordered(waveFields_calculator, inputs), total=Ntimes):
        pass

    # Execute the Louiville Mapping (Parallel Processing)
    processes_count = 32  # Number of CPU cores to commit to this operation
    pool_object = mp.Pool(processes_count)
    inputs = range(Ntimes)
    for _ in tqdm(pool_object.imap_unordered(ParallelPotential_calculator, inputs), total=Ntimes):
        pass

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_ray_eqns['time'][0])),deepcopy(data_dict_ray_eqns['time'][1])],
        'mu_w': deepcopy(data_dict_ray_eqns['mu_w']),
        'chi_w': deepcopy(data_dict_ray_eqns['chi_w']),
        'z': deepcopy(data_dict_ray_eqns['z']),
        'E_perp': [np.array(Eperp), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [np.array(Bperp), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [np.array(Emu), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'potential_perp': [np.array(PotentialPerp), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Perpendicular Potential', 'VAR_TYPE': 'data'}],
        'potential_mu': [np.array(PotentialPara), {'DEPEND_0': 'time', 'DEPEND_1': 'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Parallel Potential', 'VAR_TYPE': 'data'}],
        'mu_grid': [WaveFieldsToggles.mu_grid,deepcopy(data_dict_ray_eqns['mu_w'][1])],
        'alt_grid': [WaveFieldsToggles.alt_grid, deepcopy(data_dict_ray_eqns['z'][1])],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}]
    }

    # Calculate the resonance window
    potential_para_max = np.array([np.max(np.abs(data_dict_output['potential_mu'][0][i])) for i in range(len(data_dict_output['potential_mu'][0]))])
    DAW_vel = data_dict_ray_eqns['omega'][0]/deepcopy(data_dict_ray_eqns['k_mu'][0])
    data_dict_output['resonance_high'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel + np.sqrt(2*stl.q0*potential_para_max/stl.m_e))
    data_dict_output['resonance_low'][0] = 0.5*(stl.m_e/stl.q0)*np.square((DAW_vel - np.sqrt(2 * stl.q0 * potential_para_max / stl.m_e)))
    data_dict_output['DAW_velocity'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel)

    ################
    # --- OUTPUT ---
    ################

    # save this particular run
    outputPath = rf'{WaveFieldsToggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/wave_fields_{DistributionToggles.z0_obs}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)