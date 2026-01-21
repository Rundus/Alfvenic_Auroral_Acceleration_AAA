# Script to test out how the wave fields equations generate the wave

# Imports
import numpy as np
from tqdm import tqdm
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses
from copy import  deepcopy
import spaceToolsLib as stl
from glob import glob
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles


#################
# --- TOGGLES ---
#################
E_scale = 1000

###########################
# --- [1] Define a grid ---
###########################

# --- Chi-Dimension ---
N_chi = 1  # number of points in chi direction
chi_low, chi_high = [0.108446, 0.108450]
chi_range = np.linspace(chi_low, chi_high, N_chi)

# --- MU-Dimension ---
# determine minimum/maximum mu value for the TOP colattitude
N_mu = 80  # number of points in mu direction
mu_min, mu_max = [-1, -0.3]
mu_range = np.linspace(mu_min, mu_max, N_mu)

# --- Phi-Dimension ---
N_phi = 50
phi_min,phi_max = np.radians(-7), np.radians(7)
phi_range = np.linspace(phi_min,phi_max,N_phi)

# CONSTRUCT THE GRID
chi_grid, mu_grid, phi_grid = np.meshgrid(chi_range,mu_range,phi_range)

#####################################
# --- [2] EVALUATE FIELDS ON GRID ---
#####################################

data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
wave_pos_vector = np.array([data_dict_wavescale['mu_w'][0],data_dict_wavescale['chi_w'][0],data_dict_wavescale['phi_w'][0]]).T

# times = [0.1*(i) for i in range(2)] # in simulation seconds
times = data_dict_wavescale['time'][0] # in simulation seconds
Epara_tracker = np.zeros(shape=(len(times),len(mu_grid)))
Eperp_store = np.zeros(shape=(len(times),len(phi_range),len(mu_range)))
potential_store = np.zeros(shape=(len(times),len(phi_range),len(mu_range)))
Epara_store = np.zeros(shape=(len(times),len(phi_range),len(mu_range)))

for loopidx, time in enumerate(times):
    potential = np.zeros_like(phi_grid)
    Ephi = np.zeros_like(phi_grid)
    Epara = np.zeros_like(phi_grid)

    for i in tqdm(range(len(mu_range))):
        for j in range(len(chi_range)):
            for k in range(len(phi_range)):
                eval_pos = [mu_grid[i,j,k], chi_grid[i,j,k], phi_grid[i,j,k]]
                potential[i][j][k] = WaveFieldsClasses().field_generator(time, eval_pos, type='potential')
                Ephi[i][j][k] = WaveFieldsClasses().field_generator(time, eval_pos, type='eperp')
                Epara[i][j][k] = WaveFieldsClasses().field_generator(time,eval_pos,type='epara')

    Eperp_store[loopidx] = Ephi[:,0,:].T
    Epara_store[loopidx] = Epara[:,0,:].T
    potential_store[loopidx] = potential[:,0,:].T

    idx_tme = np.abs(data_dict_wavescale['time'][0]-time).argmin()
    #############################
    # --- [3] PLOT EVERYTHING ---
    #############################

    chi_choice = (chi_high+chi_low)/2 # in deg
    chi_idx = np.abs(chi_range-chi_choice).argmin()

    mu_yGrid = mu_grid[:,chi_idx,:]
    phi_xGrid = np.degrees(phi_grid[:,chi_idx,:])
    potential_zGrid = potential[:,chi_idx,:]
    ePerp_zGrid = Ephi[:,chi_idx,:]
    ePara_zGrid = Epara[:, chi_idx, :]

    # add the data to the E-para tracker
    phi_idx = np.abs(phi_range-wave_pos_vector[idx_tme][2]).argmin()
    Epara_tracker[loopidx] = ePara_zGrid[:,phi_idx]

data_dict_output = {
    'Potential' : [[], {'DEPEND_0':'time','DEPEND_2':'mu_range','DEPEND_1':'phi_range','UNITS':'V','LABLAXIS':'&Phi;'}],
    'Eperp' : [[],{'DEPEND_0':'time','DEPEND_2':'mu_range','DEPEND_1':'phi_range','UNITS':'mVm','LABLAXIS':'E!B&perp;!N'}],
    'Emu' : [[],{'DEPEND_0':'time','DEPEND_2':'mu_range','DEPEND_1':'phi_range','UNITS':'mV/m','LABLAXIS':'E!B&mu;!N'}],
    'time': [np.array(times),deepcopy(data_dict_wavescale['time'][1])],
    'Epara_tracker' : [[],{'DEPEND_1':'mu_range','DEPEND_0':'time','UNITS':'mV/m','LABLAXIS':'E!B&mu;!N'}],
    'mu_range':[[],{'DEPEND_0':None,'UNITS':None,'LABLAXIS':'&mu;'}],
    'phi_range':[[],{'DEPEND_0':None,'UNITS':'Deg','LABLAXIS':'&phi;'}],
}

data_dict_output['Potential'][0] = np.array(potential_store)
data_dict_output['Eperp'][0] = E_scale*np.array(Eperp_store)
data_dict_output['Emu'][0] = E_scale*np.array(Epara_store)
data_dict_output['Epara_tracker'][0] = E_scale*np.array(Epara_tracker)
data_dict_output['mu_range'][0] = np.array(mu_range)
data_dict_output['phi_range'][0] = np.degrees(np.array(phi_range))

# path = r'C:/Data/physicsModels/alfvenic_auroral_acceleration_AAA/wave_fields/'
path = r'//home/connor/Data/physicsModels/alfvenic_auroral_acceleration_AAA/wave_fields/'
stl.outputDataDict(data_dict=data_dict_output,
                   outputPath=path + '/wave_fields.cdf')
