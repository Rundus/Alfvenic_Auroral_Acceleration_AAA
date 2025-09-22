# --- executable.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles

#################
# --- IMPORTS ---
#################
import time
import spaceToolsLib as stl
import warnings
from src.Alfvenic_Auroral_Acceleration_AAA.executable_toggles import dict_executable
warnings.filterwarnings("ignore")
start_time = time.time()

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################
# re-run everything
if dict_executable['regen_EVERYTHING']==1:
    for key in dict_executable.keys():
        dict_executable[key] = 1


if dict_executable['regen_scale_length']==1:
    # spatial environment
    print('Solving IVP for Scale Lengths',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_RK45_generator import scale_length_RK45_generator
    scale_length_RK45_generator()
    stl.Done(start_time)






