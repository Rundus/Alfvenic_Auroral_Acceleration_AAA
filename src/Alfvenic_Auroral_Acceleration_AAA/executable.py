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

if dict_executable['regen_ray_equation_expressions']==1:
    # spatial environment
    print('Regenerating Ray Equation Expressions',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations import ray_equation_expression_generator
    stl.Done(start_time)

if dict_executable['regen_scale_length']==1:
    # spatial environment
    print('Solving IVP for Scale Lengths',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_RK45_generator import scale_length_RK45_generator
    scale_length_RK45_generator()
    stl.Done(start_time)

if dict_executable['regen_plasma_environment']==1:
    # spatial environment
    print('Evaluating Plasma Environment',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_generator import plasma_environment_generator
    plasma_environment_generator()
    stl.Done(start_time)

if dict_executable['regen_wave_fields']==1:
    # Wave Fields
    print('Calculating Wave Fields',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_generator import wave_fields_generator
    wave_fields_generator()
    stl.Done(start_time)

if dict_executable['regen_particle_distributions'] == 1:
    # particle distributions
    print('Calculating Liouville Mapping',end='\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_generator import distribution_generator
    distribution_generator()
    stl.Done(start_time)





