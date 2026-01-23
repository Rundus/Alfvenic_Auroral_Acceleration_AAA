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
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
warnings.filterwarnings("ignore")
start_time = time.time()

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################
# Run the Code for Each observation altitude
for altitude_val in DistributionToggles.Observation_altitudes:

    DistributionToggles.z0_obs = altitude_val

    print('--------------------')
    print(stl.color.RED + f'--- Altitude {DistributionToggles.z0_obs} km ---' + stl.color.END)
    print('--------------------')

    # re-run everything
    if dict_executable['regen_EVERYTHING']==1:
        for key in dict_executable.keys():
            dict_executable[key] = 1

    if dict_executable['regen_environment_expressions']==1:
        print('\n--- Regenerating Ray Equation Expressions ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_generator import environment_expressions_generator
        environment_expressions_generator()

    if dict_executable['regen_ray_equations']==1:
        print('\n--- Solving Ray Equation IVP for scale Length ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_generator import ray_equations_RK45_generator
        ray_equations_RK45_generator()

    if dict_executable['regen_plasma_environment']==1:
        print('\n--- Evaluating Plasma Environment ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_generator import plasma_environment_generator
        plasma_environment_generator()

    if dict_executable['regen_wave_fields']==1:
        print('\n--- Calculating Wave Fields ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_generator import wave_fields_generator
        wave_fields_generator()

    if dict_executable['regen_particle_distributions'] == 1:
        print('\n--- Calculating Liouville Mapping ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_generator import distribution_generator
        distribution_generator()

    if dict_executable['regen_flux_calculation'] ==1:
        print('\n--- Calculating Differential Flux ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.flux.flux_generator import flux_generator
        flux_generator()

    if dict_executable['regen_field_particle_correlation'] == 1:
        print('\n--- Calculating Field-Particle Correlation ---', end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_generator import field_particle_correlation_generator
        field_particle_correlation_generator()

    stl.Done(start_time)





