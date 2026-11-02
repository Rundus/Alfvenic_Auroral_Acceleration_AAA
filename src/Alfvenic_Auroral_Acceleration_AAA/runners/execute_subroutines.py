# --- execute_subroutines.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: execute the AAA code

def run_AAA(path_code_src, path_sim_output, dict_run_settings, dict_subroutines):

    #################
    # --- IMPORTS ---
    #################
    import time
    import spaceToolsLib as stl
    import warnings
    from Alfvenic_Auroral_Acceleration_AAA.runners.executable_classes import ExecutableClasses
    warnings.filterwarnings("ignore")
    start_time = time.time()

    # determine which configuration was chosen
    key_run_config = ExecutableClasses().determine_configuration_key(dict_run_settings)

    # Build the output path of the simulation
    path_sim_output += f'/run_{key_run_config}/'

    # Generate the directories for this run
    ExecutableClasses().generate_run_directories(path_sim_output)

    # Generate the Configuration File for this run
    ExecutableClasses().generate_run_JSON(path_sim_output,key_run_config)

    # Print the Configuration File
    ExecutableClasses().print_run_toggles(path_sim_output,key_run_config)

    # ---------------------------
    # --- EXECUTE SUBROUTINES ---
    # ---------------------------

    # re-run everything
    if dict_subroutines['regen_EVERYTHING']==1:
        for key in dict_subroutines.keys():
            dict_subroutines[key] = 1
        pass

    if dict_subroutines['regen_environment_expressions']==1:
        print('\n--- Regenerating Environment Equation Expressions ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_generator import environment_expressions_generator
        environment_expressions_generator(key_run_config, path_sim_output)

    if dict_subroutines['regen_spatial_grid'] == 1:
        print('\n--- Regenerating Spatial Environment ---', end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.spatial_grid.spatial_grid_generator import spatial_grid_generator
        spatial_grid_generator(key_run_config, path_sim_output)

    if dict_subroutines['regen_plasma_environment']==1:
        print('\n--- Evaluating Plasma Environment ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_generator import plasma_environment_generator
        plasma_environment_generator()

    if dict_subroutines['regen_wave_potentials']==1:
        print('\n--- Calculating Wave Potentials ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_generator import wave_potentials_generator
        wave_potentials_generator()

    if dict_subroutines['animate_wave_potentials']==1:
        print('\n--- Animating Wave Potentials ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_animator import animate_wave_potentials_generator
        animate_wave_potentials_generator()

    if dict_subroutines['regen_liouville_mapping'] == 1:
        print('\n--- Calculating Liouville Mapping ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_generator import liouville_mapping_generator
        liouville_mapping_generator()

    if dict_subroutines['regen_detector_flux'] ==1:
        print('\n--- Calculating Detector Differential Flux ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.detector_flux.detector_flux_generator import detector_flux_generator
        detector_flux_generator()

    if dict_subroutines['plot_detector_flux'] ==1:
        print('\n--- Plotting Detector Differential Flux ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.detector_flux import detector_flux_plotter

    if dict_subroutines['regen_field_particle_correlation'] == 1:
        print('\n--- Calculating Field-Particle Correlation ---', end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_generator import field_particle_correlation_generator
        field_particle_correlation_generator()

    stl.Done(start_time)





