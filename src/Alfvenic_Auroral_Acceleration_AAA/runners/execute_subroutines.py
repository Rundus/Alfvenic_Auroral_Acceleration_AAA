# --- execute_subroutines.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: execute the AAA code

def run_AAA_simulation():

    #################
    # --- IMPORTS ---
    #################
    import time
    import spaceToolsLib as stl
    import warnings
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles,EnvironmentExpressionsToggles,SpatialGridToggles,WavePotentialsToggles, LiouvilleToggles
    import numpy as np
    from Alfvenic_Auroral_Acceleration_AAA.runners.executable_classes import ExecutableClasses
    warnings.filterwarnings("ignore")
    start_time = time.time()

    # Generate the directories for this run
    ExecutableClasses().generate_run_directories()

    # Generate the Configuration File for this run
    ExecutableClasses().generate_run_JSON()

    # ---------------------------
    # --- EXECUTE SUBROUTINES ---
    # ---------------------------

    dict_executable = RunToggles.dict_executable.copy()

    # re-run everything
    if dict_executable['regen_EVERYTHING']==1:
        for key in dict_executable.keys():
            dict_executable[key] = 1
        pass

    if dict_executable['regen_environment_expressions']==1:
        print('\n--- Regenerating Ray Equation Expressions ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_generator import environment_expressions_generator
        environment_expressions_generator()
        ExecutableClasses().update_run_JSON(
            {
                'expression_generator':{
                    'density_model': f'{EnvironmentExpressionsToggles().wDenModel_key}',
                }
            }
        )

    if dict_executable['regen_spatial_grid'] == 1:
        print('\n--- Regenerating Spatial Environment ---', end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.spatial_grid.spatial_grid_generator import spatial_grid_generator
        spatial_grid_generator()
        ExecutableClasses().update_run_JSON(
            {
                'spatial_grid':{
                    'z_para_min_km':SpatialGridToggles.z_para_min,
                    'z_para_max_km': SpatialGridToggles.z_para_max,
                    'Num_points_mu':SpatialGridToggles.N_mu,
                    'L_Shell_init':SpatialGridToggles.L_Shell,
                }
             }
                                            )

    # Verify the density model used is the same as those in the pickle files
    ExecutableClasses().check_density_model()

    if dict_executable['regen_plasma_environment']==1:
        print('\n--- Evaluating Plasma Environment ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_generator import plasma_environment_generator
        plasma_environment_generator()

    if dict_executable['regen_wave_potentials']==1:
        print('\n--- Calculating Wave Potentials ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_generator import wave_potentials_generator
        wave_potentials_generator()
        ExecutableClasses().update_run_JSON(
            {
               'Wave_Potentials':{
                   'Phi0_eV': WavePotentialsToggles.Phi_0,
                   'Lambda_perp0_km': WavePotentialsToggles.Lambda_perp0,
               'f_0_Hz': WavePotentialsToggles.f_0,
                   'Sigma_P':WavePotentialsToggles.SIGMA_P,
               }
            })
    if dict_executable['animate_wave_potentials']==1:
        print('\n--- Animating Wave Potentials ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_animator import animate_wave_potentials_generator
        animate_wave_potentials_generator()

    if dict_executable['regen_liouville_mapping'] == 1:
        print('\n--- Calculating Liouville Mapping ---',end='\n')
        from Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_generator import liouville_mapping_generator
        liouville_mapping_generator()

        ExecutableClasses().update_run_JSON(
            {
                'Liouville_mapping': {
                    'mapping_altitudes':LiouvilleToggles.mapping_alts,
                    'RK45_rtol': LiouvilleToggles.RK45_rtol,
                    'RK45_atol': LiouvilleToggles.RK45_atol,
                    'Observed_energies':LiouvilleToggles.energy_range_obs,
                    'Observed_pitch_angles':LiouvilleToggles.pitch_range_obs,
                    'Observed_time_resolution_seconds':LiouvilleToggles.time_rez,
                    'upper_termination_altitude_km':LiouvilleToggles.upper_termination_altitude,
                    'lower_termination_altitude_km': LiouvilleToggles.lower_termination_altitude,
                }
            })

    if dict_executable['regen_detector_flux'] ==1:
        print('\n--- Calculating Detector Differential Flux ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.detector_flux.detector_flux_generator import detector_flux_generator
        detector_flux_generator()

    if dict_executable['plot_detector_flux'] ==1:
        print('\n--- Plotting Detector Differential Flux ---',end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.detector_flux import detector_flux_plotter

    if dict_executable['regen_field_particle_correlation'] == 1:
        print('\n--- Calculating Field-Particle Correlation ---', end='\n')
        # field_particle_correlation_generator()

        print('\n--- Calculating Field-Particle Correlation (Vel Space) ---', end='\n')
        from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_generator_velspace import field_particle_correlation_generator_vel
        field_particle_correlation_generator_vel()

    stl.Done(start_time)





