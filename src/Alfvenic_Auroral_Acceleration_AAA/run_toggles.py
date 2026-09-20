
"""
    This is where the run-level toggles are stored
"""

class RunToggles:

    # --- Run Identification ---
    run_number = 0

    # --- SubRoutine Options ---
    dict_executable = {
        'regen_EVERYTHING': 0,
        'regen_environment_expressions': 0,
        'regen_spatial_grid': 0,
        'regen_plasma_environment': 0,
        'regen_wave_potentials': 0,
        'animate_wave_potentials': 0,
        'regen_liouville_mapping': 1,
        'regen_detector_flux': 1,
        'regen_field_particle_correlation': 0
    }

    # --- FILE I/O ---
    store_output = True
    # sim_root_path = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = rf'/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/run_{run_number}'

    sim_root_path = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    sim_data_output_path = rf'C:/data/alfvenic_auroral_acceleration_AAA/run_{run_number}'

    # sim_root_path = r'C:/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/Alfvenic_Auroral_Acceleration_AAA'
    # sim_data_output_path = rf'C:/Data/MODELS/alfvenic_auroral_Acceleration_AAA/run_{run_number}'
