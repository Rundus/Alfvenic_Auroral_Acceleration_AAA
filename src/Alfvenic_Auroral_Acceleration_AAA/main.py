# --- SubRoutine Options ---
dict_executable = {
    'regen_EVERYTHING': 0,
    'regen_environment_expressions': 1,
    'regen_spatial_grid': 1,
    'regen_plasma_environment': 1,
    'regen_wave_potentials': 1,
    'animate_wave_potentials': 1,
    'regen_liouville_mapping': 1,
    'regen_detector_flux': 1,
    'plot_detector_flux': 1,
    'regen_field_particle_correlation': 1
}

if __name__ == "__main__":
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.execute_subroutines import run_AAA_simulation
    run_AAA_simulation(dict_executable)