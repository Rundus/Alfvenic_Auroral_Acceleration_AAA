# --- Set the File I/O for the simulation ---

# --- Choose the run configuration ---
dict_run_settings = {
    'custom_0':False,
    'custom_1':False,
    'kletzing&Hu2001': True,
    'chaston2003_nightside':False,
    'chaston2003_sunlit_boundary':False,
    'tanaka2005':False
}

# --- subroutines to execute for the chosen run ---
dict_subroutines = {
    'regen_EVERYTHING': 0,
    'regen_environment_expressions': 0,
    'regen_spatial_grid': 0,
    'regen_plasma_environment': 0,
    'regen_wave_potentials': 1,
    'animate_wave_potentials': 1,
    'regen_liouville_mapping': 0,
    'regen_detector_flux': 0,
    'plot_detector_flux': 0,
    'regen_field_particle_correlation': 0,
}

# --- Run the Simulation ---
if __name__ == "__main__":
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.execute_subroutines import run_AAA
    run_AAA(dict_run_settings, dict_subroutines)