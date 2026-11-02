# --- Set the File I/O for the simulation ---

# Laptop
path_code_src = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
path_sim_output = rf'C:/data/alfvenic_auroral_acceleration_AAA/'

# Linux PC
# path_code_src = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
# path_sim_output = rf'/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/'

# Home PC
# path_code_src = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
# path_sim_output = rf'C:/data/alfvenic_auroral_acceleration_AAA/'


# --- Choose the run configuration ---
dict_run_settings = {
    'custom_0':False,
    'custom_1':True,
    'kletzing&Hu2001': False,
    'chaston2003_nightside':False,
    'chaston2003_sunlit_boundary':False,
    'tanaka2005':False
}

# --- subroutines to execute for the chosen run ---
dict_subroutines = {
    'regen_EVERYTHING': 0,
    'regen_environment_expressions': 0,
    'regen_spatial_grid': 1,
    'regen_plasma_environment': 0,
    'regen_wave_potentials': 0,
    'animate_wave_potentials': 0,
    'regen_liouville_mapping': 0,
    'regen_detector_flux': 0,
    'plot_detector_flux': 0,
    'regen_field_particle_correlation': 0,
}

# --- Run the Simulation ---
if __name__ == "__main__":
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.execute_subroutines import run_AAA
    run_AAA(path_code_src, path_sim_output, dict_run_settings, dict_subroutines)