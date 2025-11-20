class WaveFieldsToggles:


    # Initial Electric Wave Field Strength - At the initial position
    E_perp_0 = 5/1000 # in V/m

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/wave_fields'