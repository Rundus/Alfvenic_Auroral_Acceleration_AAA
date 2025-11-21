class WaveFieldsToggles:


    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = 1000 # In Volts. maximum potential in the perpendicular direction

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/wave_fields'