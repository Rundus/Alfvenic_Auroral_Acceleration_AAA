class WaveFieldsToggles:

    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = -1*5000 # In Volts. maximum potential in the perpendicular direction

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/wave_fields'

    # --- Inverted-V potential ---
    inV_Volts = 5000 # in Volts
    inV_Zmin = 5000  # in km
    inV_Zmax = 10000 # in km