class ResultsToggles:

    # --- File I/O ---
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_toggles import EnvironmentExpressionsToggles
    outputFolder = f'{SimToggles.sim_data_output_path}/results/{EnvironmentExpressionsToggles().wDenModel_key}/'