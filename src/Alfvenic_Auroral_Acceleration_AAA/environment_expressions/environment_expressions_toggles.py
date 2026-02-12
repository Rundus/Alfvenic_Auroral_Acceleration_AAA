


class EnvironmentExpressionsToggles:

    # DENSITY MODELS

    # [1]
    useChaston2006 = False # SOMETHING DEFINITLY WRONG WITH O2 density
    chaston2006_model_path = 'chaston2006'

    # [2]
    useShroeder2021 = False # Works fine. Alfven Velocity (~2E8)
    shroeder2021_model_path = 'shroeder2021'

    # [3]
    useChaston2003_nightside = True # VA is nearly light speed, but omega/k_parallel is ~2E7
    chaston2003_model_path = 'chaston2003_nightside'

    # [4]
    useChaston2003_cusp = False # Works fine. Weak Alfven Velocity (1E7), but good balance of inertial effects
    chaston2003_model_path = 'chaston2003_cusp'