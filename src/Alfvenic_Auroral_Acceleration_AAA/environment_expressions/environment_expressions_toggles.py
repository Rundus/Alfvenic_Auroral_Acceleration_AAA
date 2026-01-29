


class EnvironmentExpressionsToggles:

    # Density Model
    useChaston2006 = False # SOMETHING DEFINITLY WRONG WITH O2 density
    useShroeder2021 = False # Works fine. Pretty weak Alfven Velocity (~1E7)
    useChaston2003_nightside = True # SOMETHING WRONG WITH ALFVEN VELOCITY? It's nearly light speed!
    useChaston2003_cusp = False # Works fine. Weak Alfven Velocity (1E7), but good balance of inertial effects