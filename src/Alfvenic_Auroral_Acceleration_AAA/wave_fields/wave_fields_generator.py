

def wave_fields_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    import time
    from copy import deepcopy

    # --- File-specific imports ---
    from scipy.integrate import solve_ivp
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_toggles import ScaleLengthToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    import dill