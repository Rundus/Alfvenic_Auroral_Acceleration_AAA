import spaceToolsLib as stl

class SpatialGridToggles:

    ##################################
    # --- SPATIAL ENVIRONMENT GRID ---
    ##################################
    # DEFINE SIMULATION EXTENT in terms of geophysical parameters
    L_Shell = 8.5
    z_para_min = 100 # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude
    z_para_max = 30000 # [km] distance along a geomagnetic field starting from Earth's surface, NOT altitude

    ######################
    # --- MU-Dimension ---
    ######################
    N_mu = 20000  # number of points in mu direction


