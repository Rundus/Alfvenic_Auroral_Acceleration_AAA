import numpy as np
import spaceToolsLib as stl

class PlasmaEnvironmentToggles:


    # --- Plasma Environment ---

    # Plasma Sheet (Hot)
    Te_PS = 200  # [eV] Temperature of the isotropic Plasma Sheet Distribution
    n0_PS = 1  # [cm^-3] Density of the plasma sheet population at the dipole geomagnetic equator
    Emin_PS = 0 # [eV]
    Emax_PS = 1000  # [eV]

    # Ionosphere/Plasmasphere/Exosphere (Cold)
    Te_cold = 1 # [eV] Temperature of the cold ionospheric plasma up to 20,000 km
    Emin_cold = 0  # [eV]
    Emax_cold = 3  # [eV]

    # --- Loss Cone Information ---
    alt_lost = 550  # [km] altitude which any particles which reach this have distribution=0. The exobase is where particles are essentially collisionless





