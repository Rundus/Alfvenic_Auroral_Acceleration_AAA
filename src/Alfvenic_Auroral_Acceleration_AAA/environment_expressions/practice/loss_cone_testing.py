from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
import numpy as np

# --- get the environment varibles ---
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

# --- define the source region location ---
theta = 89.9 # at the magnetic equator. Note: using exactly 90 runs into numerical issues
r = np.square(np.sin(np.radians(theta)))/RayEquationToggles.chi0_w # distance in kilometers from Earth's center

# find the mu value at this radius
mu_val = round(-1*np.cos(np.radians(theta))/np.square(r),5)

# --- get the magnetic field at the source region ---
B_source = envDict['B_dipole'](mu_val, RayEquationToggles.chi0_w)

# --- Determine where mirroring occurs ---






