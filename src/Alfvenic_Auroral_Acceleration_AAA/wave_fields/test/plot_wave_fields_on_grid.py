# Script to test out how the wave fields equations generate the wave

# Imports
import matplotlib.pyplot as plt
import numpy as np
import spaceToolsLib as stl
from tqdm import tqdm
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses

###########################
# --- [1] Define a grid ---
###########################

# --- Chi-Dimension ---
N_chi = 20  # number of points in chi direction
chi_low, chi_high = [0.108446,0.108950]
chi_range = np.linspace(chi_low, chi_high, N_chi)

# --- MU-Dimension ---
# determine minimum/maximum mu value for the TOP colattitude
N_mu = 30  # number of points in mu direction
mu_min, mu_max = [-0.9,-0.4]
mu_range = np.linspace(mu_min, mu_max, N_mu)

# --- Phi-Dimension ---
N_phi = 40
phi_min,phi_max = np.radians(-10),np.radians(10)
phi_range = np.linspace(phi_min,phi_max,N_phi)

# CONSTRUCT THE GRID
chi_grid, mu_grid, phi_grid = np.meshgrid(chi_range,mu_range,phi_range)

#####################################
# --- [2] EVALUATE FIELDS ON GRID ---
#####################################
time = 0.4 # in simulation seconds
potential = np.zeros_like(phi_grid)

for i in tqdm(range(len(mu_range))):
    for j in range(len(chi_range)):
        for k in range(len(phi_range)):
            potential[i][j][k] = WaveFieldsClasses().Potential_phi(time,mu_grid[i,j,k],chi_grid[i,j,k],phi_grid[i,j,k])

# print(np.where(potential!=0))