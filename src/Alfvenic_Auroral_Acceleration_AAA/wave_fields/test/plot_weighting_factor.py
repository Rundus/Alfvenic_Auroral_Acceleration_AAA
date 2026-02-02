import matplotlib.pyplot as plt
import numpy as np

from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import *
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import *

weight = np.zeros_like(WaveFieldsToggles.mu_grid)
for idx in range(len(WaveFieldsToggles.mu_grid)):
    eval_pos = [WaveFieldsToggles.mu_grid[idx], RayEquationToggles.chi0_w, RayEquationToggles.phi0_w]
    weight[idx]= WaveFieldsClasses().field_generator(0.1, eval_pos, type='eperp')


fig, ax = plt.subplots()
ax.plot(stl.Re*(SimClasses.r_muChi(WaveFieldsToggles.mu_grid,[RayEquationToggles.chi0_w for i in range(len(WaveFieldsToggles.mu_grid))])-1), weight)
ax.set_xlim(0,20000)
plt.show()