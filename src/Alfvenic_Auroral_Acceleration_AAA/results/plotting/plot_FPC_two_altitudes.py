# --- plot_FP_two_altitudes.py ---
# Description: Plot the FPC data and results of the AAA at two altitudes from
# the data in ...>results


### IMPORTS ###
import matplotlib.pyplot as plt
import numpy as np
import spaceToolsLib as stl
from glob import glob
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_toggles import FPCToggles



#--- get the data ---
alts = [500,3000]
data_dicts_dist = []
data_dicts_FPC = []
data_dicts_flux = []
for i in range(len(alts)):
    data_dicts_FPC.append(stl.loadDictFromFile(glob(f'{SimToggles.sim_data_output_path}/results/{alts[i]}km/field_particle_correlation.cdf')[0]))
    data_dicts_flux.append(stl.loadDictFromFile(f'{SimToggles.sim_data_output_path}/results/{alts[i]}km/flux.cdf'))


# --- plot the results ---
fig, ax = plt.subplots(2,2)
fig.set_figwidth(15)
fig.set_figheight(15)

for i in range(len(alts)):

    # Distribution Function
    ax[i][0].pcolormesh(
        FPCToggles.v_perp_space,
        FPCToggles.v_para_space,
        np.sum(data_dicts_FPC[0]['Distribution_Function'][0],axis=0).T,
        cmap='seismic',
    )
    ax[i][0].set_ylabel(r'$V_{\parallel}$')
    ax[i][0].set_ylabel(r'$V_{\perp}$')

    # FPC
    ax[i][1].pcolormesh(
        FPCToggles.v_perp_space,
        FPCToggles.v_para_space,
        data_dicts_FPC[0]['FPC'][0].T,
        cmap='seismic',
    )
    ax[i][1].set_ylabel(r'$V_{\parallel}$')
    ax[i][1].set_ylabel(r'$V_{\perp}$')

plt.show()