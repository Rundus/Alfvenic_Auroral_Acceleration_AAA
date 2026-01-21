# --- plot_FP_two_altitudes.py ---
# Description: Plot the FPC data and results of the AAA at two altitudes from
# the data in ...>results


### IMPORTS ###
import matplotlib.pyplot as plt
import numpy as np
import spaceToolsLib as stl
from glob import glob

from sympy.printing.pretty.pretty_symbology import line_width

from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *
from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_toggles import FPCToggles
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles


#--- get the data ---
alts = [500, 3000] # Pick 2 altitudes

data_dicts_dist = []
data_dicts_FPC = []
data_dicts_flux = []
data_dicts_waves = []
for i in range(len(alts)):
    data_dicts_FPC.append(stl.loadDictFromFile(glob(f'{SimToggles.sim_data_output_path}/results/{alts[i]}km/FPC_{alts[i]}km.cdf')[0]))
    data_dicts_flux.append(stl.loadDictFromFile(f'{SimToggles.sim_data_output_path}/results/{alts[i]}km/flux_{alts[i]}km.cdf'))
    data_dicts_waves.append(stl.loadDictFromFile(f'{SimToggles.sim_data_output_path}/results/{alts[i]}km/wave_fields_{alts[i]}km.cdf'))

# get the normalization
vth = np.sqrt(2*stl.q0*DistributionToggles.Te_PS/stl.m_e)

# --- plot the results ---
plt.style.use(rf'{SimToggles.sim_root_path}/results/plotting/plot_FPC_mpl_style_sheet.mplstyle')
fig, ax = plt.subplots(2,2)

for i in range(len(alts)):

    # get the distribution data only in the window of the wave
    grad = np.gradient(data_dicts_FPC[i]['E_mu_corr'][0])
    finder = np.where(np.abs(grad) > 0)[0]
    low_idx = finder[0]
    high_idx = finder[-1]

    # Find the DAW resonance bands at the relevant altitudes
    alt_idx = np.abs(data_dicts_waves[i]['z'][0] - alts[i]).argmin()
    DAW_res_low, DAW_res_high, DAW = data_dicts_waves[i]['resonance_low'][0][alt_idx], data_dicts_waves[i]['resonance_high'][0][alt_idx], data_dicts_waves[i]['DAW_velocity'][0][alt_idx]

    # Distribution Function
    ax[0][i].set_title(r'$f_{avg,e}(v_{\parallel}, v_{\perp})' +f'@ ${alts[i]} km')
    ax[0][i].set_xlim(-3.5,3.5)
    ax[0][i].set_ylim(0,5)
    cmap = ax[0][i].pcolormesh(
        FPCToggles.v_para_space / vth,
        FPCToggles.v_perp_space/vth,
        np.mean(data_dicts_FPC[i]['Distribution_Function'][0][low_idx:high_idx+1,:,:],axis=0),
        cmap=stl.apl_rainbow_black0_cmap(),
        norm='log',
        vmin=1E-18,
        vmax=1E-13
    )
    cb = plt.colorbar(cmap)

    # FPC
    ax[1][i].set_title('$C_{E_{\parallel}}(v_{\perp}, v_{\parallel})' + f'@ ${alts[i]} km')
    cmap = ax[1][i].pcolormesh(
        FPCToggles.v_para_space / vth,
        FPCToggles.v_perp_space/vth,
        data_dicts_FPC[i]['FPC'][0],
        cmap='bwr',
        vmin=-2E-33,
        vmax=2E-33
    )
    ax[1][i].set_xlim(-3.5, 3.5)
    ax[1][i].set_ylim(0, 5)

    if i ==0:
        ax[0][i].set_ylabel(r'$V_{\perp}/v_{th,PS}$')
        ax[1][i].set_ylabel(r'$V_{\perp}/v_{th,PS}$')

    ax[1][i].set_xlabel(r'$V_{\parallel}/v_{th,PS}$')
    cb = plt.colorbar(cmap)

    # plot the resonance bands
    for k in range(2):
        ax[k][i].axvline(x=np.sqrt(2 * stl.q0 * DAW_res_low / stl.m_e)/vth, linestyle='--',alpha=0.35)
        ax[k][i].axvline(x=np.sqrt(2 * stl.q0 * DAW_res_high / stl.m_e)/vth, linestyle='--',alpha=0.35)
        ax[k][i].axvline(x=np.sqrt(2 * stl.q0 * DAW / stl.m_e)/vth, linestyle='-',alpha=0.35)

plt.tight_layout()
fig.savefig(f'{ResultsToggles.outputFolder}/plotting/plots/FPC_twoAlts.png')