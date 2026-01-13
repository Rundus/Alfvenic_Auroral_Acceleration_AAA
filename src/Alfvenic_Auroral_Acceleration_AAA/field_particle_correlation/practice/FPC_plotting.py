from copy import deepcopy
import numpy as np
import spaceToolsLib as stl
import matplotlib.pyplot as plt
from scipy.signal import correlate
from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_toggles import FPCToggles

# --- load the FPC data ---
data_dict = stl.loadDictFromFile('/home/connor/Data/physicsModels/alfvenic_auroral_acceleration_AAA/field_particle_correlation/field_particle_correlation.cdf')

# define some variables
time = deepcopy(data_dict['time'][0])
Dist = deepcopy(data_dict['Distribution_Function'][0])
E_mu = deepcopy(data_dict['E_mu_corr'][0])
df_dvpara = deepcopy(data_dict['df_dvpara'][0])

# choose where to look at the correlation
E_parallel = 1 # in eV
E_perp = 1 # in eV

v_para_idx = np.abs(FPCToggles.v_para_space - np.sqrt(2*stl.q0*E_parallel/stl.m_e)).argmin()
v_perp_idx = np.abs(FPCToggles.v_perp_space - np.sqrt(2*stl.q0*E_perp/stl.m_e)).argmin()

print(v_perp_idx,v_para_idx)

# calculate the correlation
dist_corr = Dist[:,v_perp_idx, v_para_idx]
df_dvpara_corr = df_dvpara[:,v_perp_idx,v_para_idx]
corr = correlate(df_dvpara_corr, -1*E_mu, mode='same')
print(df_dvpara_corr,-1*E_mu,np.sum(corr))

# plot things
fig, ax = plt.subplots(3)
fig.set_figwidth(10)
fig.set_figheight(10)
fig.suptitle('$v_{\perp}$ =' + f'{0.5*(stl.m_e/stl.q0)*np.square(FPCToggles.v_perp_space[v_perp_idx])} eV\n' +
             '$v_{\parallel}$ =' + f'{0.5*(stl.m_e/stl.q0)*np.square(FPCToggles.v_para_space[v_para_idx])} eV')
ax[0].plot(time, -1*E_mu)
ax[0].set_ylabel('$E_{\parallel}$')
ax[0].set_xlabel('Time [s]')
ax[1].plot(time, df_dvpara_corr)
ax[1].set_ylabel(r'$(ev_{\parallel}^{2}/2) \partial f_{e}/ \partial v_{\parallel}$')
ax[1].set_xlabel('Time [s]')
ax[2].plot(corr)
ax[2].set_ylabel(r'$Corr<(ev_{\parallel}^{2}/2) \partial f_{e}/\partial v_{\parallel}, E_{\parallel}>$')
plt.show()
