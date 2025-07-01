# --- imports ---
import matplotlib.pyplot as plt
import spaceToolsLib as stl

##################
# --- PLOTTING ---
##################
figure_width = 14  # in inches
figure_height = 8.5  # in inches
Title_fontSize = 25
Label_fontSize = 25
Tick_fontSize = 25
Tick_length = 10
Tick_width = 2
minorTick_fontSize = 20
minorTick_length = 5
minorTick_width = 1
Text_fontsize = 20
Plot_lineWidth = 2.5
Legend_fontSize = 20
dpi = 100


#########################
# --- GET THE TOGGLES ---
#########################
from src.Alfvenic_Auroral_Acceleration_AAA.spatial_environment.spatial_environment_toggles import *
from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_classes import *
data_dict_spatial = stl.loadDictFromFile(f'{SpatialToggles.outputFolder}\\spatial_environment.cdf')

#########################
# --- EVALUATE MODELS ---
#########################

# geocentric distance
r_geocentric = np.linalg.norm(np.array([data_dict_spatial['GSE_X'][0],data_dict_spatial['GSE_Y'][0],data_dict_spatial['GSE_Z'][0]]).T, axis=1)
r_alt = data_dict_spatial['simAlt'][0]/(stl.Re*stl.m_to_km) + 1

# Ti
model = Ti()
Te = model.shroeder2021(simAlt=data_dict_spatial['simAlt'][0])

# ne
model = ne()
ne = model.kletzingTorbert1994(simAlt=data_dict_spatial['simAlt'][0])/np.power(stl.cm_to_m,3) # convert to cm^-3

model = ion_composition()
nOp_to_ni = model.Chaston2006(simAlt=data_dict_spatial['simAlt'][0])

##################
# --- PLOTTING ---
##################

fig, ax = plt.subplots(4)

xData = data_dict_spatial['simAlt'][0]/(stl.Re*stl.m_to_km)

# geocentric distance
ax[0].plot(xData, r_alt, color='black',linestyle='--')
ax[0].plot(xData, r_geocentric, color='blue')
ax[0].set_ylim(0, 7)

# ne
ax[1].plot(xData, ne)
ax[1].set_yscale('log')
ax[1].set_ylim(1E2, 1E5)
ax[1].set_ylabel('$n_{e}$ [cm$^{-3}$]')


# Te
ax[2].plot(xData, Te)
ax[2].set_yscale('log')
ax[2].set_ylim(1E0, 1E4)
ax[2].set_ylabel('$T_{e}$ [eV]')

# mi_to_mp
ax[3].plot(xData, mi_to_mp)
ax[3].set_ylim(0, 15)
ax[3].set_ylabel('$m_{i}/m_{p}$')
ax[3].set_xlabel('s($R_{E}$)')

for i in range(4):
    ax[i].set_xlim(0, 6)

plt.tight_layout()

plt.savefig(r'C:\Users\cfelt\PycharmProjects\Alfvenic_Auroral_Acceleration_AAA\src\Alfvenic_Auroral_Acceleration_AAA\plasma_environment\test_scripts\plasma_environment.png')

