import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import  spaceToolsLib as stl
from tqdm import tqdm

# --- Example data (replace with yours) ---

data_dict_potentials = stl.loadDictFromFile('/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/wave_fields/wave_fields_characteristics.cdf')
wFrames = len(data_dict_potentials['time'][0])
# wFrames = 1000
time = data_dict_potentials['time'][0][0:wFrames]
alt = (data_dict_potentials['z'][0]/(stl.Re*stl.m_to_km))

# Data
data1 = data_dict_potentials['E_mu'][0][0:wFrames]
data1[np.isnan(data1)] = 0
data1[np.isinf(np.abs(data1))] = 0

data2 = data_dict_potentials['Phi'][0][0:wFrames]
data2[np.isnan(data2)] = 0
data2[np.isinf(np.abs(data2))] = 0

scaling3 = 1
data3 = data_dict_potentials['Az'][0][0:wFrames]/scaling3
data3[np.isnan(data3)] = 0
data3[np.isinf(np.abs(data3))] = 0


# --- Set up figure ---
fig, ax = plt.subplots(2)
line1, = ax[0].plot([], [], lw=2,color='tab:blue')
line2, = ax[1].plot([], [], lw=2,color='tab:blue',label='$\Phi$')
axA = ax[1].twinx()
line3, = axA.plot([], [], lw=2,color='tab:orange',label='$A_{z}$')

ax[0].set_ylim(data1.min(), data1.max())
ax[0].set_ylabel(r"$E_{\mu}$ [mV/m]")
ax[0].set_xlim(alt.min(), alt.max())

ax[1].set_ylim(data2.min(), data2.max())
ax[1].set_ylabel("$\Phi$ [eV]")
ax[1].set_xlabel("Altitude [$R_{E}$]")
ax[1].set_xlim(alt.min(), alt.max())
axA.set_ylim(data3.min(), -1*data3.min())
axA.set_ylabel("$A_{z}$ " +f"[Wb/m]")
axA.set_xlim(alt.min(), alt.max())
title = ax[0].set_title("")
ax[1].legend([line2,line3],[r'$\Phi$','$A_{z}$'],loc='upper right')


# --- Animation functions ---
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])

    title.set_text("")

    return line1,line2,line3, title

def update(frame):

    profile1 = data1[frame, :]
    profile2 = data2[frame, :]
    profile3 = data3[frame, :]

    line1.set_data(alt, profile1)
    line2.set_data(alt, profile2)
    line3.set_data(alt, profile3)

    title.set_text(f"Time = {time[frame]:.2f}")

    return line1,line2,line3, title



ani = FuncAnimation(fig, update,
                    frames=tqdm([i for i in range(len(time))]),
                    init_func=init,
                    interval=2, blit=True)

# print('Saving Animation')
ani.save("time_altitude_profile.mp4", dpi=200)
fig.tight_layout()

plt.show()