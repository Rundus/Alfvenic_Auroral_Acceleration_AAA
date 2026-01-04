# verify_distributions.py
# Description: Use the distribution functions in distribution_classes to plot them over a range of
# velocities



# imports
import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
import matplotlib.pyplot as plt



# define the velocity grids
N = len(DistributionToggles.vel_space_mu_range)
M = len(DistributionToggles.vel_space_perp_range)
Distribution = np.zeros(shape=(N,M))

for i in range(N):
    for j in range(M):
        Distribution[i][j] = DistributionClasses().Maxwellian(
                                n=DistributionToggles.n_PS,
                               Te=DistributionToggles.Te_PS,
                               vel_perp=DistributionToggles.vel_space_perp_range[i],
                               vel_para=DistributionToggles.vel_space_mu_range[j])




# --- Plot everything ---
fig, ax = plt.subplots()
cmap = ax.pcolormesh(
    DistributionToggles.vel_space_mu_range,
    DistributionToggles.vel_space_perp_range,
    Distribution,
    norm='log',
    cmap='turbo'
)
print(Distribution)
cbar = fig.colorbar(cmap)
plt.show()