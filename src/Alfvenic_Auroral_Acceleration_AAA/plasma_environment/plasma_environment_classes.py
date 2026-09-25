import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import PlasmaEnvironmentToggles,RunToggles
from glob import glob
import numpy as np

class PlasmaEnvironmentClasses:

    def __init__(self):
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
        envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
        self.B_dipole = envDict['B_dipole']
        data_dict_spatial = stl.loadDictFromFile(glob(rf'{RunToggles.sim_data_output_path}//spatial_grid/*.cdf*')[0])


        # --- Exobase (loss altitude) on this field line ---
        # determine the equatorial loss cone angle
        self.r_lost = 1 + PlasmaEnvironmentToggles.alt_lost / stl.Re
        self.chi_lost = data_dict_spatial['chi'][0][0]
        self.colat_lost = np.arcsin(np.sqrt(self.chi_lost * self.r_lost))
        self.mu_lost = - np.sqrt(np.cos(self.colat_lost)) / self.r_lost
        self.B_lost = self.B_dipole(self.mu_lost, self.chi_lost)

        # --- Normalization of the plasma-sheet Maxwellian ---
        # n0_PS is the density AT THE EQUATOR, where the empty loss cone already removes a sliver:
        # n_eq = n0*cos(alpha_LC,eq). The Maxwellian itself must be normalized to n0.
        self.mu_eq = -1E-4  # very close to zero but not quite to avoid singularities. Represents the geomagnetic equator for perfect dipole
        self.chi_eq = data_dict_spatial['chi'][0][0]
        self.B_eq = self.B_dipole(self.mu_eq, self.chi_eq)
        self.loss_cone_eq = np.arcsin(np.sqrt(self.B_eq / self.B_lost)) #[Radians]
        self.n0_PS_norm = (stl.cm_to_m ** 3) * PlasmaEnvironmentToggles.n0_PS / np.cos(self.loss_cone_eq)  # [m^-3]

    @np.errstate(invalid="ignore") # THIS TURNS OFF WARNINGS FOR THE FOLLOWING FUNCTION
    def loss_cone_angle(self, mu, chi): # Calculates the loss cone angle for the run-specific spatial grid
        # use equatorial loss cone to get loss cone everywhere else. Set any domain error issues == 90
        sin2 = self.B_dipole(mu,chi)/self.B_lost
        return np.arcsin(np.sqrt(np.clip(sin2,0,1))) # [rad]; 90 deg at and below the exobase

    def n_density_PS_loss_cone(self,loss_cone):
        # for a bi-directional loss cone, the zero-th moment gives
        # n(s) = n0 * cos(alpha_LS(s))
        # n0 is the maxwellian normalization value, NOT a plasma density
        # At the geomagnetic equator, n_eq = n0*cos(alpha_LS(s=eq))
        # thus, n(s) = n_eq *cos(alpha_LS(s)) /cos(alpha_LS(s=eq))
        return self.n0_PS_norm*np.cos(loss_cone)

