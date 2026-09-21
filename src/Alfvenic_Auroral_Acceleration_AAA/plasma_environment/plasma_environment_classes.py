import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_toggles import PlasmaEnvironmentToggles
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
from glob import glob
import numpy as np

class PlasmaEnvironmentClasses:

    def __init__(self):
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
        envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
        self.B_dipole = envDict['B_dipole']
        data_dict_spatial = stl.loadDictFromFile(glob(rf'{RunToggles.sim_data_output_path}//spatial_grid/*.cdf*')[0])

        # determine the equatorial loss cone angle
        self.r_lost = 1 + PlasmaEnvironmentToggles.alt_lost / stl.Re
        self.chi_lost = data_dict_spatial['chi'][0][0]
        self.colat_lost = np.arcsin(np.sqrt(self.chi_lost * self.r_lost))
        self.mu_lost = - np.sqrt(np.cos(self.colat_lost)) / self.r_lost
        self.B_lost = self.B_dipole(self.mu_lost, self.chi_lost)
        self.mu_eq = -1E-4  # very close to zero but not quite to avoid singularities. Represents the geomagnetic equator for perfect dipole
        self.chi_eq = data_dict_spatial['chi'][0][0]
        self.B_eq = self.B_dipole(self.mu_eq, self.chi_eq)
        self.loss_cone_eq = np.arcsin(np.sqrt(self.B_eq / self.B_lost))

    @np.errstate(invalid="ignore") # THIS TURNS OFF WARNINGS FOR THE FOLLOWING FUNCTION
    def loss_cone_angle(self, mu, chi): # Calculates the loss cone angle for the run-specific spatial grid
        # use equatorial loss cone to get loss cone everywhere else. Set any domain error issues == 90
        loss_cone = np.arcsin(np.sin(self.loss_cone_eq) * np.sqrt(self.B_dipole(mu,chi) / self.B_eq))
        return loss_cone

    def n_density_PS_loss_cone(self,loss_cone):
        return (stl.cm_to_m ** 3) * PlasmaEnvironmentToggles.n0_PS * np.cos(loss_cone)

