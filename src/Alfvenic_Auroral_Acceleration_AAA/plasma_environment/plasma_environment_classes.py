from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

class PlasmaEnvironmentClasses:

    def loss_cone_angle(self, mu, chi, mu_eq, chi_eq):
        B_eq = envDict['B_dipole'](mu_eq,chi_eq)
        B_eval = envDict['B_dipole'](mu,chi)

