from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

# At equator
mu = -0.001
chi = 0.11406

print(envDict['B_dipole'](mu,chi))