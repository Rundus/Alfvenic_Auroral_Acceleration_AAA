import json
import os
import glob
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *

class ExecutableClasses:

    def check_density_model(self):
        # Determine which density model was used to generate the pickle files
        folder_path = f'{SimToggles.sim_root_path}/environment_expressions/pickled_expressions/'
        model_config_path = f'{folder_path}/model_config.json'

        with open(model_config_path,'r') as configFile:
            config_dict = json.load(configFile)
            if EnvironmentExpressionsToggles().wDenModel_key != config_dict['density_model']:
                raise Exception('Pickled model does not match simulation configuration. Try re-generating pickle files.')

    def check_obs_altitude(self):

        # Checks if the output folder of the run exists. If not, create it
        if not os.path.exists(f'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km'):
            os.makedirs(f'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km')

    def generate_run_JSON(self):

        # Determine the Density model used
        config_dict = {}

        config_dict = {**config_dict,
                       **{
                            'Density_Model':f'{EnvironmentExpressionsToggles().wDenModel_key}',
                           'Observation': {
                                'z_obs': DistributionToggles.Observation_altitudes,
                                'time_rez': DistributionToggles.time_rez,
                                'time_obs_start':DistributionToggles.time_obs_start,
                               'time_obs_end': DistributionToggles.time_obs_end,
                               'time_rez_waves':DistributionToggles.time_rez_waves,
                               'E_max(log)':DistributionToggles.E_max,
                               'E_min(log)':DistributionToggles.E_min,
                               'N_energy_space_points':DistributionToggles.N_energy_space_points
                               # 'Pitch_Range':list(DistributionToggles.pitch_range),
                               # 'Energy_Range':list(DistributionToggles.energy_range)
                                           },
                           'Simulation_Extent':{
                               'upper_termination_altitude':DistributionToggles.upper_termination_altitude,
                               'lower_termination_altitude':DistributionToggles.lower_termination_altitude,
                               'RK45_rtol':DistributionToggles.RK45_rtol,
                               'RK45_atol':DistributionToggles.RK45_atol
                           },
                           'Plasma_Sheet':
                               {
                                   'n_PS':DistributionToggles.n_PS,
                                   'Te_PS':DistributionToggles.Te_PS,
                                   'Emax_PS':DistributionToggles.Emax_PS,
                                   'Emin_PS':DistributionToggles.Emin_PS
                               },
                           'Ray_Equations':{
                               'RK45_rtol':RayEquationToggles.RK45_rtol,
                               'RK45_atol': RayEquationToggles.RK45_atol,
                               'lower_boundary':RayEquationToggles.lower_boundary,
                               'upper_boundary':RayEquationToggles.upper_boundary,
                               'f_0':RayEquationToggles.f_0,
                               'Lambda_perp0':RayEquationToggles.Lambda_perp0,
                               'z0_w':RayEquationToggles.z0_w,
                               'Theta0_w':RayEquationToggles.Theta0_w
                           },
                           'Wave_Fields':{
                               'Phi_perp':WaveFieldsToggles.Phi_0
                           }

                          }
                       }

        # JSON I/O
        folder_path = f'{SimToggles.sim_data_output_path}/results/{EnvironmentExpressionsToggles().wDenModel_key}/'
        outpath = f'{folder_path}/run_config.json'

        # check if folder exists, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(outpath, 'w') as outfile:
            json.dump(config_dict, outfile, indent=3)

        return


ExecutableClasses().check_density_model()
