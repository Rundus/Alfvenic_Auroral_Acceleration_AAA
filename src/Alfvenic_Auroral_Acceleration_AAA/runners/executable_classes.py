import json
import os
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles

class ExecutableClasses:

    def generate_run_directories(self):

        folders_paths = [
            'spatial_grid',
            'plasma_environment',
            'wave_potentials',
            'liouville_mapping',
            'field_particle_correlation',
            'flux',
            'results'
        ]

        for path in folders_paths:
            target_path = f'{RunToggles.sim_data_output_path}/{path}/'
            if not os.path.exists(target_path):
                os.makedirs(target_path)


    def check_density_model(self):
        # Determine which density model was used to generate the pickle files
        folder_path = f'{RunToggles.sim_data_output_path}'
        model_config_path = f'{folder_path}/run_config.json'

        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_toggles import EnvironmentExpressionsToggles

        with open(model_config_path,'r') as configFile:
            config_dict = json.load(configFile)
            if EnvironmentExpressionsToggles().wDenModel_key != config_dict['expression_generator']['density_model']:
                raise Exception('Pickled model does not match runners configuration. Try re-generating pickle files.')


    def update_run_JSON(self, dict_update):
        import numpy as np
        import tempfile

        def _to_jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f'{type(obj).__name__} is not JSON serializable')

        file_path = f'{RunToggles.sim_data_output_path}/run_config.json'

        with open(file_path, 'r') as f:
            data = json.load(f)
        data.update(dict_update)

        # serialize fully in memory — if this raises, the file on disk is untouched
        text = json.dumps(data, indent=3, default=_to_jsonable)

        folder = os.path.dirname(file_path)
        fd, tmp_path = tempfile.mkstemp(dir=folder, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)   # atomic on Windows and POSIX
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


    def generate_run_JSON(self):

        # Determine the Density model used
        config_dict = {}


        config_dict = {**config_dict,
                       **{
                            # 'Density_Model':f'{EnvironmentExpressionsToggles().wDenModel_key}',
                           # 'Observation': {
                           #      'z_obs': mapping_alt,
                           #      'time_rez': DistributionToggles.time_rez,
                           #      'time_obs_start':DistributionToggles.time_obs_start,
                           #     'time_obs_end': DistributionToggles.time_obs_end,
                           #     'time_rez_waves':DistributionToggles.time_rez_waves,
                           #     'E_max_obs(log)':DistributionToggles.E_max_obs,
                           #     'E_min_obs(log)':DistributionToggles.E_min_obs,
                           #     'N_energy_space_points':DistributionToggles.N_energy_space_points
                           #     # 'Pitch_Range':list(DistributionToggles.pitch_range),
                           #     # 'Energy_Range':list(DistributionToggles.energy_range)
                           #                 },
                           # 'Plasma_Sheet':
                           #     {
                           #         'n_PS':DistributionToggles.n_PS,
                           #         'Te_PS':DistributionToggles.Te_PS,
                           #         'Emax_PS':DistributionToggles.Emax_PS,
                           #         'Emin_PS':DistributionToggles.Emin_PS
                           #     },
                          }
                       }

        # JSON I/O
        folder_path = f'{RunToggles.sim_data_output_path}'
        json_path = f'{folder_path}/run_config.json'

        # check if folder exists, if not create it
        if not os.path.exists(json_path):
            with open(json_path, 'w') as outfile:
                json.dump(config_dict, outfile, indent=3)
