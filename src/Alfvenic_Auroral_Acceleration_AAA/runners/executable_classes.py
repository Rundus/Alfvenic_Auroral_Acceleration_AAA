import json
import os
import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.runners.FileIO import FileIOClass

class ExecutableClasses:

    def determine_configuration_key(self,dict_run_settings):

        for key, val in dict_run_settings.items():
            if val:
                FileIOClass.set_run(key) # for the ENTIRE simulation run, set the model key e.g. custom_1, kletzingHu2001 etc
                return key

        raise Exception('No run configuration was chosen. Go to main.py and set "True" for a case in dict_run_settings.')

    def generate_run_directories(self):

        folders_paths = [
            'spatial_grid',
            'plasma_environment',
            'plasma_environment/pickled_expressions',
            'wave_potentials',
            'liouville_mapping',
            'field_particle_correlation',
            'detector_flux',
        ]

        for path in folders_paths:
            target_path = f'{FileIOClass.path_sim_output}/{path}/'
            if not os.path.exists(target_path):
                os.makedirs(target_path)

    def print_run_toggles(self):

        run_module = self.load_configuration_module()
        config_dict = self.load_configuration_dict(run_module)

        for name, cls in config_dict.items():
            data = cls
            width = max(map(len, data), default=0)
            print(f"── {name} " + "─" * max(0, 60 - len(name)))
            for key, val in data.items():
                if isinstance(val, np.ndarray) and val.size > 8:
                    val = f"{val.size} pts, {val.min():.6g} to {val.max():.6g}"
                elif isinstance(val, dict) and all(isinstance(v, (bool, int)) and v in (0, 1) for v in val.values()):
                    val = ", ".join(k for k, v in val.items() if v)
                elif isinstance(val, float):
                    val = f"{val:.6g}"
                print(f"  {key:<{width}} = {val}")

    def load_configuration_module(self):

        import importlib
        path_run_module = f'src.Alfvenic_Auroral_Acceleration_AAA.runners.'

        if FileIOClass.key_run_config == 'custom_0':
            path_run_module += 'custom_runs.custom_run_0'

        elif FileIOClass.key_run_config == 'custom_1':
            path_run_module += 'custom_runs.custom_run_1'

        elif FileIOClass.key_run_config == 'kletzing&Hu2001':
            path_run_module += 'predefined_runs.kletzing&Hu2001'

        elif FileIOClass.key_run_config == 'chaston2003_nightside':
            path_run_module += 'predefined_runs.chaston2003_nightside'

        elif FileIOClass.key_run_config == 'chaston2003_sunlit_boundary':
            path_run_module += 'predefined_runs.chaston2003_sunlit_boundary'

        elif FileIOClass.key_run_config == 'tanaka2005':
            path_run_module += 'predefined_runs.Tanaka2005'

        run_module = importlib.import_module(path_run_module)

        return run_module

    def load_configuration_dict(self,run_module):
        '''
            Returns the values in the configuration file as a dictionary
        '''
        # collect all the variables held within the classes of the configuration file
        run_config_dict = {}
        for class_name, cls in vars(run_module).items():
            # keep only classes defined in this run file (skips stl, np, os and any imported classes)
            if not isinstance(cls, type) or cls.__module__ != run_module.__name__:
                continue
            # keep only the values: skip __dunder__ entries and any methods
            run_config_dict[class_name] = {
                name: val for name, val in vars(cls).items()
                if not name.startswith('_')
                   and not callable(val)
                   and not isinstance(val, (staticmethod, classmethod, property))
            }
        return run_config_dict

    def generate_run_JSON(self):

        # Load the configuration module
        run_module = self.load_configuration_module()
        run_config_dict = self.load_configuration_dict(run_module)

        # Write out the configuration dict to a JSON file
        def to_json_safe(obj):  # json.dump calls this for anything it can't write itself
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # pitch_range_obs, energy_range_obs
            if isinstance(obj, np.generic):
                return obj.item()  # np.float64, np.bool_, etc.
            return str(obj)

        with open(f'{FileIOClass.path_sim_output}/run_config.json', 'w') as outfile:
            json.dump(run_config_dict, outfile, indent=3, default=to_json_safe)

