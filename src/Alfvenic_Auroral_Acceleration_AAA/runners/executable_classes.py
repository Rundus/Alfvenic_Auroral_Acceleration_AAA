import json
import os
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles,EnvironmentExpressionsToggles
import inspect

class ExecutableClasses:

    def generate_run_directories(self):

        folders_paths = [
            'spatial_grid',
            'plasma_environment',
            'wave_potentials',
            'liouville_mapping',
            'field_particle_correlation',
            'detector_flux',
            'field_particle_correlation',
            'pre_defined_runs'
        ]

        for path in folders_paths:
            target_path = f'{RunToggles.sim_data_output_path}/{path}/'
            if not os.path.exists(target_path):
                os.makedirs(target_path)


    def check_density_model(self):
        # Determine which density model was used to generate the pickle files
        folder_path = f'{RunToggles.sim_data_output_path}'
        model_config_path = f'{folder_path}/run_config.json'

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

    def update_toggle(self,module,class_name,attr,value):
        classes = {name: cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__}
        cls = classes[class_name]
        setattr(cls, attr, value)

    def load_class_toggles(self,module):
        return {name: cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__}

    def load_class_toggle_data(self, cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith('__') and not callable(v)}

    def update_class_toggles(self, module, module_overrides):

        # load the current class info
        dict_overrides = self.load_class_toggles(module_overrides)
        dict_current = self.load_class_toggles(module)

        # load the overrides data
        override_settings = {cls_name: self.load_class_toggle_data(val) for cls_name, val in dict_overrides.items()}

        for class_name, cls_obj in dict_current.items():

            if class_name !='RunToggles': # DONT update the FileIO toggles

                for attr, val in override_settings[class_name].items():
                    self.update_toggle(module,class_name,attr,val)

    def load_preset_run(self):
        from src.Alfvenic_Auroral_Acceleration_AAA import run_toggles

        # find which pre-defined classes to load and update the toggles
        if not RunToggles.dict_run_settings['custom']:

             # Load the toggles if predefined
            if RunToggles.dict_run_settings['Kletzing&Hu_2001']:
                from src.Alfvenic_Auroral_Acceleration_AAA.runners.predefined_runs import kletzingHu2001
                module_predefined = kletzingHu2001

             # update the toggles for the run
            self.update_class_toggles(run_toggles, module_predefined)

    # def print_run_toggles(self):
    #     from src.Alfvenic_Auroral_Acceleration_AAA import run_toggles
    #     dict = self.load_class_toggles(run_toggles)
    #     settings = {cls_name: self.load_class_toggle_data(val) for cls_name, val in dict.items()}
    #     for key, val in settings.items():
    #         print(key)
    #         print(val)

    def print_run_toggles(self):
        import numpy as np
        from src.Alfvenic_Auroral_Acceleration_AAA import run_toggles

        for name, cls in self.load_class_toggles(run_toggles).items():
            data = self.load_class_toggle_data(cls)
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

    def generate_run_JSON(self):

        # Determine the Density model used
        config_dict = {}

        # JSON I/O
        folder_path = f'{RunToggles.sim_data_output_path}'
        json_path = f'{folder_path}/run_config.json'

        # check if folder exists, if not create it
        if not os.path.exists(json_path):
            with open(json_path, 'w') as outfile:
                json.dump(config_dict, outfile, indent=3)
