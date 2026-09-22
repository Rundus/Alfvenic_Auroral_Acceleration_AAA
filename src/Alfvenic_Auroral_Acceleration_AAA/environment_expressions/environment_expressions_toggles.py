


class EnvironmentExpressionsToggles:

    def __init__(self):
        self.environment_density_dict ={
                'chaston2006':False,
                'shroeder2021':True,
                'chaston2003_nightside':False,
                'chaston2003_cusp': False
            }

        # FILE I/O
        self.wDenModel_key = [key for key in self.environment_density_dict.keys() if self.environment_density_dict[key]][0]
