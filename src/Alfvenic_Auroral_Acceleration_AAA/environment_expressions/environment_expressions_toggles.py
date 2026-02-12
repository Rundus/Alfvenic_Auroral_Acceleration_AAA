


class EnvironmentExpressionsToggles:

    def __init__(self):
        self.environment_expression_dict ={
                'chaston2006':False,
                'shroeder2021':False,
                'chaston2003_nightside':True,
                'chaston2003_cusp': False
            }

        # FILE I/O
        self.wDenModel_key = [key for key in self.environment_expression_dict.keys() if self.environment_expression_dict[key]][0]
