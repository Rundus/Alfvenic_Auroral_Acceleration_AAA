import os


class FileIOClass:

    # Laptop
    path_code_src = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
    path_sim_data = rf'C:/data/alfvenic_auroral_acceleration_AAA/'

    # Linux PC
    # path_code_src = r'/home/connor/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
    # path_sim_data = rf'/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/'

    # Home PC
    # path_code_src = r'C:/Users/conno/PycharmProjects/Alfvenic_Auroral_Acceleration_AAA/src/'
    # path_sim_data = rf'C:/data/alfvenic_auroral_acceleration_AAA/'

    #############################
    # DON'T touch the stuff below
    #############################
    key_run_config = os.environ.get('AAA_RUN_KEY')
    path_sim_output = os.environ.get('PATH_SIM_OUTPUT')

    @classmethod
    def set_run(cls, key):
        cls.key_run_config = key  # this process
        cls.path_sim_output = f'{cls.path_sim_data}/run_{key}/'
        os.environ['AAA_RUN_KEY'] = key
        os.environ['PATH_SIM_OUTPUT'] = f'{cls.path_sim_data}/run_{key}/'