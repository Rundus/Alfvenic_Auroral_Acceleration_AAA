# --- executable.py ---
# --- Author: C. Feltman ---
# DESCRIPTION: regenerate the IONOSPHERE plasma environment toggles


#################
# --- IMPORTS ---
#################
import time
import spaceToolsLib as stl
import warnings
from src.Alfvenic_Auroral_Acceleration_AAA.executable_toggles import dict_executable
warnings.filterwarnings("ignore")
start_time = time.time()

################################
# --- --- --- --- --- --- --- --
# --- ENVIRONMENT GENERATORS ---
# --- --- --- --- --- --- --- --
################################
# re-run everything
if dict_executable['regen_EVERYTHING']==1:
    for key in dict_executable.keys():
        dict_executable[key] = 1





if dict_executable['regen_spatial_environment']==1:
    # spatial environment
    stl.prgMsg('Regenerating Spatial Environment')
    from src.Alfvenic_Auroral_Acceleration_AAA.spatial_environment.spatial_environment_generator import generate_spatial_environment
    generate_spatial_environment()
    stl.Done(start_time)





if dict_executable['regenBgeo']==1:
    # Geomagnetic environment
    stl.prgMsg('Regenerating Geomagnetic Field\n')
    from src.Alfvenic_Auroral_Acceleration_AAA.geomagnetic_field.geomagnetic_field_generator import generate_GeomagneticField
    generate_GeomagneticField()
    stl.Done(start_time)





if dict_executable['regen_background_plasma_environment']==1:
    # plasma environment
    stl.prgMsg('Regenerating Plasma Environment')
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_generator import generate_plasma_environment
    generate_plasma_environment()
    stl.Done(start_time)




