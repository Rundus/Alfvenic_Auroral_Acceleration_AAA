def generate_GeomagneticField():
    # --- common imports ---
    import spaceToolsLib as stl
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.spatial_environment_grid.spatial_environment_toggles import SpatialToggles
    from copy import deepcopy

    # --- file-specific imports ---
    from src.Alfvenic_Auroral_Acceleration_AAA.archive.geomagnetic_field_grid.geomagnetic_field_toggles import GeomagneticToggles

    # prepare the output
    data_dict_output = {}

    #######################
    # --- LOAD THE DATA ---
    #######################
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SpatialToggles.outputFolder}\*.cdf*')[0])

    ########################################
    # --- GENERATE THE B-FIELD & TOGGLES ---
    ########################################
    # description: For a range of LShells and altitudes get the Latitude of each point and define a longitude.
    # output everything as a 2D grid of spatial

    # prepare the data
    chi = deepcopy(data_dict_spatial['chi'][0])
    mu = deepcopy(data_dict_spatial['mu'][0])
    alt = deepcopy(data_dict_spatial['alt'][0])
    lat = deepcopy(data_dict_spatial['lat'][0])
    long = deepcopy(data_dict_spatial['long'][0])
    grid_Bgeo = np.zeros(shape=(len(mu),len(chi)))
    grid_Bgrad = np.zeros(shape=(len(mu), len(chi)))

    for idx in tqdm(range(len(chi))):

        lats = lat[:,idx]
        alts = alt[:,idx]/stl.m_to_km # Convert to kilometers from earth's surface
        longs = long[:,idx]

        # Get the Chaos model
        B = stl.CHAOS(lats, longs, alts, [SpatialToggles.target_time for i in range(len(alts))])
        Bgeo = (1E-9) * np.array([np.linalg.norm(Bvec) for Bvec in B])
        Bgrad = np.gradient(Bgeo,alts)

        # store the data
        grid_Bgeo[:, idx] = Bgeo
        grid_Bgrad[:, idx] = Bgrad


    # --- Construct the Data Dict ---
    for key in data_dict_spatial.keys():
        data_dict_spatial[key][1]['VAR_TYPE'] = 'support_data'
    data_dict_output = { **data_dict_spatial,
                         **{'Bgeo': [grid_Bgeo, {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS':'T', 'LABLAXIS': 'Bgeo', 'VAR_TYPE': 'data'}],
                            'Bgrad': [grid_Bgrad, {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS':'T/m', 'LABLAXIS': 'Bgrad', 'VAR_TYPE': 'data'}],
                        }}

    outputPath = rf'{GeomagneticToggles.outputFolder}\geomagneticfield.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
