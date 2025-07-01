
def generate_plasma_environment():

    # --- imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy
    from tqdm import tqdm
    from glob import glob

    # import the toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_toggles import PlasmaToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.spatial_environment.spatial_environment_toggles import SpatialToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.geomagnetic_field.geomagnetic_field_toggles import GeomagneticToggles

    # import the physics models
    from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_classes import ni,ion_composition

    #######################
    # --- LOAD THE DATA ---
    #######################
    # get the geomagnetic field data dict
    data_dict_Bgeo = stl.loadDictFromFile(glob(rf'{GeomagneticToggles.outputFolder}\*.cdf*')[0])
    data_dict_spatial = stl.loadDictFromFile(glob(f'{SpatialToggles.outputFolder}\*.cdf*')[0])

    ############################
    # --- PREPARE THE OUTPUT ---
    ############################

    # Get the spatial dimensions
    chi = deepcopy(data_dict_spatial['chi'][0])
    mu = deepcopy(data_dict_spatial['mu'][0])
    alt = deepcopy(data_dict_spatial['alt'][0])
    lat = deepcopy(data_dict_spatial['lat'][0])
    long = deepcopy(data_dict_spatial['long'][0])

    data_dict_output = {
        'Te': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'K', 'LABLAXIS': 'Te', 'VAR_TYPE':'data'}],
        'Ti': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'K', 'LABLAXIS': 'Ti', 'VAR_TYPE':'data'}],

        'n_total': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'Plasma Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],

        'n_O+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'O+ Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],
        'n_H+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'LABLAXIS': 'H+ Density', 'UNITS': 'm!A-3', 'VAR_TYPE':'data'}],

        'C_O+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'O+ Concentration', 'LABLAXIS': None, 'VAR_TYPE':'data'}],
        'C_H+': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'H+ Concentration', 'LABLAXIS': None, 'VAR_TYPE':'data'}],

        'Omega_O+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'O+ Larmor Freq.', 'VAR_TYPE':'data'}],
        'Omega_H+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'H+ Larmor Freq.', 'VAR_TYPE':'data'}],

        'rho_O+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'm', 'LABLAXIS': 'O+ Larmor Radius', 'VAR_TYPE':'data'}],
        'rho_H+':[np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi', 'DEPEND_0':'mu', 'UNITS': 'm', 'LABLAXIS': 'O+ Larmor Radius', 'VAR_TYPE':'data'}],

        'm_eff_i': [np.zeros(shape=(len(mu),len(chi))), {'DEPEND_1':'chi','DEPEND_0':'mu', 'UNITS': 'kg', 'LABLAXIS': 'm_eff_i', 'VAR_TYPE':'data'}],
        'Omega_eff_i': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'rad/s', 'LABLAXIS': 'Omega_eff_i', 'VAR_TYPE': 'data'}],
        'rho_eff_i': [np.zeros(shape=(len(mu), len(chi))), {'DEPEND_1': 'chi', 'DEPEND_0': 'mu', 'UNITS': 'm', 'LABLAXIS': 'rho_eff_i', 'VAR_TYPE': 'data'}]
    }

    #######################
    # --- DENSITY MODEL ---
    #######################
    model = ni()
    n_O, n_H = model.Chaston2002(simAlt=alt)
    data_dict_output['n_total'][0] = n_O + n_H
    data_dict_output['n_O+'][0] = n_O
    data_dict_output['n_H+'][0] = n_H

    ############################
    # --- ION CONCENTRATIONS ---
    ############################
    model = ion_composition()
    n_O_n_i_ratio = model.Chaston2006(alt)
    data_dict_output['C_O+'][0] = n_O_n_i_ratio
    data_dict_output['C_H+'][0] = 1 - n_O_n_i_ratio


    # ##################################
    # # --- INDIVIDUAL ION DENSITIES ---
    # ##################################
    # if plasmaToggles.useACESII_density_Profile:
    #     data_dict_ACESII_ni_spectrum = stl.loadDictFromFile(rf'{plasmaToggles.outputFolder}\ACESII_ni_spectrum\ACESII_ni_spectrum.cdf')
    #     for idx, key in enumerate(Ikeys):
    #         data_dict_output[f'n_{key}'][0] = 1E6*np.multiply(data_dict_ACESII_ni_spectrum[f'ni_spectrum'][0], data_dict_output[f'C_{key}'][0])
    # else:
    #     for idx, key in enumerate(Ikeys):
    #         data_dict_output[f'n_{key}'][0] = 1E6*deepcopy(data_dict_output[f'n_{key}'][0])
    #
    #
    # ###########################
    # # --- TOTAL ION DENSITY ---
    # ###########################
    # if plasmaToggles.useACESII_density_Profile:
    #     data_dict_output['ni'][0] = 1E6 * deepcopy(data_dict_ACESII_ni_spectrum['ni_spectrum'][0])
    # else:
    #     data_dict_output['ni'][0] = 1E6 * np.array([deepcopy(data_dict_output[f"n_{key}"][0]) for key in plasmaToggles.wIons])
    #
    # ##################
    # # --- ION MASS ---
    # ##################
    # # get the effective mass based on the IRI
    # data_dict_output['m_eff_i'][0] = np.sum(np.array([deepcopy(data_dict_output[f"C_{key}"][0])*Imasses[idx] for idx,key in enumerate(plasmaToggles.wIons)]), axis=0)
    #
    # #################################
    # # --- ELECTRON PLASMA DENSITY ---
    # #################################
    # if plasmaToggles.useACESII_density_Profile:
    #     data_dict_ACESII_ni_spectrum = stl.loadDictFromFile(rf'{plasmaToggles.outputFolder}\ACESII_ni_spectrum\ACESII_ni_spectrum.cdf')
    #     ne_density = 1E6*deepcopy(data_dict_ACESII_ni_spectrum['ni_spectrum'][0])*deepcopy(data_dict_output['ne_ni_ratio'][0])
    # else:
    #     ne_density = 1E6 * deepcopy(data_dict_output['Ne'][0])  # convert data into m^-3
    #
    # data_dict_output['ne'] = data_dict_output.pop('Ne')
    # data_dict_output['ne'] = [ne_density, {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': 'm!A-3!N', 'LABLAXIS': 'ne'}]
    #
    #
    # #####################
    # # --- PLASMA BETA ---
    # #####################
    # plasmaBeta = (2 * stl.u0 *stl.kB)*(data_dict_output['ne'][0] * data_dict_output['Te'][0] ) / np.power(data_dict_Bgeo['Bgeo'][0],2)
    # data_dict_output = {**data_dict_output, **{'beta_e': [plasmaBeta, {'DEPEND_0': 'simLShell','DEPEND_1': 'simAlt', 'UNITS': None, 'LABLAXIS': 'beta_e'}]}}
    #
    # ##########################
    # # --- PLASMA FREQUENCY ---
    # ##########################
    # plasmaDensity = data_dict_output['ne'][0]
    # plasmaFreq = np.array([np.sqrt(plasmaDensity[i] * (stl.q0 * stl.q0) / (stl.ep0 * stl.m_e)) for i in range(len(plasmaDensity))])
    # data_dict_output = {**data_dict_output, **{'plasmaFreq': [plasmaFreq, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'plasmaFreq'}]}}
    #
    #
    # ############################
    # # --- ION CYCLOTRON FREQ ---
    # ############################
    # n_ions = np.array([data_dict_output[f"n_{key}"][0] for key in Ikeys])
    # ionCyclotron_ions = np.array([stl.q0 * data_dict_Bgeo['Bgeo'][0] / mass for mass in Imasses])
    # ionCyclotron_eff = np.sum(ionCyclotron_ions * n_ions, axis=0) / data_dict_output['ni'][0]
    # electronCyclotron = stl.q0 * data_dict_Bgeo['Bgeo'][0] / stl.m_e
    #
    # data_dict_output = {**data_dict_output,
    #                     **{'Omega_e': [electronCyclotron, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'electronCyclotron'}]},
    #                     **{'Omega_i_eff': [ionCyclotron_eff, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': 'ionCyclotron_eff'}]},
    #                     **{f'Omega_{key}': [ionCyclotron_ions[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'rad/s', 'LABLAXIS': f'ionCyclotron_{key}'}] for idx, key in enumerate(Ikeys)}
    #              }
    #
    #
    #
    # ###########################
    # # --- ION LARMOR RADIUS ---
    # ###########################
    # Ti = data_dict_output['Ti'][0]
    # n_ions = np.array([data_dict_output[f"n_{key}"][0] for idx, key in enumerate(Ikeys)])
    # vth_ions = np.array([np.sqrt(2) * np.sqrt(8 * stl.kB * Ti / mass) for mass in Imasses])  # the np.sqrt(2) comes from the vector sum of two dimensions
    # ionLarmorRadius_ions = np.array([vth_ions[idx] / data_dict_output[f"Omega_{key}"][0] for idx, key in enumerate(Ikeys)])
    # ionLarmorRadius_eff = np.sum(n_ions * ionLarmorRadius_ions, axis=0) / data_dict_output['ni'][0]
    # data_dict_output = {**data_dict_output,
    #              **{'ionLarmorRadius_eff': [ionLarmorRadius_eff, {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': 'ionLarmorRadius_eff'}]},
    #              **{f'ionLarmorRadius_{key}': [ionLarmorRadius_ions[idx], {'DEPEND_0': 'simLShell', 'DEPEND_1': 'simAlt', 'UNITS': 'm', 'LABLAXIS': f'ionLarmorRadius_{key}'}] for idx, key in enumerate(Ikeys)}
    #              }



    #####################
    # --- OUTPUT DATA ---
    #####################
    for key in data_dict_spatial.keys():
        data_dict_spatial[key][1]['VAR_TYPE'] = 'support_data'
    data_dict_output = {**data_dict_spatial,**data_dict_output}

    outputPath = rf'{PlasmaToggles.outputFolder}\plasma_environment.cdf'
    stl.outputCDFdata(outputPath, data_dict_output)
