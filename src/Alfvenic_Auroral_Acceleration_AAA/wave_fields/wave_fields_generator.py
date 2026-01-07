import scipy.integrate


def wave_fields_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles as toggles
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses2D as WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from tqdm import tqdm
    from scipy.integrate import simpson

    # --- Load the wave simulation data ---
    data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])

    # prepare the output
    data_dict_output = {
        'time': deepcopy(data_dict_wavescale['time']),
        'mu_w': deepcopy(data_dict_wavescale['mu_w']),
        'chi_w': deepcopy(data_dict_wavescale['chi_w']),
        'z': deepcopy(data_dict_wavescale['z']),
        'E_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [np.array([]), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'potential_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Perpendicular Potential', 'VAR_TYPE': 'data'}],
        'potential_mu': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Parallel Potential', 'VAR_TYPE': 'data'}],
        'mu_grid': [[],deepcopy(data_dict_wavescale['mu_w'][1])],
        'alt_grid': [[], deepcopy(data_dict_wavescale['z'][1])],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}]
    }

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################

    # create the grid on which to plot everything
    # --- MU-Dimension ---
    # determine minimum/maximum mu value for the TOP colattitude
    N_mu = 500  # number of points in mu direction
    mu_min, mu_max = [-1, -0.1]
    mu_grid = np.linspace(mu_min, mu_max, N_mu)
    alt_grid = np.array(stl.Re*(ScaleLengthClasses.r_muChi(mu_grid,[SimToggles.chi0 for i in range(len(mu_grid))])-1))

    # prepare some variables
    times = deepcopy(data_dict_wavescale['time'][0])
    Eperp_store = np.zeros(shape=(len(times), len(mu_grid)))
    potential_perp_store = np.zeros(shape=(len(times), len(mu_grid)))
    Epara_store = np.zeros(shape=(len(times), len(mu_grid)))
    Bperp_store = np.zeros(shape=(len(times), len(mu_grid)))
    potential_para_store = np.zeros(shape=(len(times), len(mu_grid)))

    # perform the loop
    print(f'Number of Iterations: {len(times)}\n')
    for loopidx, time in tqdm(enumerate(times)):
        potential = np.zeros_like(mu_grid)
        Ephi = np.zeros_like(mu_grid)
        Epara = np.zeros_like(mu_grid)
        Bperp = np.zeros_like(mu_grid)

        for i in range(len(mu_grid)):
            eval_pos = [mu_grid[i], SimToggles.chi0, SimToggles.phi0]
            potential[i] = WaveFieldsClasses().field_generator(time, eval_pos, type='potential')
            Ephi[i] = WaveFieldsClasses().field_generator(time, eval_pos, type='eperp')
            Epara[i] = WaveFieldsClasses().field_generator(time, eval_pos, type='epara')
            Bperp[i] = WaveFieldsClasses().field_generator(time, eval_pos, type='bperp')

        # Store the outputs
        Eperp_store[loopidx] = Ephi
        Epara_store[loopidx] = Epara
        potential_perp_store[loopidx] = potential
        Bperp_store[loopidx] = Bperp

        # calculate the parallel potential via line integration
        potential_para_store[loopidx] = np.array([-1*simpson(y=Epara[0:i],x=stl.m_to_km*alt_grid[0:i]) if i != 0 else 0 for i in range(len(mu_grid)) ])

    # store everything
    data_dict_output['E_perp'][0] = Eperp_store
    data_dict_output['E_mu'][0] = Epara_store
    data_dict_output['potential_perp'][0] = potential_perp_store
    data_dict_output['potential_mu'][0] = potential_para_store
    data_dict_output['B_perp'][0] = Bperp_store
    data_dict_output['mu_grid'][0] = mu_grid
    data_dict_output['alt_grid'][0] = alt_grid


    # Calculate the resonance window
    potential_para_max = np.array([np.max(np.abs(data_dict_output['potential_mu'][0][i])) for i in range(len(data_dict_output['potential_mu'][0]))])
    DAW_vel = data_dict_wavescale['omega'][0]/deepcopy(data_dict_wavescale['k_mu'][0])
    data_dict_output['resonance_high'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel + np.sqrt(2*stl.q0*potential_para_max/stl.m_e))
    data_dict_output['resonance_low'][0] = 0.5*(stl.m_e/stl.q0)*np.square((DAW_vel - np.sqrt(2 * stl.q0 * potential_para_max / stl.m_e)))
    data_dict_output['DAW_velocity'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel)

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{toggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)