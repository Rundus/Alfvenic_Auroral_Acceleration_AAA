from timebudget import timebudget

@timebudget
def wave_fields_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from tqdm import tqdm
    from scipy.integrate import simpson

    # --- Load the wave simulation data ---
    data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/ray_equations/*.cdf*')[0])

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_ray_eqns['time'][0])),deepcopy(data_dict_ray_eqns['time'][1])],
        'mu_w': deepcopy(data_dict_ray_eqns['mu_w']),
        'chi_w': deepcopy(data_dict_ray_eqns['chi_w']),
        'z': deepcopy(data_dict_ray_eqns['z']),
        'E_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [np.array([]), {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'potential_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Perpendicular Potential', 'VAR_TYPE': 'data'}],
        'potential_mu': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'alt_grid', 'UNITS': 'V', 'LABLAXIS': 'Parallel Potential', 'VAR_TYPE': 'data'}],
        'mu_grid': [[],deepcopy(data_dict_ray_eqns['mu_w'][1])],
        'alt_grid': [[], deepcopy(data_dict_ray_eqns['z'][1])],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}]
    }

    ################################################
    # --- EVALUATE FUNCTIONS ON SIMULATION SPACE ---
    ################################################

    # prepare some variables
    times = SimToggles.RK45_tspan[1] - deepcopy(data_dict_output['time'][0])
    Eperp_store = np.zeros(shape=(len(times), len(WaveFieldsToggles.mu_grid)))
    potential_perp_store = np.zeros(shape=(len(times), len(WaveFieldsToggles.mu_grid)))
    Epara_store = np.zeros(shape=(len(times), len(WaveFieldsToggles.mu_grid)))
    Bperp_store = np.zeros(shape=(len(times), len(WaveFieldsToggles.mu_grid)))
    potential_para_store = np.zeros(shape=(len(times), len(WaveFieldsToggles.mu_grid)))

    # perform the loop
    print(f'Number of Iterations: {len(times)}\n')
    for loopidx, time in tqdm(enumerate(times)):
        potential = np.zeros_like(WaveFieldsToggles.mu_grid)
        Ephi = np.zeros_like(WaveFieldsToggles.mu_grid)
        Epara = np.zeros_like(WaveFieldsToggles.mu_grid)
        Bperp = np.zeros_like(WaveFieldsToggles.mu_grid)

        for i in range(len(WaveFieldsToggles.mu_grid)):
            eval_pos = [WaveFieldsToggles.mu_grid[i], RayEquationToggles.chi0_w, RayEquationToggles.phi0_w]
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
        potential_para_store[loopidx] = np.array([-1*simpson(y=Epara[0:i],x=stl.m_to_km*WaveFieldsToggles.alt_grid[0:i]) if i != 0 else 0 for i in range(len(WaveFieldsToggles.mu_grid)) ])

    # store everything
    data_dict_output['E_perp'][0] = Eperp_store
    data_dict_output['E_mu'][0] = Epara_store
    data_dict_output['potential_perp'][0] = potential_perp_store
    data_dict_output['potential_mu'][0] = potential_para_store
    data_dict_output['B_perp'][0] = Bperp_store
    data_dict_output['mu_grid'][0] = WaveFieldsToggles.mu_grid
    data_dict_output['alt_grid'][0] = WaveFieldsToggles.alt_grid

    # Calculate the resonance window
    potential_para_max = np.array([np.max(np.abs(data_dict_output['potential_mu'][0][i])) for i in range(len(data_dict_output['potential_mu'][0]))])
    DAW_vel = data_dict_ray_eqns['omega'][0]/deepcopy(data_dict_ray_eqns['k_mu'][0])
    data_dict_output['resonance_high'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel + np.sqrt(2*stl.q0*potential_para_max/stl.m_e))
    data_dict_output['resonance_low'][0] = 0.5*(stl.m_e/stl.q0)*np.square((DAW_vel - np.sqrt(2 * stl.q0 * potential_para_max / stl.m_e)))
    data_dict_output['DAW_velocity'][0] = 0.5*(stl.m_e/stl.q0)*np.square(DAW_vel)

    ################
    # --- OUTPUT ---
    ################
    outputPath = rf'{WaveFieldsToggles.outputFolder}/wave_fields.cdf'
    stl.outputDataDict(outputPath, data_dict_output)