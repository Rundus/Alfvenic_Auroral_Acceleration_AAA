"""
alfven_min.py -- minimal version of the solver.

Same scheme and same answers as alfven_solver.py, stripped to the essentials:
no validation, no options, no diagnostics.  Read this one to understand the
method; use the other one when you want the guardrails.

    dPhi/dt + alpha dA/dz = 0,   dA/dt + beta dPhi/dz = 0
    alpha = s*Z,  beta = s/Z,  s = wave speed,  Z = impedance = 1/(mu0 Sigma_A)
"""

from timebudget import timebudget
@timebudget
def wave_potentials_generator():
    import numpy as np
    import spaceToolsLib as stl
    from glob import glob
    from typing import Callable, Optional, Sequence
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_toggles import WavePotentialsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_classes import WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses


    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    data_dict_spatial = stl.loadDictFromFile(glob(rf"{RunToggles.sim_data_output_path}/spatial_grid/*.cdf")[0])
    data_dict_plasma = stl.loadDictFromFile(glob(rf"{RunToggles.sim_data_output_path}/plasma_environment/*.cdf")[0])
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

    data_dict_output = {
        'mu': data_dict_spatial['mu'].copy(),
        'chi': data_dict_spatial['chi'].copy(),
        'alt': data_dict_spatial['alt'].copy(),
        'r': data_dict_spatial['r'].copy(),
        'f0': [np.array([WavePotentialsToggles.f_0]), {'UNITS': 'Hz', 'LABLAXIS': 'Frequency'}],
        'k_perp': [[], {'DEPEND_0': 'alt', 'UNITS': 'm!A-1!N', 'LABLAXIS': 'k!B&perp;!N', 'VAR_TYPE': 'data'}],
        'lambda_perp': [[], {'DEPEND_0': 'alt', 'UNITS': 'm', 'LABLAXIS': '&lambda;!B&perp;!N', 'VAR_TYPE': 'data'}],
        'inertial_term': [[], {'DEPEND_0': 'alt', 'UNITS': None, 'LABLAXIS': '(1+(&lambda;k!B&perp;!N)!A2!N)!A1/2!N', 'VAR_TYPE': 'data'}],
        'alpha': [[], {'DEPEND_0': None}],
        'beta': [[], {'DEPEND_0': None}],
        'DAW_velocity_eV': [[], {'DEPEND_0': 'alt', 'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        'DAW_velocity': [[], {'DEPEND_0': 'alt', 'UNITS': 'm/s', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        'SIGMA_A': [[], {'DEPEND_0': 'alt', 'UNITS': 'S', 'LABLAXIS': 'Alfven Conductivity', 'VAR_TYPE': 'data'}],
    }

    # ==================================================
    # 0. WAVE PERPENDICULAR SCALE
    # ==================================================
    simMu = data_dict_spatial["mu"][0]
    simChi = data_dict_spatial["chi"][0]

    flux_tube_scaling = np.sqrt(envDict['B_dipole'](simMu[0], simChi[0]) / envDict['B_dipole'](simMu, simChi))
    lambda_perp = (WavePotentialsToggles.Lambda_perp0 * stl.m_to_km) * flux_tube_scaling
    k_perp = (2 * np.pi / lambda_perp)

    data_dict_output['lambda_perp'][0] = lambda_perp
    data_dict_output['k_perp'][0] = k_perp
    data_dict_output['inertial_term'][0] = np.sqrt(1 + np.square(k_perp * data_dict_plasma['lambda_e'][0]))
    data_dict_output['DAW_velocity'][0] = data_dict_plasma['V_A'][0] / data_dict_output['inertial_term'][0]
    data_dict_output['alpha'][0] = data_dict_plasma['V_A'][0]
    data_dict_output['beta'][0] = 1/data_dict_output['inertial_term'][0].copy()
    data_dict_output['SIGMA_A'][0] = 1/(stl.u0*data_dict_plasma['V_A'][0]*data_dict_output['inertial_term'][0])

    # ==================================================
    # 1. CHOOSE THE WAVE DRIVER
    # ==================================================
    inputs = (WavePotentialsToggles.Phi_0, WavePotentialsToggles.f_0)
    if WavePotentialsToggles.driver_dict['gaussian_pulse'] == 1:
        driver = WaveFieldsClasses().gaussian_pulse(*inputs)
    elif WavePotentialsToggles.driver_dict['gaussian_wavepacket'] == 1:
        driver = WaveFieldsClasses().gaussian_wavepacket(*inputs)
    elif WavePotentialsToggles.driver_dict['tapered_half_sine_pulse'] == 1:
        driver = WaveFieldsClasses().tapered_half_sine_pulse(*inputs)
    elif WavePotentialsToggles.driver_dict['tapered_sine_pulse'] == 1:
        driver = WaveFieldsClasses().tapered_sine(*inputs)

    # ==================================================
    # 2. SOLVE HYPERBOLIC PDEs
    # ==================================================
    s = data_dict_plasma['V_A'][0].copy() / data_dict_output['inertial_term'][0].copy()
    Z = data_dict_plasma['V_A'][0].copy() * data_dict_output['inertial_term'][0].copy()
    t_out, Phis, As, dA_dt, Flux_bot, Flux_top = WaveFieldsClasses().solve_hyperbolic(
        z=data_dict_spatial['alt'][0],
        s=s,
        Z=Z,
        # sigma_P=WavePotentialsToggles.SIGMA_P,
        sigma_P=data_dict_output['SIGMA_A'][0][0],
        drive=driver,
        t_end=WavePotentialsToggles.t_end,
        n_out=WavePotentialsToggles.n_out,
        cfl=WavePotentialsToggles.cfl
    )

    # ==================================================
    # 3. VERIFY CONSERVATION OF ENERGY
    # ==================================================
    z = data_dict_output['alt'][0].copy()
    zf = np.empty(z.size+1)
    zf[1:-1] = 0.5 * (z[:-1] + z[1:])
    zf[0] = z[0] - 0.5 * (z[1] - z[0])
    zf[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    dz = np.diff(zf)
    alpha, beta = s * Z, s / Z
    energy_density = 0.5 * (Phis ** 2 / alpha + As ** 2 / beta)
    E_inj = np.max(-Flux_top)
    E = np.sum(energy_density * dz, axis=1)
    E_norm = E/E_inj

    # ==================================================
    # 4. CALCULATE WAVE FIELDS
    # ==================================================
    E_perp = data_dict_output['k_perp'][0]*Phis # E⊥ = −∇⊥Φ→ |E⊥| = k_perp * Phi [V/m]
    B_perp = data_dict_output['k_perp'][0]*As #|B⊥| = k_perp * A_par      [T]
    E_para = np.square(data_dict_output['k_perp'][0]*data_dict_plasma['lambda_e'][0])*dA_dt
    S_para = np.square(data_dict_output['k_perp'][0])*As*Phis/stl.u0

    # ==================================================
    # 5. STORE OUTPUT
    # ==================================================
    data_dict_output.update({
        "time": [t_out, {"UNITS": "s", "LABLAXIS": "Time"}],
        "f0": [np.array([WavePotentialsToggles.f_0]), {"UNITS": "Hz", "LABLAXIS": "Frequency"}],
        "A_para": [As, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "Wb/m", "LABLAXIS": "A!Bz!N", "VAR_TYPE": "data"}],
        "Phi": [Phis, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "V", "LABLAXIS": "&Phi;", "VAR_TYPE": "data"}],
        'system_energy_normalized': [E_norm,{"DEPEND_0": "time", "UNITS": None, "LABLAXIS": "Normalized Energy", "VAR_TYPE": "data"}],
        'E_perp':[E_perp, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "V/m", "LABLAXIS": "E!B&perp;!N", "VAR_TYPE": "data"}],
        'B_perp': [B_perp, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "T", "LABLAXIS": "B!B&perp;!N", "VAR_TYPE": "data"}],
        'E_para': [E_para, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "V/m", "LABLAXIS": "E!B&para;!N", "VAR_TYPE": "data"}],
        'S_para': [S_para, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "T", "LABLAXIS": "W/m!A2!N", "VAR_TYPE": "data"}],
    })

    if RunToggles.store_output:
        outputPath = rf"{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf"
        stl.outputDataDict(outputPath, data_dict_output)

    return data_dict_output


