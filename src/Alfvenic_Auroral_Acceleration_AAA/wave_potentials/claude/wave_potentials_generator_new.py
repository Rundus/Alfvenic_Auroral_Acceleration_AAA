"""
wave_potentials_generator_new.py
================================

Drop-in replacement for `wave_potentials_generator()`.  Keeps your data_dict
plumbing and toggles; replaces the numerics with `alfven_solver`.

Put `alfven_solver.py` somewhere importable, e.g.
    src/Alfvenic_Auroral_Acceleration_AAA/wave_potentials/alfven_solver.py
and adjust the import below.
"""

from timebudget import timebudget


@timebudget
def wave_potentials_generator():
    import numpy as np
    import spaceToolsLib as stl
    from glob import glob

    from src.Alfvenic_Auroral_Acceleration_AAA.wave_potentials.wave_potentials_toggles import (
        WavePotentialsToggles,
    )
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import (
        EnvironmentExpressionsClasses,
    )
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_potentials.claude.alfven_solver import (
        AlfvenSolver,
        FieldLineMedium,
        half_sine_pulse,
    )

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    data_dict_spatial = stl.loadDictFromFile(
        glob(rf"{RunToggles.sim_data_output_path}/spatial_grid/*.cdf")[0]
    )
    data_dict_plasma = stl.loadDictFromFile(
        glob(rf"{RunToggles.sim_data_output_path}/plasma_environment/*.cdf")[0]
    )
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

    simMu = data_dict_spatial["mu"][0]
    simChi = data_dict_spatial["chi"][0]
    simAlt = data_dict_spatial["z"][0]        # <-- confirm the units of this!

    # ------------------------------------------------------------------
    # geometry and medium
    # ------------------------------------------------------------------
    # NOTE ON UNITS.  Everything below is SI: z in metres, V_A in m/s,
    # lambda_e in m, k_perp in 1/m.  Your original code multiplied both the
    # altitude and Lambda_perp0 by `stl.m_to_km`, which can only be correct for
    # one of them.  Set KM_TO_M explicitly and delete the ambiguity.
    KM_TO_M = 1.0e3

    z = np.asarray(simAlt, dtype=float)       # if simAlt is in km: z = simAlt * KM_TO_M
    if not np.all(np.diff(z) > 0):            # solver needs ionosphere -> magnetosphere
        raise ValueError("z must be strictly increasing; flip your arrays if needed")

    flux_tube_scaling = np.sqrt(
        envDict["B_dipole"](simMu[0], simChi[0]) / envDict["B_dipole"](simMu, simChi)
    )
    lambda_perp = (WavePotentialsToggles.Lambda_perp0 * KM_TO_M) * flux_tube_scaling
    k_perp = 2.0 * np.pi / lambda_perp

    V_A = np.asarray(data_dict_plasma["V_A"][0], dtype=float)
    lambda_e = np.asarray(data_dict_plasma["lambda_e"][0], dtype=float)

    medium = FieldLineMedium(z=z, V_A=V_A, lambda_e=lambda_e, k_perp=k_perp)
    print(medium.summary())

    # resolution guard: you need >~ 12 cells per wavelength everywhere
    ppw = (medium.s / WavePotentialsToggles.f_0) / medium.dz
    if ppw.min() < 12:
        print(
            f"WARNING: only {ppw.min():.1f} cells per wavelength at "
            f"z = {medium.z[np.argmin(ppw)]:.3e} m -- refine the grid there"
        )

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    driver = half_sine_pulse(WavePotentialsToggles.Phi_0, WavePotentialsToggles.f_0)

    solver = AlfvenSolver(
        medium,
        sigma_P=WavePotentialsToggles.SIGMA_P,
        driver=driver,
        order=2,
        limiter="mc",
        time_integrator="rk3",
        cfl=0.4,
    )
    result = solver.run(
        t_end=WavePotentialsToggles.t1,
        t_start=WavePotentialsToggles.t0,
        n_snapshots=getattr(WavePotentialsToggles, "n_frames", 400),
        progress=True,
    )
    print(f"steps: {result.n_steps}, dt = {result.dt:.3e} s")

    # ------------------------------------------------------------------
    # package output
    # ------------------------------------------------------------------
    data_dict_output = {
        "time": [result.t, {"UNITS": "s", "LABLAXIS": "Time"}],
        "mu": data_dict_spatial["mu"].copy(),
        "chi": data_dict_spatial["chi"].copy(),
        "z": [medium.z, {"UNITS": "m", "LABLAXIS": "z"}],
        "r": data_dict_spatial["r"].copy(),
        "f0": [np.array([WavePotentialsToggles.f_0]), {"UNITS": "Hz", "LABLAXIS": "Frequency"}],
        "k_perp": [k_perp, {"DEPEND_0": "z", "UNITS": "m!A-1!N", "LABLAXIS": "k!B&perp;!N", "VAR_TYPE": "data"}],
        "lambda_perp": [lambda_perp, {"DEPEND_0": "z", "UNITS": "m", "LABLAXIS": "&lambda;!B&perp;!N", "VAR_TYPE": "data"}],
        "inertial_term": [medium.inertial, {"DEPEND_0": "z", "UNITS": None, "LABLAXIS": "(1+(&lambda;k!B&perp;!N)!A2!N)!A1/2!N", "VAR_TYPE": "data"}],
        "alpha": [medium.alpha, {"DEPEND_0": "z"}],
        "beta": [medium.beta, {"DEPEND_0": "z"}],
        "Az": [result.A, {"DEPEND_0": "time", "DEPEND_1": "z", "UNITS": "Wb/m", "LABLAXIS": "A!Bz!N", "VAR_TYPE": "data"}],
        "Phi": [result.Phi, {"DEPEND_0": "time", "DEPEND_1": "z", "UNITS": "V", "LABLAXIS": "&Phi;", "VAR_TYPE": "data"}],
        # this is the true wave speed: v_A/(1+k_perp^2 lambda_e^2)^{1/2}, with the
        # relativistic correction included.  Your original used V_A, not v_A.
        "DAW_velocity": [medium.s, {"DEPEND_0": "z", "UNITS": "m/s", "LABLAXIS": "DAW Velocity", "VAR_TYPE": "data"}],
        "Sigma_A": [medium.Sigma_A, {"DEPEND_0": "z", "UNITS": "S", "LABLAXIS": "Alfven Conductivity", "VAR_TYPE": "data"}],
        "energy": [result.energy, {"DEPEND_0": "time", "LABLAXIS": "Wave energy", "VAR_TYPE": "data"}],
    }

    if RunToggles.store_output:
        outputPath = rf"{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf"
        stl.outputDataDict(outputPath, data_dict_output)

    return data_dict_output

wave_potentials_generator()