"""
alfven_min.py -- minimal version of the solver.

Same scheme and same answers as alfven_solver.py, stripped to the essentials:
no validation, no options, no diagnostics.  Read this one to understand the
method; use the other one when you want the guardrails.

    dPhi/dt + alpha dA/dz = 0,   dA/dt + beta dPhi/dz = 0
    alpha = s*Z,  beta = s/Z,  s = wave speed,  Z = impedance = 1/(mu0 Sigma_A)
"""


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
from typing import Callable, Optional, Sequence


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
        'Sigma_A': [[], {'DEPEND_0': 'alt', 'UNITS': 'S', 'LABLAXIS': 'Alfven Conductivity', 'VAR_TYPE': 'data'}],
    }

    simMu = data_dict_spatial["mu"][0]
    simChi = data_dict_spatial["chi"][0]          # <-- confirm the units of this!

    # ==================================================
    # 0. WAVE PERPENDICULAR SCALE
    # ==================================================
    flux_tube_scaling = np.sqrt(envDict['B_dipole'](simMu[0], simChi[0]) / envDict['B_dipole'](simMu, simChi))
    lambda_perp = (WavePotentialsToggles.Lambda_perp0 * stl.m_to_km) * flux_tube_scaling
    k_perp = (2 * np.pi / lambda_perp)
    data_dict_output['lambda_perp'][0] = lambda_perp
    data_dict_output['k_perp'][0] = k_perp
    data_dict_output['inertial_term'][0] = np.sqrt(1 + np.square(k_perp * data_dict_plasma['lambda_e'][0]))
    data_dict_output['DAW_velocity'][0] = data_dict_plasma['V_A'][0] / data_dict_output['inertial_term'][0]


    MU0 = 4.0e-7 * np.pi
    s = data_dict_plasma['V_A'][0].copy()/data_dict_output['inertial_term'][0].copy()
    Z= data_dict_plasma['V_A'][0].copy()*data_dict_output['inertial_term'][0].copy()

    def half_sine_pulse(Phi_0: float, f_0: float) -> Callable[[float], float]:
        """Single positive half-sine hump of duration 1/f_0 (your current driver).

        Note this is unipolar: it carries a net DC offset, so the domain retains a
        residual potential after the pulse leaves.  Use `gaussian_wavepacket` if you
        want clean spectral content around f_0.
        """
        omega = 2.0 * np.pi * f_0

        def drive(t: float) -> float:
            if t < 0.0 or t >= 1.0 / f_0:
                return 0.0
            return Phi_0 * np.sin(0.5 * omega * t)

        return drive

    driver = half_sine_pulse(WavePotentialsToggles.Phi_0, WavePotentialsToggles.f_0)

    def solve(z, s, Z, sigma_P, drive, t_end, n_out=200, cfl=0.4, iono_sign=+1):
        """z increasing, ionosphere first.  drive(t) -> Phi_0(t).

        iono_sign=+1 applies  A + mu0 sigma_P Phi = 0  (the physical one).
        """
        N = z.size
        zf = np.r_[z[0] - (z[1] - z[0]) / 2, (z[:-1] + z[1:]) / 2, z[-1] + (z[-1] - z[-2]) / 2]
        dz = np.diff(zf)
        alpha, beta = s * Z, s / Z
        ZL, ZR = Z[:-1], Z[1:]
        zfi = zf[1:-1]  # interior faces
        Phi, A = np.zeros(N), np.zeros(N)

        def faces(q):
            """Linear reconstruction to both sides of every interior face."""
            sl = np.zeros(N)
            sl[1:-1] = (q[2:] - q[:-2]) / (z[2:] - z[:-2])  # central slopes; 0 in end cells
            return q[:-1] + sl[:-1] * (zfi - z[:-1]), q[1:] + sl[1:] * (zfi - z[1:])

        def rhs(Phi, A, t):
            PhiL, PhiR = faces(Phi)
            AL, AR = faces(A)
            a = PhiL + ZL * AL  # right-going invariant, from the left cell
            b = PhiR - ZR * AR  # left-going  invariant, from the right cell
            Af, Pf = np.empty(N + 1), np.empty(N + 1)
            Af[1:-1] = (a - b) / (ZL + ZR)  # exact Riemann solution at the face
            Pf[1:-1] = (ZR * a + ZL * b) / (ZL + ZR)

            g = iono_sign * MU0 * sigma_P * Z[0]  # = Sigma_P / Sigma_A
            Pf[0] = (Phi[0] - Z[0] * A[0]) / (1.0 + g)  # only W^- reaches this face
            Af[0] = -iono_sign * MU0 * sigma_P * Pf[0]

            d = drive(t)  # matched -> transparent
            Pf[-1] = (Phi[-1] + Z[-1] * A[-1] + d) / 2.0  # only W^+ reaches this face
            Af[-1] = (Pf[-1] - d) / Z[-1]

            return -alpha * np.diff(Af) / dz, -beta * np.diff(Pf) / dz

        dt = cfl * np.min(dz / s)
        t_out = np.linspace(0.0, t_end, n_out)
        Phis, As = np.zeros((n_out, N)), np.zeros((n_out, N))
        t = 0.0
        for k in range(1, n_out):
            while t < t_out[k] - 1e-15:
                h = min(dt, t_out[k] - t)
                kp, ka = rhs(Phi, A, t)  # SSP-RK2
                P1, A1 = Phi + h * kp, A + h * ka
                kp2, ka2 = rhs(P1, A1, t + h)
                Phi, A = 0.5 * (Phi + P1 + h * kp2), 0.5 * (A + A1 + h * ka2)
                t += h
            Phis[k], As[k] = Phi, A
        return t_out, Phis, As

    t_out, Phis, As = solve(
        z=data_dict_spatial['alt'][0],
        s=s,
        Z=Z,
        sigma_P=1,
        drive=driver,
        t_end=3,
    )

    # ------------------------------------------------------------------
    # package output
    # ------------------------------------------------------------------
    data_dict_output.update({
        "time": [t_out, {"UNITS": "s", "LABLAXIS": "Time"}],
        "f0": [np.array([WavePotentialsToggles.f_0]), {"UNITS": "Hz", "LABLAXIS": "Frequency"}],
        "A_para": [As, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "Wb/m", "LABLAXIS": "A!Bz!N", "VAR_TYPE": "data"}],
        "Phi": [Phis, {"DEPEND_0": "time", "DEPEND_1": "alt", "UNITS": "V", "LABLAXIS": "&Phi;", "VAR_TYPE": "data"}],
    })

    if RunToggles.store_output:
        outputPath = rf"{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf"
        stl.outputDataDict(outputPath, data_dict_output)

    return data_dict_output

wave_potentials_generator()


