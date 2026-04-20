from timebudget import timebudget
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *

@timebudget
def wave_fields_generator():
    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
    from tqdm import tqdm
    from scipy.integrate import simpson
    import multiprocessing as mp

    # --- Load the needed data ---
    data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/ray_equations/ray_equations.cdf')[0])
    data_dict_plasEvrn = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/plasma_environment.cdf')[0])

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_ray_eqns['time'][0])),deepcopy(data_dict_ray_eqns['time'][1])],
        'mu_w': deepcopy(data_dict_ray_eqns['mu_w']),
        'chi_w': deepcopy(data_dict_ray_eqns['chi_w']),
        'z': deepcopy(data_dict_ray_eqns['z']),
        'E_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'V/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'Az': [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'Wb/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'Phi': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'V', 'LABLAXIS': '&Phi;', 'VAR_TYPE': 'data'}],
        'resonance_low': [[],{'DEPEND_0': 'z', 'UNITS': 'eV', 'LABLAXIS': 'Resonance Low', 'VAR_TYPE': 'data'}],
        'resonance_high': [[], {'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'Resonance High', 'VAR_TYPE': 'data'}],
        'DAW_velocity_eV':[[],{'DEPEND_0': 'z',  'UNITS': 'eV', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
        'DAW_velocity': [[], {'DEPEND_0': 'z', 'UNITS': 'm/s', 'LABLAXIS': 'DAW Velocity', 'VAR_TYPE': 'data'}],
    }

    # --- load the environment variables ---
    from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
    envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

    # --- Form the spatial simulation grid ---
    N_alt = 1000
    Rf = (1 + RayEquationToggles.upper_boundary/stl.Re)
    Theta_at_zf = np.arcsin(np.sqrt(RayEquationToggles.chi0_w * Rf))
    muF = -np.sqrt(np.cos(Theta_at_zf))/Rf

    simMUs = np.linspace(RayEquationToggles.u0_w,muF, N_alt)
    simChis = np.array([RayEquationToggles.chi0_w for i in range(N_alt)])
    simAlts = (SimClasses.r_muChi(simMUs, simChis)-1)*stl.Re

    # --- Form the temporal simulation grid ---
    alpha = np.square(envDict['V_A'](simMUs, simChis)) / (1 + np.square(envDict['V_A'](simMUs, simChis)/stl.lightSpeed ))
    lambda_e = envDict['lambda_e'](simMUs,simChis)
    k_perp = (2*np.pi/RayEquationToggles.Lambda_perp0)* np.sqrt(envDict['B_dipole'](simMUs, simChis)/envDict['B_dipole'](RayEquationToggles.u0_w, RayEquationToggles.chi0_w))
    beta = 1/(1 + np.square(k_perp*lambda_e))

    s = np.sqrt(alpha*beta)
    sim_length = 1 # in seconds
    deltaTs = 0.89*np.diff(simAlts*stl.m_to_km)/np.max(s) # the CFL stability criteria requires deltaT <= deltaZ/max(s)
    deltaT_stable = min(deltaTs)
    N_points_time = int(sim_length/deltaT_stable)
    simDeltaT = np.linspace(0,sim_length,N_points_time)

    # --- form the solution arrays ---
    Az = np.zeros(shape=(N_points_time,N_alt))
    Phi = np.zeros(shape=(N_points_time, N_alt))

    # --- Form the boundary condition values ---
    SIGMA_P = 1 # in mhos
    SIGMA_A = 1/(stl.u0*np.sqrt(alpha)) # Alfven conductance throughout entire simulation region
    sigmaP = stl.u0*SIGMA_P
    sigmaA = stl.u0*SIGMA_A[-1] # at the magnetosphere

    # --- Create the driving Electrostatic Potential function ---

    # sinusoid
    Phi0_driver= 3600 # in eV
    k_perp_driver = k_perp[-1]
    freq_driver = 4 # in Hz
    def driver_func(t, z, Phi0_driver, k_perp_driver,freq_driver):
        if t >= 1/freq_driver:
            return 0
        else:
            return (Phi0_driver)*np.sin(t*2*np.pi*freq_driver)


    ###############################
    # --- RK45 SOLVE THE SYSTEM ---
    ###############################
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # ==================================================
    # 1. NON-UNIFORM GRID
    # ==================================================
    z = simAlts*stl.m_to_km

    # precompute grid spacing
    dz_fwd = np.diff(z, append=z[-1])
    dz_bwd = np.diff(z, prepend=z[0])

    # ==================================================
    # 2. PHYSICAL PARAMETERS
    # ==================================================
    s = np.sqrt(alpha * beta)

    # ==================================================
    # 3. BOUNDARY PARAMETERS (ORIGINAL PROBLEM)
    # ==================================================
    Sigma_P = 0
    Sigma_A = 0

    sigma_P = 4 * np.pi * Sigma_P / stl.lightSpeed
    sigma_A = 4 * np.pi * Sigma_A / stl.lightSpeed

    def Phi0(t):
        if t > (1 / freq_driver):
            return 0
        else:
            return Phi0_driver*np.sin(2 * np.pi * freq_driver * t)  # sinusoidal driver

    # ==================================================
    # 4. INITIAL CONDITIONS
    # ==================================================
    A0 = np.zeros_like(z)
    Phi0_init = np.exp(-80 * (z - 0.5) ** 2)

    wp = np.sqrt(alpha) * A0 + np.sqrt(beta) * Phi0_init
    wm = np.sqrt(alpha) * A0 - np.sqrt(beta) * Phi0_init

    U0 = np.concatenate([wp, wm])

    # ==================================================
    # 5. NON-UNIFORM UPWIND DERIVATIVES
    # ==================================================
    def dw_plus_dz(wp):
        d = np.zeros_like(wp)
        for i in range(1, len(wp)):
            d[i] = (wp[i] - wp[i - 1]) / dz_bwd[i]
        d[0] = d[1]
        return d

    def dw_minus_dz(wm):
        d = np.zeros_like(wm)
        for i in range(len(wm) - 1):
            d[i] = (wm[i + 1] - wm[i]) / dz_fwd[i]
        d[-1] = d[-2]
        return d

    # ==================================================
    # 6. RHS (FULL PHYSICS + CORRECT BCs)
    # ==================================================
    def rhs(t, U):

        wp = U[:N_alt].copy()
        wm = U[N_alt:].copy()

        # reconstruct physical variables
        A = (wp + wm) / (2 * np.sqrt(alpha))
        Phi = (wp - wm) / (2 * np.sqrt(beta))

        # ==================================================
        # LEFT BOUNDARY (A + sigma_P Phi = 0)
        # ==================================================
        wp[0] = (
                        -wm[0] * (1 / np.sqrt(alpha[0]) + sigma_P / np.sqrt(beta[0]))
                ) / (
                        1 / np.sqrt(alpha[0]) - sigma_P / np.sqrt(beta[0])
                )

        # ==================================================
        # RIGHT BOUNDARY (impedance + sinusoidal drive)
        # ==================================================

        # impedance-consistent outgoing solution
        wm_imp = (
                         - wp[-1] * (1 / np.sqrt(alpha[-1]) - sigma_A / np.sqrt(beta[-1]))
                         - 2 * sigma_A * 0.0
                 ) / (
                         1 / np.sqrt(alpha[-1]) + sigma_A / np.sqrt(beta[-1])
                 )

        # add controlled sinusoidal injection
        eta = 1
        wm[-1] = wm_imp + eta * Phi0(t)

        # ==================================================
        # CHARACTERISTIC EVOLUTION
        # ==================================================
        dwp = -s * dw_plus_dz(wp)
        dwm = s * dw_minus_dz(wm)

        return np.concatenate([dwp, dwm])

    # ==================================================
    # 7. TIME INTEGRATION
    # ==================================================
    t0, t1 = 0.0, 14.0

    frames = 200
    t_eval = np.linspace(t0, t1, frames)

    sol = solve_ivp(
        rhs,
        (t0, t1),
        U0,
        method='RK45',
        # t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    # ==================================================
    # 8. RECONSTRUCT PHYSICAL VARIABLES
    # ==================================================
    wp = sol.y[:N_alt, :]
    wm = sol.y[N_alt:, :]

    A = (wp + wm) / (2 * np.sqrt(alpha[:, None]))
    Phi = (wp - wm) / (2 * np.sqrt(beta[:, None]))

    for thing in list(Phi):
        print(thing)
    print(np.shape(Phi))

    # # ==================================================
    # # 9. ANIMATION
    # # ==================================================
    # fig, ax = plt.subplots()
    #
    # line1, = ax.plot([], [], label="A")
    # line2, = ax.plot([], [], label="Phi")
    #
    # ax.set_xlim(simAlts[0], simAlts[-1])
    # ax.set_ylim(-5000, 5000)
    # ax.legend()
    #
    # title = ax.set_title("")
    #
    # def update(i):
    #     line1.set_data(z, A[:, i])
    #     line2.set_data(z, Phi[:, i])
    #     title.set_text(f"t = {sol.t[i]:.3f}")
    #     return line1, line2, title
    #
    # anim = FuncAnimation(fig, update, frames=frames)
    #
    # anim.save("/home/connor/Desktop/final_characteristic_solver.gif",
    #           writer=PillowWriter(fps=20))
    #
    # print("Saved: final_characteristic_solver.gif")

    data_dict_output['Az'][0] = A.T
    data_dict_output['Phi'][0] = Phi.T
    data_dict_output['z'][0] = z
    data_dict_output['time'][0] = sol.t

    # ==================================================
    # 10. OUTPUT DATA
    # ==================================================
    outputPath = rf'{WaveFieldsToggles.outputFolder}/wave_fields_rk45.cdf'
    stl.outputDataDict(outputPath, data_dict_output)

    if SimToggles.store_output:
        # save the results
        outputPath = rf'{ResultsToggles.outputFolder}/{DistributionToggles.z0_obs}km/wave_fields_{DistributionToggles.z0_obs}km_rk45.cdf'
        stl.outputDataDict(outputPath, data_dict_output)



