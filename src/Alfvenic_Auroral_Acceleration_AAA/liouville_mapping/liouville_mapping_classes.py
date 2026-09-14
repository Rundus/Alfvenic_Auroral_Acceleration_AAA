# Simulation Imports
from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_toggles import LiouvilleToggles
from scipy.special import gamma
from src.Alfvenic_Auroral_Acceleration_AAA.spatial_grid.spatial_classes import SpatialClasses
import numpy as np
import spaceToolsLib as stl
from itertools import product
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles




class LiouvilleClasses:

    def __init__(self,mapping_alt):

        # --- DEFINE INITIAL CONDITIONS ---
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
        envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
        from glob import glob
        from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
        data_dict_spatial = stl.loadDictFromFile(glob(rf"{RunToggles.sim_data_output_path}/spatial_grid/*.cdf")[0])
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]
        self.mapping_alt = mapping_alt
        self.Te = envDict['Te']
        self.ne_density = envDict['n_density']
        self.chi0_obs = data_dict_spatial['chi'][0][0]
        self.r0 = 1 + self.mapping_alt / stl.Re
        self.colat0_rad = np.arcsin(np.sqrt(self.chi0_obs * self.r0))
        self.u0_obs = - np.sqrt(np.cos(self.colat0_rad)) / self.r0
        self. B0 = self.B_dipole(self.u0_obs, self.chi0_obs)

        # form the regular grid interpolator for E-parallel
        from scipy.interpolate import RegularGridInterpolator
        data_dict_potentials = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')
        data_dict_spatial = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf')
        mu_grid = data_dict_spatial['mu'][0]
        time_grid = data_dict_potentials['time'][0]
        self.Epara = RegularGridInterpolator((time_grid, mu_grid), data_dict_potentials['E_para'][0])

    # The
    def equations_of_motion(self, t, S, deltaT, uB):
        # State Vector - [mu, chi, vel_mu, vel_chi]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[2] / self.h_factors[0](S[0], S[1])

        # dchi/dt
        # DchiDt = S[3] / self.h_factors[1](S[0], S[1])
        DchiDt = 0

        # --- Velocity ---
        # dv_mu/dt

        # magnetic mirroring effects
        DvmuDt_mirror = - (uB/stl.m_e) * (self.dB_dipole_dmu(S[0],S[1])/self.h_factors[0](S[0],S[1]))

        # inverted-V effects
        # DvmuDt_inV = (stl.q0/stl.m_e)*ElectrostaticPotentialClasses().invertedVEField([S[0],S[1],S[2]])

        # Wave Field Effects
        # print(f'DeltaT={deltaT}',f'T={t+deltaT}\n')
        DvmuDt_Alfven =  - (stl.q0 / stl.m_e) * self.Epara(deltaT + t, S[0])
        # Combine all the parallel effects together
        DvmuDt = DvmuDt_mirror
        # DvmuDt = DvmuDt_mirror + DvmuDt_Alfven

        # dv_chi/dt
        DvchiDt = 0

        return [DmuDt, DchiDt, DvmuDt, DvchiDt]

    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(self, S):

        alt = stl.Re*(SpatialClasses.r_muChi(S[0],S[1]) - 1)

        # top boundary checker
        top_boundary_checker = alt - LiouvilleToggles.upper_termination_altitude

        return top_boundary_checker

    escaped_upper.terminal = True

    def escaped_lower(self, S):
        alt = stl.Re  * (SpatialClasses.r_muChi(S[0], S[1]) - 1)

        # lower boundary
        lower_boundary_checker = alt - LiouvilleToggles.lower_termination_altitude

        return lower_boundary_checker
    escaped_lower.terminal = True

    #####################
    # --- RK45 SOLVER ---
    #####################
    def rk45_solver(self, t_span, s0, deltaT, uB):
        from scipy.integrate import solve_ivp

        soln = solve_ivp(fun=self.equations_of_motion,
                         t_span=t_span,
                         y0=s0,
                         method=self.RK45_method,
                         rtol=LiouvilleToggles.RK45_rtol,
                         atol=LiouvilleToggles.RK45_atol,
                         events=[self.escaped_lower, self.escaped_upper],
                         args=tuple([deltaT, uB])
                         )
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        vel_Mu = soln.y[2, :]
        vel_chi = soln.y[3, :]
        return [T, particle_mu, particle_chi, vel_Mu, vel_chi]

    def liouville_mapper(self, mapping_alt):
        import multiprocessing as mp
        import tqdm as tqdm

        # --- DEFINE PARALLELIZED OUTPUTS ---
        Ntimes = len(LiouvilleToggles.obs_times)
        Nptchs = len(LiouvilleToggles.pitch_range_obs)
        Nengy = len(LiouvilleToggles.energy_range_obs)

        # --- Prepare the output data array ---
        # Distribution Function Array (Parallel Process)
        mp_array_1 = mp.Array('d', Ntimes * Nptchs * Nengy)
        arr_1 = np.frombuffer(mp_array_1.get_obj())
        Distribution = arr_1.reshape((Ntimes, Nptchs, Nengy))

        def parallel_processing_mapper(tmeIdx):

            for ptchIdx, engyIdx in product(*[range(Nptchs), range(Nengy)]):

                # get the initial state vector of the particle at z_obs
                engyVal = LiouvilleToggles.energy_range_obs[engyIdx]
                ptchVal = np.radians(LiouvilleToggles.pitch_range_obs[ptchIdx])
                vperp = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.sin(ptchVal)
                vpara = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.cos(ptchVal)
                v_mu = -1 * vpara # flip direction of v_mu to align with modified dipole coordinates
                s0 = [self.u0_obs, self.chi0_obs, v_mu, vperp]

                # get the solver arguments
                deltaT = LiouvilleToggles.obs_times[tmeIdx]
                uB = (0.5 * stl.m_e * np.square(vperp)) / self.B0

                # Perform the RK45 Solver for the equations of motion
                [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = self.rk45_solver(
                    t_span=[0, -deltaT],
                    s0=s0,
                    deltaT=deltaT,
                    uB = uB)


                ################################
                # --- PERPENDICULAR DYNAMICS ---
                ################################
                # geomagnetic field experienced by particle
                B_mag_particle = self.B_dipole(particle_mu.copy(), particle_chi.copy())
                mapped_v_perp = vperp * np.sqrt(B_mag_particle / np.array([self.B0 for i in range(len(B_mag_particle))]))

                #######################################
                # --- CALCULATE MAPPED DISTRIBUTION ---
                #######################################
                Distribution[tmeIdx][ptchIdx][engyIdx] = LiouvilleClasses().Maxwellian(
                    vperp=mapped_v_perp[-1],
                    vpara=-1 *particle_vel_Mu[-1],
                    density= self.ne_density(particle_mu[-1],particle_chi[-1]),
                    Te=self.Te(particle_mu[-1],particle_chi[-1]),
                    Emax=10**LiouvilleToggles.E_min_obs,
                    Emin=10**LiouvilleToggles.E_max_obs
                )

        processes_count = 20  # Number of CPU cores to commit to this operation
        pool_object = mp.Pool(processes_count)
        inputs = range(Ntimes)
        for _ in tqdm(pool_object.imap_unordered(parallel_processing_mapper, inputs), total=Ntimes):
            pass

        return Distribution



    ################################
    # --- DISTRIBUTION FUNCTIONS ---
    ################################
    def Maxwellian(self, vperp, vpara, density, Te, Emax, Emin):
        """
                :param vpara: Particle Velocity parallel to the background geomagnetic field in [m/s]
                :type vpara: float

                :param vperp: Particle Velocity parallel to the background geomagnetic field in [m/s]
                :type vperp: float

                :return: Plasma Distribution Function in [m^-6 s^-3] evaluated at vpara, vperp
                """
        if 0.5 * (stl.m_e / stl.q0) * (np.square(vpara) + np.square(vperp)) > Emax:  # check if energy is above the specific level the distribution
            return 0
        elif 0.5 * (stl.m_e / stl.q0) * (np.square(vpara) + np.square(vperp)) < Emin:  # check if energy is below the specific level the distribution:
            return 0
        else:
            return density * np.sqrt(np.power(stl.m_e / (2 * np.pi * Te * stl.q0), 3)) * np.exp(-0.5 * stl.m_e * (np.square(vperp) + np.square(vpara)) / (stl.q0 * Te))

    def Kappa(self, mass, Vperp,Vpara,charge ,n, Te, vpara, vperp, kappa):
        # Input: density [cm^-3], Temperature [eV], Velocities [m/s]
        # output: the distribution function in SI units [s^3 m^-6]
        Emag = (0.5 * mass * (Vperp ** 2 + Vpara ** 2)) / charge
        Ek = Te * (1 - 3 / (2 * kappa))
        return (1E6) * n * np.power(mass / (2 * np.pi * kappa * stl.q0 * Ek), 3 / 2) * (gamma(kappa + 1) / gamma(kappa - 0.5)) * np.power(1 + Emag / (kappa * Ek), -(kappa + 1))


