# Simulation Imports
from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_toggles import LiouvilleToggles
from scipy.special import gamma
from src.Alfvenic_Auroral_Acceleration_AAA.spatial_grid.spatial_classes import SpatialClasses
import numpy as np
import spaceToolsLib as stl
from itertools import product
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles

_WORKER = {}

def _init_worker(mapping_alt):
    _WORKER['obj'] = LiouvilleClasses(mapping_alt)

def _map_one_time(tmeIdx):
    return tmeIdx, _WORKER['obj'].map_single_time(tmeIdx)


class LiouvilleClasses:

    def __init__(self,mapping_alt):

        # form the regular grid interpolator for E-parallel
        from scipy.interpolate import RegularGridInterpolator
        data_dict_potentials = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')
        data_dict_spatial = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf')

        self.mapping_alt = mapping_alt
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]
        self.Te = envDict['Te']
        self.ne_density = envDict['n_density']
        self.chi0_obs = data_dict_spatial['chi'][0][0]
        self.r0 = 1 + self.mapping_alt / stl.Re
        self.colat0_rad = np.arcsin(np.sqrt(self.chi0_obs * self.r0))
        self.u0_obs = - np.sqrt(np.cos(self.colat0_rad)) / self.r0
        self.B0 = self.B_dipole(self.u0_obs, self.chi0_obs)

        mu_grid = data_dict_spatial['mu'][0]
        time_grid = data_dict_potentials['time'][0]
        self.Epara = RegularGridInterpolator((time_grid, mu_grid), data_dict_potentials['E_para'][0],bounds_error=False, fill_value=0.0)

    def map_single_time(self, tmeIdx):
        N_ptch = len(LiouvilleToggles.pitch_range_obs)
        N_engy = len(LiouvilleToggles.energy_range_obs)
        block = np.zeros((N_ptch, N_engy))

        for ptchIdx, engyIdx in product(range(N_ptch), range(N_engy)):
            engyVal = LiouvilleToggles.energy_range_obs[engyIdx]
            ptchVal = np.radians(LiouvilleToggles.pitch_range_obs[ptchIdx])
            speed = np.sqrt(2 * stl.q0 * engyVal / stl.m_e)
            vperp = speed * np.sin(ptchVal)
            vpara = speed * np.cos(ptchVal)
            s0 = [self.u0_obs, self.chi0_obs, -vpara, vperp]

            deltaT = LiouvilleToggles.obs_times[tmeIdx]
            uB = (0.5 * stl.m_e * np.square(vperp)) / self.B0

            T, p_mu, p_chi, p_vel_mu, p_vel_chi = self.rk45_solver(
                t_span=[0, -deltaT], s0=s0, deltaT=deltaT, uB=uB)

            B_mag_particle = self.B_dipole(p_mu, p_chi)
            mapped_v_perp = vperp * np.sqrt(B_mag_particle / self.B0)

            block[ptchIdx][engyIdx] = self.Maxwellian(
                vperp=mapped_v_perp[-1],
                vpara=-1 * p_vel_mu[-1],
                density=self.ne_density(p_mu[-1], p_chi[-1]),
                Te=self.Te(p_mu[-1], p_chi[-1]),
                Emin=10 ** LiouvilleToggles.E_min_obs,
                Emax=10 ** LiouvilleToggles.E_max_obs,
            )
        return block

    def liouville_mapper(self):
        import multiprocessing as mp
        from tqdm import tqdm

        N_time = len(LiouvilleToggles.obs_times)
        N_ptch = len(LiouvilleToggles.pitch_range_obs)
        N_engy = len(LiouvilleToggles.energy_range_obs)
        Distribution = np.zeros((N_time, N_ptch, N_engy))

        with mp.Pool(processes=20, initializer=_init_worker, initargs=(self.mapping_alt,)) as pool:
            for tmeIdx, block in tqdm(pool.imap_unordered(_map_one_time, range(N_time)), total=N_time):
                Distribution[tmeIdx] = block

        return Distribution

    # def liouville_mapper(self):
    #     import multiprocessing as mp
    #     from tqdm import tqdm
    #
    #     # --- DEFINE PARALLELIZED OUTPUTS ---
    #     N_time = len(LiouvilleToggles.obs_times)
    #     N_ptch = len(LiouvilleToggles.pitch_range_obs)
    #     N_engy = len(LiouvilleToggles.energy_range_obs)
    #
    #     # --- Prepare the output data array ---
    #     # Distribution Function Array (Parallel Process)
    #     mp_array_1 = mp.Array('d', N_time * N_ptch * N_engy)
    #     arr_1 = np.frombuffer(mp_array_1.get_obj())
    #     Distribution = arr_1.reshape((N_time, N_ptch, N_engy))
    #
    #     def parallel_processing_mapper(tmeIdx):
    #
    #         for ptchIdx, engyIdx in product(*[range(N_ptch), range(N_engy)]):
    #
    #             # get the initial state vector of the particle at z_obs
    #             engyVal = LiouvilleToggles.energy_range_obs[engyIdx]
    #             ptchVal = np.radians(LiouvilleToggles.pitch_range_obs[ptchIdx])
    #             vperp = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.sin(ptchVal)
    #             vpara = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.cos(ptchVal)
    #             v_mu = -1 * vpara # flip direction of v_mu to align with modified dipole coordinates
    #             s0 = [self.u0_obs, self.chi0_obs, v_mu, vperp]
    #
    #             # get the solver arguments
    #             deltaT = LiouvilleToggles.obs_times[tmeIdx]
    #             uB = (0.5 * stl.m_e * np.square(vperp)) / self.B0
    #
    #             # Perform the RK45 Solver for the equations of motion
    #             [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = self.rk45_solver(
    #                 t_span=[0, -deltaT],
    #                 s0=s0,
    #                 deltaT=deltaT,
    #                 uB = uB)
    #
    #
    #             ################################
    #             # --- PERPENDICULAR DYNAMICS ---
    #             ################################
    #             # geomagnetic field experienced by particle
    #             B_mag_particle = self.B_dipole(particle_mu.copy(), particle_chi.copy())
    #             mapped_v_perp = vperp * np.sqrt(B_mag_particle / self.B0 )
    #
    #             #######################################
    #             # --- CALCULATE MAPPED DISTRIBUTION ---
    #             #######################################
    #             Distribution[tmeIdx][ptchIdx][engyIdx] = self.Maxwellian(
    #                 vperp=mapped_v_perp[-1],
    #                 vpara=-1 *particle_vel_Mu[-1],
    #                 density= self.ne_density(particle_mu[-1],particle_chi[-1]),
    #                 Te=self.Te(particle_mu[-1],particle_chi[-1]),
    #                 Emin=10**LiouvilleToggles.E_min_obs,
    #                 Emax=10**LiouvilleToggles.E_max_obs
    #             )
    #
    #     processes_count = 20  # Number of CPU cores to commit to this operation
    #     pool_object = mp.Pool(processes_count)
    #     inputs = range(N_time)
    #     for _ in tqdm(pool_object.imap_unordered(parallel_processing_mapper, inputs), total=N_time):
    #         pass
    #
    #     return Distribution

    # The
    def equations_of_motion(self, t, S, deltaT, uB):
        # State Vector - [mu, chi, vel_mu, vel_chi]

        # --- Position ---
        # dmu/dt
        DmuDt = S[2] / self.h_factors[0](S[0], S[1])

        # dchi/dt
        # DchiDt = S[3] / self.h_factors[1](S[0], S[1])
        DchiDt = 0

        # --- Velocity ---
        # dv_mu/dt

        # magnetic mirroring
        DvmuDt_mirror = - (uB/stl.m_e) * (self.dB_dipole_dmu(S[0],S[1])/self.h_factors[0](S[0],S[1]))

        # inverted-V
        # DvmuDt_inV = (stl.q0/stl.m_e)*ElectrostaticPotentialClasses().invertedVEField([S[0],S[1],S[2]])

        # EM Field
        DvmuDt_Alfven = - (stl.q0 / stl.m_e) * self.Epara(np.array([[deltaT + t, S[0]]]))[0]

        # Combine all the parallel effects
        DvmuDt = DvmuDt_mirror
        # DvmuDt = DvmuDt_mirror + DvmuDt_Alfven

        # dv_chi/dt
        DvchiDt = 0

        return [DmuDt, DchiDt, DvmuDt, DvchiDt]

    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(self, t, S, deltaT, uB):

        alt = stl.Re*(SpatialClasses.r_muChi(S[0],S[1]) - 1)

        # top boundary checker
        top_boundary_checker = alt - LiouvilleToggles.upper_termination_altitude

        return top_boundary_checker

    escaped_upper.terminal = True

    def escaped_lower(self, t, S, deltaT, uB):
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
                         method=LiouvilleToggles.RK45_method,
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


