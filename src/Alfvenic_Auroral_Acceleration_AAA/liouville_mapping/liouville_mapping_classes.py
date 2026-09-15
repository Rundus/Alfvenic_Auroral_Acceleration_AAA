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
import math
from scipy.interpolate import RegularGridInterpolator

_WORKER = {}

def _init_worker(mapping_alt):
    _WORKER['obj'] = LiouvilleClasses(mapping_alt)

def _map_one_time(tmeIdx):
    return tmeIdx, _WORKER['obj'].map_single_time(tmeIdx)


class LiouvilleClasses:

    def __init__(self,mapping_alt):

        # form the regular grid interpolator for E-parallel
        data_dict_potentials = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')
        data_dict_spatial = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf')

        # Calculate the Observation Info
        self.mapping_alt = mapping_alt
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]
        self.Te = envDict['Te']
        self.ne_density = envDict['n_density']
        self.chi_obs = data_dict_spatial['chi'][0][0]
        self.r0 = 1 + self.mapping_alt / stl.Re
        self.colat0_rad = np.arcsin(np.sqrt(self.chi_obs * self.r0))
        self.u0_obs = - np.sqrt(np.cos(self.colat0_rad)) / self.r0
        self.B0 = self.B_dipole(self.u0_obs, self.chi_obs)
        self.observation_times = np.linspace(LiouvilleToggles.time_obs_start, LiouvilleToggles.time_obs_end, LiouvilleToggles.N_obs_points) # list of observation times

        # Calculate Loss Cone properties
        self.r_lost = 1 + LiouvilleToggles.alt_lost / stl.Re
        self.chi_lost = data_dict_spatial['chi'][0][0]
        self.colat_lost = np.arcsin(np.sqrt(self.chi_lost * self.r_lost))
        self.mu_lost = - np.sqrt(np.cos(self.colat_lost)) / self.r_lost
        self.B_lost = self.B_dipole(self.mu_lost, self.chi_lost)
        self.mu_eq = -1E-4 # very close to zero but not quite to avoid singularities. Represents the geomagnetic equator for perfect dipole
        self.chi_eq = data_dict_spatial['chi'][0][0]
        self.B_eq = self.B_dipole(self.mu_eq, self.chi_eq)
        self.pitch_eq_lost = math.asin(math.sqrt(self.B_eq/self.B_lost))

        # Construct the Interpolator Object
        self.mu_grid = data_dict_spatial['mu'][0]
        self.time_grid = data_dict_potentials['time'][0]
        self.Epara = data_dict_potentials['E_para'][0].copy()
        self.Eperp = data_dict_potentials['E_perp'][0].copy()
        self.Bperp = data_dict_potentials['B_perp'][0].copy()

        if LiouvilleToggles.injected_wave_time_delay > 0:

            # --- Adjust the wave Interpolator ---
            deltaT = np.gradient(self.time_grid)[0]
            N_additional_points = int(LiouvilleToggles.injected_wave_time_delay/deltaT)
            zeros = np.zeros((N_additional_points, self.Epara.shape[1]), dtype=self.Epara.dtype)

            # adjust the fields size
            self.Epara = np.vstack([zeros, self.Epara])
            self.Eperp = np.vstack([zeros, self.Eperp])
            self.Bperp = np.vstack([zeros, self.Bperp])

            # adjust the fields time grid size
            self.time_grid = np.concatenate([np.array([deltaT*i for i in range(N_additional_points)]),self.time_grid+LiouvilleToggles.injected_wave_time_delay])

            # --- Adjust the observation times ---
            deltaT_obs = np.gradient(self.observation_times)[0]
            N_additional_obs_points = int(LiouvilleToggles.injected_wave_time_delay/deltaT_obs)
            self.observation_times = np.concatenate([np.array([deltaT_obs*i for i in range(N_additional_obs_points)]),self.observation_times+LiouvilleToggles.injected_wave_time_delay])

        self.Epara = RegularGridInterpolator((self.time_grid, self.mu_grid),self.Epara,bounds_error=False, fill_value=0.0)

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
            s0 = [self.u0_obs, self.chi_obs, -vpara, vperp]

            t_obs = self.observation_times[tmeIdx]
            uB = (0.5 * stl.m_e * np.square(vperp)) / self.B0

            T, p_mu, p_chi, p_vel_mu, p_vel_chi = self.rk45_solver(t_span=[0, -t_obs], s0=s0, deltaT=t_obs, uB=uB)

            # Collect the mapped particle properties
            mapped_v_para = p_vel_mu[-1]
            mapped_mu = p_mu[-1]
            mapped_chi = p_chi[-1]
            mapped_alt = stl.Re*(SpatialClasses.r_muChi(mapped_mu,mapped_chi)-1)
            mapped_B_mag = self.B_dipole(mapped_mu, mapped_chi)
            mapped_v_perp = vperp * math.sqrt(mapped_B_mag / self.B0)
            mapped_pitch = math.atan(abs(mapped_v_para/mapped_v_perp))

            # Determine the local loss cone based off the equatorial loss cone
            mapped_loss_cone = math.asin(math.sin(self.pitch_eq_lost)*math.sqrt(mapped_B_mag/self.B_eq))

            if LiouvilleToggles.use_loss_cone_bool and mapped_pitch <= mapped_loss_cone: # check if within the loss cone. i.e. if alpha_obs <= alpha_los = arcsin(sqrt(B_obs/B_lower_boundary))
                block[ptchIdx][engyIdx] = 0
            else:
                block[ptchIdx][engyIdx] = self.Maxwellian(
                    vperp=mapped_v_perp,
                    vpara=-1 * mapped_v_para,
                    density=self.ne_density(mapped_mu, mapped_chi),
                    Te=self.Te(mapped_mu, mapped_chi),
                    Emin=10 ** LiouvilleToggles.E_min_obs,
                    Emax=10 ** LiouvilleToggles.E_max_obs,
                )
        return block

    def observed_fields(self):
        # Create the Interpolation Objects
        # Note: fill_value =0 means no wave field outside the simulted domain whereas fille_value =none extrapolates linearly
        interp_epara = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Epara, bounds_error=False, fill_value=0.0)
        interp_eperp = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Eperp, bounds_error=False, fill_value=0.0)
        interp_bperp = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Bperp, bounds_error=False, fill_value=0.0)

        # determine the observation mu-value
        r0 = 1 + self.mapping_alt / stl.Re
        colat0_rad = np.arcsin(np.sqrt(self.chi_obs * r0))
        u0_obs = - np.sqrt(np.cos(colat0_rad)) / r0

        # Calculate the interpolated wave-fields at the observation points
        T_end = LiouvilleToggles.time_obs_end + LiouvilleToggles.injected_wave_time_delay if LiouvilleToggles.injected_wave_time_delay > 0 else LiouvilleToggles.time_obs_end
        N_obs_wave_points = int(T_end / LiouvilleToggles.time_rez_waves)
        obs_waves_times = np.linspace(0, T_end, N_obs_wave_points)
        eval_points = np.array([[obs_waves_times[i], u0_obs] for i in range(N_obs_wave_points)])
        E_perp_obs = interp_eperp(eval_points)
        E_para_obs = interp_epara(eval_points)
        B_perp_obs = interp_bperp(eval_points)

        return E_para_obs, E_perp_obs,B_perp_obs, obs_waves_times


    def liouville_mapper(self):
        import multiprocessing as mp
        from tqdm import tqdm

        N_time = len(self.observation_times)
        N_ptch = len(LiouvilleToggles.pitch_range_obs)
        N_engy = len(LiouvilleToggles.energy_range_obs)
        Distribution = np.zeros((N_time, N_ptch, N_engy))

        with mp.Pool(processes=20, initializer=_init_worker, initargs=(self.mapping_alt,)) as pool:
            for tmeIdx, block in tqdm(pool.imap_unordered(_map_one_time, range(N_time)), total=N_time):
                Distribution[tmeIdx] = block

        return Distribution

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
        # DvmuDt_Alfven = - (stl.q0 / stl.m_e) * self.Epara(np.array([[deltaT + t, S[0]]]))[0]

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


