# Simulation Imports
import scipy.integrate
from scipy.special import gamma
import numpy as np
import spaceToolsLib as stl
from itertools import product
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
from src.Alfvenic_Auroral_Acceleration_AAA.plasma_environment.plasma_environment_classes import PlasmaEnvironmentClasses
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import LiouvilleToggles,PlasmaEnvironmentToggles

_WORKER = {}

def _init_worker(mapping_alt):
    _WORKER['obj'] = LiouvilleMapping(mapping_alt)

def _map_one_time(tmeIdx):
    return tmeIdx, _WORKER['obj'].map_single_time(tmeIdx)

class LiouvilleMapping:

    def __init__(self,mapping_alt):
        # form the regular grid interpolator for E-parallel
        data_dict_potentials = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')
        data_dict_spatial = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/spatial_grid/spatial_grid.cdf')

        # Define the Simulation Boundary Info
        self.mu_top = data_dict_spatial['mu'][0][-1]
        self.mu_bot = data_dict_spatial['mu'][0][0]

        # Calculate the Observation Info
        self.mapping_alt = mapping_alt
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]
        self.chi_obs = data_dict_spatial['chi'][0][0]
        self.r_obs = 1 + self.mapping_alt / stl.Re
        self.colat_obs_rad = np.arcsin(np.sqrt(self.chi_obs * self.r_obs))
        self.mu_obs = - np.sqrt(np.cos(self.colat_obs_rad)) / self.r_obs
        self.B0 = self.B_dipole(self.mu_obs, self.chi_obs)
        self.observation_times = np.linspace(LiouvilleToggles.time_obs_start, LiouvilleToggles.time_obs_end, LiouvilleToggles.N_obs_points)  # list of observation times

        # Construct the Wave Interpolator Object
        self.mu_grid = data_dict_spatial['mu'][0]
        self.time_grid = data_dict_potentials['time'][0]
        self.Emu = data_dict_potentials['E_mu'][0].copy()
        self.Eperp = data_dict_potentials['E_perp'][0].copy()
        self.Bperp = data_dict_potentials['B_perp'][0].copy()

        # Construct a Plasma Environment Class
        self.plasma_environment_object = PlasmaEnvironmentClasses()

        # Delay the wave injection
        if LiouvilleToggles.injected_wave_time_delay > 0:
            # --- Adjust the wave Interpolator ---
            deltaT_time = np.gradient(self.time_grid)[0]
            N_additional_points = int(LiouvilleToggles.injected_wave_time_delay / deltaT_time)
            zeros = np.zeros((N_additional_points, self.Emu.shape[1]), dtype=self.Emu.dtype)

            # adjust the fields size
            self.Emu = np.vstack([zeros, self.Emu])
            self.Eperp = np.vstack([zeros, self.Eperp])
            self.Bperp = np.vstack([zeros, self.Bperp])

            # adjust the fields time grid size
            self.time_grid = np.concatenate([np.array([deltaT_time * i for i in range(N_additional_points)]), self.time_grid + LiouvilleToggles.injected_wave_time_delay])

            # --- Adjust the observation times ---
            deltaT_obs = np.gradient(self.observation_times)[0]
            N_additional_obs_points = int(LiouvilleToggles.injected_wave_time_delay / deltaT_obs)
            self.observation_times = np.concatenate([np.array([deltaT_obs * i for i in range(N_additional_obs_points)]), self.observation_times + LiouvilleToggles.injected_wave_time_delay])

        self.Emu_interp = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Emu, bounds_error=False, fill_value=0.0)


        # Construct the inverted-V distribution info
        if LiouvilleToggles.use_invertedV_bool:

            # create an interpolation object between altitude and mu
            mu_interp = self.make_1D_interp_obj(data_dict_spatial['S_para'][0],data_dict_spatial['mu'][0])

            # Calculate inverted-V parameters (use a hyperbolic function)
            self.h = (LiouvilleToggles.inV_S_top + LiouvilleToggles.inV_S_bot)/2
            self.a = 1 / (0.1*np.power(10, math.floor(math.log10(abs(LiouvilleToggles.inV_potential)))))  # [Fractional Percent] Controls the edge-value of the inverted-V potential
            self.q = 2*(LiouvilleToggles.inV_S_bot - self.h)/(np.log((1 + np.sqrt(1-self.a))/(1 - np.sqrt(1-self.a))))
            self.E0 = (LiouvilleToggles.inV_potential/self.q) * (1/(np.tanh((self.h-LiouvilleToggles.inV_S_top)/self.q) - np.tanh((self.h-LiouvilleToggles.inV_S_bot)/self.q)))
            self.E_inV = self.E0*(1- np.square(np.tanh((data_dict_spatial['S_para'][0]/stl.m_to_km-self.h)/self.q)))/stl.m_to_km # [V/m] the /m_to_km is to convert to V/m instead of V/km

            S_para = data_dict_spatial['S_para'][0].copy()
            mask = (S_para < LiouvilleToggles.inV_S_bot*stl.m_to_km) & (S_para > LiouvilleToggles.inV_S_top*stl.m_to_km)
            self.E_inV[mask] = 0 # clamp any values outside the inverted-V

            # print(scipy.integrate.simpson(self.E_inV,x=S_para)) # verify the electrostatic potential is == inV_potential
            # fig, ax =plt.subplots()
            # ax.plot(1+S_para/(stl.Re*stl.m_to_km),self.E_inV*1000)
            # ax.set_ylabel('Ez [mV/m]')
            # plt.show()

            self.inV_E_interp = self.make_1D_interp_obj(data_dict_spatial['mu'][0].copy(),self.E_inV)

    def make_1D_interp_obj(self,x,y,left=0,right=0):
        # Creates an interpolation object for 1D arrays
        return lambda x_new: np.interp(x_new, x, y, left=0, right=0)



    def liouville_mapper(self):

        N_time = len(self.observation_times)
        N_ptch = len(LiouvilleToggles.pitch_range_obs)
        N_engy = len(LiouvilleToggles.energy_range_obs)
        Distribution = np.zeros((N_time, N_ptch, N_engy))
        # particle_traj_mu = np.zeros((N_time, N_ptch, N_engy, N_time))

        with mp.Pool(processes=LiouvilleToggles.processes_count, initializer=_init_worker, initargs=(self.mapping_alt,)) as pool:
            for tmeIdx, block in tqdm(pool.imap_unordered(_map_one_time, range(N_time)), total=N_time):
                Distribution[tmeIdx] = block
        return Distribution

    def map_single_time(self, tmeIdx):
        N_ptch = len(LiouvilleToggles.pitch_range_obs)
        N_engy = len(LiouvilleToggles.energy_range_obs)
        N_time = len(self.observation_times)
        block = np.zeros((N_ptch, N_engy))
        # particle_mus = np.zeros((N_ptch, N_engy, N_time))

        for ptchIdx, engyIdx in product(range(N_ptch), range(N_engy)):
            engyVal = LiouvilleToggles.energy_range_obs[engyIdx]
            ptchVal = np.radians(LiouvilleToggles.pitch_range_obs[ptchIdx])
            speed = np.sqrt(2 * stl.q0 * engyVal / stl.m_e)
            vperp = speed * np.sin(ptchVal)
            v_mu = -1*speed * np.cos(ptchVal)
            uB = (0.5 * stl.m_e * np.square(vperp)) / self.B0

            s0 = [self.mu_obs, self.chi_obs, v_mu, vperp]  # the -1 on vpara is to convert to modified dipole coordinates
            t_obs = self.observation_times[tmeIdx]

            T, p_mu, p_chi, p_vel_mu, p_vel_chi = self.rk45_solver(t_span=[0, -t_obs], s0=s0, deltaT=t_obs, uB=uB)

            # Collect the mapped particle properties
            mapped_mu = p_mu[-1]
            mapped_chi = p_chi[-1]
            mapped_v_mu = p_vel_mu[-1]
            mapped_B_mag = self.B_dipole(mapped_mu, mapped_chi)
            mapped_v_perp = vperp * math.sqrt(mapped_B_mag / self.B0)

            # --- MODIFY/EXPORT DISTRIBUTIONS ---
            E_src = 0.5 * stl.m_e * (mapped_v_perp**2 + mapped_v_mu**2)  # [J] kinetic energy at the end point
            block[ptchIdx][engyIdx] = self.f_plasma_sheet(E_src, uB) + self.f_cold(mapped_mu, mapped_chi, E_src)


        return block

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
        DvmuDt = 0

        # magnetic mirroring - the sign on this is CONFIRMED correct! needs the negative sign to ensure the force points in positive mu-direction
        DvmuDt += -(uB/stl.m_e) * (self.dB_dipole_dmu(S[0],S[1])/self.h_factors[0](S[0],S[1])) if LiouvilleToggles.use_magnetic_mirror_bool else 0

        # EM Field - the sign on this is CONFIRMED correct! needs the negative sign due to the electron charge
        DvmuDt +=  - (stl.q0 / stl.m_e) * self.Emu_interp(np.array([[deltaT + t, S[0]]]))[0] if LiouvilleToggles.use_alfven_wave_bool else 0

        # inverted-V - the sign is correct: needs a negative due to the electron charge
        DvmuDt += -(stl.q0 / stl.m_e) * self.inV_E_interp(S[0]) if LiouvilleToggles.use_invertedV_bool else 0

        # dv_chi/dt
        DvchiDt = 0

        return [DmuDt, DchiDt, DvmuDt, DvchiDt]

    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(self, t, S, deltaT, uB):

        # top boundary checker
        top_boundary_checker = S[0] - self.mu_top

        return top_boundary_checker
    escaped_upper.terminal = True

    def escaped_lower(self, t, S, deltaT, uB):

        # lower boundary
        lower_boundary_checker =S[0]-self.plasma_environment_object.mu_lost # TODO: Think about this more

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

    def observed_fields(self):
        # Create the Interpolation Objects
        # Note: fill_value =0 means no wave field outside the simulted domain whereas fille_value =none extrapolates linearly
        interp_emu = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Emu, bounds_error=False, fill_value=0.0)
        interp_eperp = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Eperp, bounds_error=False, fill_value=0.0)
        interp_bperp = RegularGridInterpolator((self.time_grid, self.mu_grid), self.Bperp, bounds_error=False, fill_value=0.0)

        # Calculate the interpolated wave-fields at the observation points
        T_end = LiouvilleToggles.time_obs_end + LiouvilleToggles.injected_wave_time_delay if LiouvilleToggles.injected_wave_time_delay > 0 else LiouvilleToggles.time_obs_end
        N_obs_wave_points = int(T_end / LiouvilleToggles.time_rez_waves)
        obs_waves_times = np.linspace(0, T_end, N_obs_wave_points)
        eval_points = np.array([[obs_waves_times[i], self.mu_obs] for i in range(N_obs_wave_points)])

        E_perp_obs = interp_eperp(eval_points)
        E_mu_obs = interp_emu(eval_points)
        B_perp_obs = interp_bperp(eval_points)

        return E_mu_obs, E_perp_obs,B_perp_obs, obs_waves_times

    def Kappa(self, mass, Vperp,Vpara,charge ,n, Te, vpara, vperp, kappa):
        # Input: density [cm^-3], Temperature [eV], Velocities [m/s]
        # output: the distribution function in SI units [s^3 m^-6]
        Emag = (0.5 * mass * (Vperp ** 2 + Vpara ** 2)) / charge
        Ek = Te * (1 - 3 / (2 * kappa))
        return (1E6) * n * np.power(mass / (2 * np.pi * kappa * stl.q0 * Ek), 3 / 2) * (gamma(kappa + 1) / gamma(kappa - 0.5)) * np.power(1 + Emag / (kappa * Ek), -(kappa + 1))

    def maxwellian(self,E_eV, n0, Te):
        return n0 * np.power(stl.m_e / (2 * np.pi * Te * stl.q0), 1.5) * np.exp(-E_eV/ Te)

    def dist_secondaries(self, E_eV):
        # calculated using the chen data for JE, specifically a constant value of JE
        jE_const =  1E9 #[eV-cm^-2-s^-1-st^-1-eV^-1]

        # Return the distribution function in [s^3 m^-6]
        dist_nonSI = (stl.m_e**2)/(2*(E_eV**2))*jE_const
        return  dist_nonSI*(stl.cm_to_m/stl.q0)**2

    def dist_backscatter(self, E_eV, n0, Te):
        # Backscatter curve - just the same maxwellian as the primary beam, but in the opposite direction
        dist = n0 * np.power(stl.m_e / (2 * np.pi * Te * stl.q0), 1.5) * np.exp(-(E_eV) / Te)
        return dist

    def inV_dist(self, mu, chi, E_src, uB):

        # Plasma parameters
        Te = PlasmaEnvironmentToggles.Te_PS
        E_eV = E_src / stl.q0
        density_val = (stl.cm_to_m**3)*PlasmaEnvironmentToggles.n0_PS

        # if you're below the inverted-V, use an accelerated Maxwellian distribution with secondaries/backscatter
        if mu <= self.inV_mu_bot:

            # check if energy is above or equal to accelerating potential
            check = E_eV-self.inV_potential

            # The primary beam
            accel_max = self.maxwellian(check,density_val,Te) if check>=0 else 0

            # secondaries curve
            secondaries = self.dist_secondaries(E_eV) if check < 0 else 0

            # backscatter curve
            backscatter = self.dist_backscatter(check, density_val,Te) if check < 0 else 0

            return accel_max + secondaries + backscatter

        # if you're within the inverted-V
        elif mu >= self.inV_mu_bot & mu <= self.inV_mu_top:  # if particle is within the inverted-V potential
            check = E_eV

        # if you're above the inverted-V (use plasma sheet)
        elif mu > self.inV_mu_top:
            return self.f_plasma_sheet(mu,chi,E_src,uB)


    def f_plasma_sheet(self, E_src, uB):
        # E_src: kinetic energy at the end of the backward trace [J]; uB: magnetic moment [J/T], conserved.
        # The electron mirrors where E = uB*B, so it reaches the exobase (a loss cone, either direction)
        # exactly when E_src > uB*B_loss. Outside the cones, f is the FULL Maxwellian normalized to n0.

        # Get the plasma sheet parameters
        if PlasmaEnvironmentToggles.use_loss_cone_bool:
            if E_src > uB * self.plasma_environment_object.B_lost: # the particle has mirrored
                return 0.0

        E_eV = E_src / stl.q0
        if not (PlasmaEnvironmentToggles.Emin_PS <= E_eV <= PlasmaEnvironmentToggles.Emax_PS): # the prticle is outside the range of the distribution
            return 0.0
        Te = PlasmaEnvironmentToggles.Te_PS

        if PlasmaEnvironmentToggles.use_loss_cone_bool:
            density_val = self.plasma_environment_object.n0_PS_norm
        else:
            density_val = (stl.cm_to_m**3)*PlasmaEnvironmentToggles.n0_PS
        return density_val * np.power(stl.m_e / (2 * np.pi * Te * stl.q0),1.5) * np.exp(-E_eV / Te)

    def f_cold(self, mu, chi, E_src):
        # cold isotropic population: local Maxwellian with the model density at the end point
        E_eV = E_src / stl.q0
        if not (PlasmaEnvironmentToggles.Emin_cold <= E_eV <= PlasmaEnvironmentToggles.Emax_cold):
            return 0.0
        Te = PlasmaEnvironmentToggles.Te_cold
        return envDict['n_density_cold'](mu, chi) * np.power(stl.m_e / (2*np.pi*Te*stl.q0), 1.5) * np.exp(-E_eV / Te)





