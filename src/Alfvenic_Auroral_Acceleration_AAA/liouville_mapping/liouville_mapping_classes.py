# Simulation Imports
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
from scipy.integrate import solve_ivp
from Alfvenic_Auroral_Acceleration_AAA.archive.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses

# General Imports
import numpy as np
import spaceToolsLib as stl
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()


class LiouvilleClasses:

    def __init__(self):
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]


    def liouville_mapping(self):

        def mapper(self, tmeIdx):
            import multiprocessing as mp

            #################################################
            # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
            #################################################
            envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
            B_dipole = envDict['B_dipole']

            # --- PREPARE PARALLELIZED OUTPUTS ---
            Ntimes = len(LiouvilleToggles.obs_times)
            Nptchs = len(LiouvilleToggles.pitch_range_obs)
            Nengy = len(LiouvilleToggles.energy_range_obs)
            sizes = [Ntimes, Nptchs, Nengy]

            # distribution
            mp_array_1 = mp.Array('d', Ntimes * Nptchs * Nengy)
            arr_1 = np.frombuffer(mp_array_1.get_obj())
            Distribution = arr_1.reshape((Ntimes, Nptchs, Nengy))

            B0 = B_dipole(LiouvilleToggles.u0_obs, LiouvilleToggles.chi0_obs)

            for ptchIdx, engyIdx in product(*[range(Nptchs), range(Nengy)]):
                # get the initial state vector
                engyVal = LiouvilleToggles.energy_range_obs[engyIdx]
                ptchVal = np.radians(LiouvilleToggles.pitch_range_obs[ptchIdx])
                v_perp0 = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.sin(ptchVal)
                v_para0 = np.sqrt(2 * stl.q0 * engyVal / stl.m_e) * np.cos(ptchVal)
                v_mu0 = -1 * v_para0

                s0 = [LiouvilleToggles.u0_obs, LiouvilleToggles.chi0_obs, v_mu0, v_perp0]

                # get the solver arguments
                deltaT = LiouvilleToggles.obs_times[tmeIdx]
                uB = (0.5 * stl.m_e * np.square(v_perp0)) / B0

                # Perform the RK45 Solver
                [T, particle_mu, particle_chi, particle_vel_Mu, particle_vel_chi] = LiouvilleClasses().louivilleMapper(
                    [0, -deltaT], s0, deltaT, uB)

                ################################
                # --- PERPENDICULAR DYNAMICS ---
                ################################
                # geomagnetic field experienced by particle
                B_mag_particle = B_dipole(deepcopy(particle_mu), deepcopy(particle_chi))
                mapped_v_perp = v_perp0 * np.sqrt(B_mag_particle / np.array([B0 for i in range(len(B_mag_particle))]))

                ####################################################
                # --- UPDATE DISTRIBUTION GRID AT runners END ---
                ####################################################

                # if deltaT == 4.16:
                #     # print(f'Energy: {engyVal} ', f' Pitch: {np.degrees(ptchVal)} ',list(particle_mu), list(particle_vel_Mu))
                #     fig, ax =plt.subplots(6, sharex=True)
                #     fig.set_figwidth(8)
                #     fig.set_figheight(12)
                #     fig.suptitle('$z_{obs}$=' + f'{LiouvilleToggles.z0_obs} km'+f'\nT(obs)={deltaT} s, Energy (obs): {round(engyVal)} eV, Pitch (obs): {np.degrees(ptchVal)} deg')
                #     particle_alt = stl.Re * (SimClasses.r_muChi(particle_mu, particle_chi) - 1)
                #     ax[0].plot(T+deltaT,particle_alt )
                #     ax[0].set_ylabel('Alt [km]')
                #
                #     ax[1].plot(T+deltaT,particle_vel_Mu/stl.m_to_km)
                #     ax[1].set_ylabel('$v_{\mu}$ [km/s]')
                #
                #     ax[2].plot(T+deltaT, np.degrees(np.arctan2(mapped_v_perp,-1*particle_vel_Mu)) )
                #     ax[2].set_ylabel(r'$\alpha$ [deg]')
                #     ax[2].axhline(y=90,color='tab:red',alpha=0.5,linestyle='--')
                #
                #     ax[3].plot(T+deltaT,0.5*(stl.m_e/stl.q0)*(np.square(particle_vel_Mu) + np.square(mapped_v_perp)))
                #     ax[3].set_ylabel('Energy [eV]')
                #
                #     particle_pos = np.array([particle_mu,particle_chi,[RayEquationToggles.phi0_w for i in range(len(T))]]).T
                #     E_mu_particle = np.array([WaveFieldsClasses().field_generator(deltaT+tme, pos, type='emu') for tme,pos in zip(T,particle_pos)])
                #     ax[4].plot(T+deltaT, E_mu_particle)
                #     ax[4].set_ylabel('$E_{\mu}$ particle')
                #
                #     wave_value = np.zeros(shape=(len(T),len(WaveFieldsToggles.mu_grid)))
                #     eval_pos = [[WaveFieldsToggles.mu_grid[idx], RayEquationToggles.chi0_w, RayEquationToggles.phi0_w] for idx in range(len(WaveFieldsToggles.mu_grid))]
                #     for idx,tmeVal in enumerate(T):
                #         wave_value[idx] = np.array([WaveFieldsClasses().field_generator(deltaT+tmeVal, pos, type='emu') for pos in eval_pos])
                #
                #     alts = np.array([stl.Re*(SimClasses.r_muChi(mu,RayEquationToggles.chi0_w)-1) for mu in WaveFieldsToggles.mu_grid])
                #     ax[5].pcolormesh(deltaT+T, alts, wave_value.T, cmap='bwr',vmin=-1E-5,vmax=1E-5)
                #     ax[5].set_xlabel('Time [s]')
                #     ax[5].set_ylabel('alt [km]')
                #     ax[5].plot(deltaT + T, particle_alt, 'ro')
                #
                #     for i in range(6):
                #         ax[i].grid(True)
                #         ax[i].invert_xaxis()
                #
                #     fig.tight_layout()
                #     fig.savefig(f'/home/connor/Data/physicsModels/alfvenic_auroral_acceleration_AAA/flux/plots/particle_trajects/traject_T{tmeIdx}_ptch{ptchIdx}_engy{engyIdx}.png')
                #     # plt.show()

                Distribution[tmeIdx][ptchIdx][engyIdx] = LiouvilleClasses().mapped_distribution(mu=particle_mu[-1],
                                                                                                chi=particle_chi[-1],
                                                                                                vel_perp=mapped_v_perp[
                                                                                                    -1],
                                                                                                vel_para=-1 *
                                                                                                         particle_vel_Mu[
                                                                                                             -1])

        processes_count = 20  # Number of CPU cores to commit to this operation
        pool_object = mp.Pool(processes_count)
        inputs = range(Ntimes)
        for _ in tqdm(pool_object.imap_unordered(louisville_mapping, inputs), total=Ntimes):
            pass

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
        DvmuDt_Alfven =  - (stl.q0 / stl.m_e) * WaveFieldsClasses().field_generator(time=deltaT + t,
                                                                                    eval_pos=[S[0],S[1]],
                                                                                    type='eMu')
        # Combine all the parallel effects together
        # DvmuDt = DvmuDt_mirror
        DvmuDt = DvmuDt_mirror + DvmuDt_Alfven


        # dv_chi/dt
        DvchiDt = 0

        return [DmuDt, DchiDt, DvmuDt, DvchiDt]

    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(self, t, S, deltaT, uB):

        alt = stl.Re*(SimClasses.r_muChi(S[0],S[1]) - 1)

        # top boundary checker
        top_boundary_checker = alt - DistributionToggles.upper_termination_altitude

        return top_boundary_checker

    escaped_upper.terminal = True

    def escaped_lower(self, t, S, deltaT, uB):
        alt = stl.Re  * (SimClasses.r_muChi(S[0], DistributionToggles.chi0_obs) - 1)

        # lower boundary
        lower_boundary_checker = alt - DistributionToggles.lower_termination_altitude

        return lower_boundary_checker
    escaped_lower.terminal = True

    #####################
    # --- RK45 SOLVER ---
    #####################
    def louivilleMapper(self, t_span, s0, deltaT, uB):
        soln = solve_ivp(fun=self.equations_of_motion,
                         t_span=t_span,
                         y0=s0,
                         method=DistributionToggles.RK45_method,
                         rtol=DistributionToggles.RK45_rtol,
                         atol=DistributionToggles.RK45_atol,
                         events=[self.escaped_lower, self.escaped_upper],
                         args=tuple([deltaT, uB])
                         )
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        vel_Mu = soln.y[2, :]
        vel_chi = soln.y[3, :]
        return [T, particle_mu, particle_chi, vel_Mu, vel_chi]

    def mapped_distribution(self, mu, chi,vel_para, vel_perp):

        # Determine if particle triggered a termination event
        particle_alt = stl.Re  * (SimClasses.r_muChi(mu, chi) - 1)

        # if particle_alt <= DistributionToggles.lower_termination_altitude: # if trajectory was forbidden due to ionospheric collisional loss
        #     return 0
        # else:
        return self.Maxwellian_PS(vel_para, vel_perp)

    def Maxwellian_PS(self, vel_para, vel_perp): # returns the maxwellian distribution for a given temperature, density and particle velocity
        """
        :param vel_para: Particle Velocity parallel to the background geomagnetic field in [m/s]
        :type vel_para: float

        :param vel_perp: Particle Velocity parallel to the background geomagnetic field in [m/s]
        :type vel_perp: float

        :return: Plasma Distribution Function in [m^-6 s^-3] evaluated at vel_para, vel_perp
        """
        # if 0.5*(stl.m_e/stl.q0)*(np.square(vel_para) + np.square(vel_perp)) > DistributionToggles.Emax_PS: # check if energy is above the specific level the distribution
        #     return 0
        # elif 0.5*(stl.m_e/stl.q0)*(np.square(vel_para) + np.square(vel_perp)) < DistributionToggles.Emin_PS: # check if energy is below the specific level the distribution:
        #     return 0
        # else:
        return DistributionToggles.n_PS*np.sqrt(np.power(stl.m_e/(2*np.pi*DistributionToggles.Te_PS*stl.q0),3)) * np.exp(-0.5*stl.m_e*(np.square(vel_perp) + np.square(vel_para))/(stl.q0*DistributionToggles.Te_PS))

    def Maxwellian_iono(self, vel_para, vel_perp):
        """
                :param vel_para: Particle Velocity parallel to the background geomagnetic field in [m/s]
                :type vel_para: float

                :param vel_perp: Particle Velocity parallel to the background geomagnetic field in [m/s]
                :type vel_perp: float

                :return: Plasma Distribution Function in [m^-6 s^-3] evaluated at vel_para, vel_perp
                """
        if 0.5 * (stl.m_e / stl.q0) * (np.square(vel_para) + np.square(vel_perp)) > DistributionToggles.Emax_iono:  # check if energy is above the specific level the distribution
            return 0
        elif 0.5 * (stl.m_e / stl.q0) * (np.square(vel_para) + np.square(vel_perp)) < DistributionToggles.Emin_iono:  # check if energy is below the specific level the distribution:
            return 0
        else:
            return DistributionToggles.n_iono * np.sqrt(np.power(stl.m_e / (2 * np.pi * DistributionToggles.Te_iono * stl.q0), 3)) * np.exp(-0.5 * stl.m_e * (np.square(vel_perp) + np.square(vel_para)) / (stl.q0 * DistributionToggles.Te_iono))

    # def Kappa(self, n, Te, vel_para, vel_perp, kappa):
    #     # Input: density [cm^-3], Temperature [eV], Velocities [m/s]
    #     # output: the distribution function in SI units [s^3 m^-6]
    #     Emag = (0.5 * mass * (Vperp ** 2 + Vpara ** 2)) / charge
    #     Ek = T * (1 - 3 / (2 * kappa))
    #     return (1E6) * n * np.power(mass / (2 * np.pi * kappa * stl.q0 * Ek), 3 / 2) * (gamma(kappa + 1) / gamma(kappa - 0.5)) * np.power(1 + Emag / (kappa * Ek), -(kappa + 1))


