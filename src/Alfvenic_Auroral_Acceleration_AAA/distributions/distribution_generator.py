


def distribution_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    import math
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import ElectrostaticPotentialClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_classes import DistributionClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses2D as WaveFieldsClasses
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    #################################################
    # --- IMPORT THE PLASMA ENVIRONMENT FUNCTIONS ---
    #################################################
    envDict = ScaleLengthClasses().loadPickleFunctions()
    B_dipole = envDict['B_dipole']
    dB_dipole_dmu = envDict['dB_dipole_dmu']
    h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]

    # The
    def equations_of_motion(t, S, deltaT):
        # State Vector - [mu, chi, phi, vel_mu, vel_chi, vel_phi]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[3] / h_factors[0](S[0], S[1])

        # dchi/dt
        DchiDt = S[4] / h_factors[1](S[0], S[1])

        # dphi/dt
        DphiDt = S[5] / h_factors[2](S[0], S[1])

        # --- Velocity ---
        # dv_mu/dt

        # magnetic mirroring only
        # DvmuDt = - (uB/stl.m_e) * (dB_dipole_dmu(S[0],S[1])/h_factors[0](S[0],S[1]))

        # inverted-V only
        # DvmuDt = - (uB / stl.m_e) * (dB_dipole_dmu(S[0], S[1]) / h_factors[0](S[0], S[1])) - (stl.q0/stl.m_e)*ElectrostaticPotentialClasses().invertedVEField([S[0],S[1],S[2]])

        # wave fields + mirroring only
        DvmuDt = (- (uB / stl.m_e) * (dB_dipole_dmu(S[0], S[1]) / h_factors[0](S[0], S[1])) - (stl.q0 / stl.m_e) * WaveFieldsClasses().field_generator(time=t-deltaT, eval_pos=[S[0],S[1],S[2]],type='epara'))

        # dv_chi/dt
        DvchiDt = 0

        # dv_phi/dt
        DvphiDt = 0

        dS = [DmuDt, DchiDt, DphiDt, DvmuDt, DvchiDt, DvphiDt]

        return dS

    #####################
    # --- RK45 SOLVER ---
    #####################

    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(t, S, deltaT):

        alt = deepcopy(stl.Re*stl.m_to_km*(ScaleLengthClasses.r_muChi(S[0],DistributionToggles.chi0) - 1))

        # top boundary checker
        top_boundary_checker = alt - DistributionToggles.upper_termination_altitude

        return top_boundary_checker
        # return lower_boundary_checkerz

    escaped_upper.terminal = True

    def escaped_lower(t,S, deltaT):
        alt = deepcopy(stl.Re * stl.m_to_km * (ScaleLengthClasses.r_muChi(S[0], DistributionToggles.chi0) - 1))

        # lower boundary
        lower_boundary_checker = alt - DistributionToggles.lower_termination_altitude

        return lower_boundary_checker
    escaped_lower.terminal = True

    # --- Run the Solver and Plot it ---
    def my_RK45_solver(t_span, s0, deltaT):
        soln = solve_ivp(fun=equations_of_motion,
                         t_span=t_span,
                         y0=s0,
                         method=SimToggles.RK45_method,
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol,
                         t_eval=SimToggles.RK45_Teval,
                         events=(escaped_lower,escaped_upper),
                         args=deltaT
                         )
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        particle_phi = soln.y[2, :]
        vel_Mu = soln.y[3, :]
        vel_chi = soln.y[4, :]
        vel_phi = soln.y[5, :]
        # print(soln.message)
        return [T, particle_mu, particle_chi, particle_phi, vel_Mu, vel_chi, vel_phi]

    #########################
    # --- PREPARE OUTPUTS ---
    #########################

    # output
    Distribution = np.zeros(shape=(len(SimToggles.RK45_Teval),
                                   len(DistributionToggles.vel_space_mu_range),
                                   len(DistributionToggles.vel_space_perp_range)
                                   ))

    particle_mu_tracker = np.zeros(shape=(len(SimToggles.RK45_Teval),
                                          len(DistributionToggles.vel_space_mu_range),
                                          len(DistributionToggles.vel_space_perp_range)
                                          ))

    particle_mu_alt_tracker = np.zeros(shape=(len(SimToggles.RK45_Teval),
                                              len(DistributionToggles.vel_space_mu_range),
                                              len(DistributionToggles.vel_space_perp_range),
                                              ))

    particle_vel_mu_tracker = np.zeros(shape=(len(SimToggles.RK45_Teval),
                                              len(DistributionToggles.vel_space_mu_range),
                                              len(DistributionToggles.vel_space_perp_range)
                                              ))

    ########################################
    # --- LOOP OVER VELOCITY PHASE SPACE ---
    ########################################
    # for tmeIdx in tqdm(range(len(SimToggles.RK45_Teval))):
    for tmeIdx in [len(SimToggles.RK45_Teval)-1]:
        for paraIdx in tqdm(range(len(DistributionToggles.vel_space_mu_range))):
            for perpIdx in range(len(DistributionToggles.vel_space_perp_range)):
                v_perp0 = DistributionToggles.vel_space_perp_range[perpIdx]
                B0 = B_dipole(DistributionToggles.u0, DistributionToggles.chi0)
                uB = (0.5*stl.m_e*np.power(v_perp0,2))/B0
                s0 = [DistributionToggles.u0,
                      DistributionToggles.chi0,
                      DistributionToggles.phi0,
                      DistributionToggles.vel_space_mu_range[paraIdx],
                      DistributionToggles.vel_space_perp_range[perpIdx],
                      DistributionToggles.vel_space_perp_range[perpIdx]]
                deltaT = tuple([SimToggles.RK45_Teval[tmeIdx]])

                [T, particle_mu, particle_chi, particle_phi, particle_vel_Mu, particle_vel_chi, particle_vel_phi] = my_RK45_solver(SimToggles.RK45_tspan, s0, deltaT)

                ################################
                # --- PERPENDICULAR DYNAMICS ---
                ################################
                # geomagnetic field experienced by particle
                B_mag_particle = B_dipole(deepcopy(particle_mu),np.array([DistributionToggles.chi0 for i in range(len(particle_mu))]))
                particle_vel_perp = v_perp0*np.sqrt(B_mag_particle/np.array([B0 for i in range(len(B_mag_particle))]))
                # pitch_angle = 180-np.degrees(np.arctan2(deepcopy(particle_vel_perp),deepcopy(particle_vel_Mu)))

                ################
                # --- ENERGY ---
                ################
                # Energy = (0.5*stl.m_e*(np.square(deepcopy(particle_vel_perp)) + np.square(deepcopy(particle_vel_Mu))))/stl.q0

                ####################################################
                # --- UPDATE DISTRIBUTION GRID AT simulation END ---
                ####################################################
                Distribution[tmeIdx][paraIdx][perpIdx] = DistributionClasses().Maxwellian(n=DistributionToggles.n_PS,
                                                                       Te=DistributionToggles.Te_PS,
                                                                       vel_perp=deepcopy(particle_vel_perp[-1]),
                                                                       vel_para=deepcopy(particle_vel_Mu[-1]))
                ##########################
                # --- Particle Tracker ---
                ##########################
                # fig, ax = plt.subplots(5,sharex=True)
                # fig.set_figwidth(10)
                # fig.set_figheight(15)
                # fig.suptitle(f'T0 = {SimToggles.RK45_Teval[tmeIdx]} seconds\n'+
                #              '$v_{\perp,0}$=' + f'{round(0.5*stl.m_e*np.power(particle_vel_perp[0],2)/stl.q0)} eV\n'+
                #              '$v_{\parallel,0}$=' + f'{math.copysign(round(0.5*stl.m_e*np.power(particle_vel_Mu[0],2)/stl.q0),particle_vel_Mu[0])} eV\n' +
                #              f'V0 = {WaveFieldsToggles.inV_Volts}V')
                # ax[0].plot(T, particle_mu)
                # ax[0].set_ylabel('$\mu$')
                # ax[1].plot(T,stl.Re*(np.array(ScaleLengthClasses.r_muChi(particle_mu,np.array([DistributionToggles.chi0 for i in range(len(particle_mu))])))-1))
                # ax[1].set_ylabel('alt [km]')
                # ax[1].set_ylim(-250, 10000)
                # ax[1].axhline(y=WaveFieldsToggles.inV_Zmin,color='red')
                # ax[1].axhline(y=WaveFieldsToggles.inV_Zmax,color='red')
                # ax[1].axhline(y=DistributionToggles.lower_termination_altitude/stl.m_to_km,color='blue',linestyle='--')
                #
                # ax[2].plot(T, particle_vel_Mu/(stl.m_to_km*1E5))
                # ax[2].set_ylabel('$v_{\mu}$ [10,000 km/s]')
                #
                # ax[3].plot(T,pitch_angle)
                # ax[3].set_ylabel('Pitch Angle [deg]')
                #
                # ax[4].plot(T, Energy)
                # ax[4].set_xlabel('Time [s] (Backward Propogated)')
                # ax[4].set_ylabel('Energy [eV]')
                #
                # for i in range(5):
                #     ax[i].grid(True)
                #     ax[i].ticklabel_format(style='plain') # remove scientific notation
                #
                # plt.gca().invert_xaxis()
                # plt.tight_layout()
                #
                # fig.savefig(f'{SimToggles.sim_data_output_path}/distributions/plots/plot{paraIdx}{perpIdx}.png')


    ################
    # --- OUTPUT ---
    ################
    data_dict_output = {
        'time': [np.array(T), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'time_eval' : [np.array(SimToggles.RK45_Teval),{'UNITS': 's', 'LABLAXIS': 'Time Eval','VAR_TYPE':'data'}],
        'particle_mu': [np.array(particle_mu_tracker), {'DEPEND_0':'time_eval','DEPEND_1':'vperp_range','DEPEND_2':'vpara_range','UNITS': None, 'LABLAXIS': '&mu;', 'VAR_TYPE':'data'}],
        'particle_vel_mu' : [np.array(particle_vel_mu_tracker), {'DEPEND_0':'time_eval','DEPEND_1':'vperp_range','DEPEND_2':'vpara_range','UNITS': 'm/s', 'LABLAXIS': 'V!B&mu;!N', 'VAR_TYPE':'data'}],
        'particle_alt_along_mu': [np.array(particle_mu_alt_tracker), {'DEPEND_0':'time_eval','DEPEND_1':'vperp_range','DEPEND_2':'vpara_range', 'UNITS': 'km', 'LABLAXIS': '&mu;!N', 'VAR_TYPE': 'data'}],
        'particle_vel_perp': [np.array(particle_vel_perp), {'DEPEND_0': 'time', 'UNITS': 'm/s', 'LABLAXIS': 'V!B&perp;!N', 'VAR_TYPE': 'data'}],
        'Distribution': [np.array(Distribution), {'DEPEND_0':'time_eval','DEPEND_2':'vperp_range','DEPEND_1':'vpara_range','UNITS':'m!A-6!Ns!A-3!N','LABLAXIS':'Distribution Function','VAR_TYPE':'data'}],
        'vperp_range': [np.array(DistributionToggles.vel_space_perp_range), {'UNITS':'m/s','LABLAXIS':'V!B&perp;!N'}],
        'vpara_range': [np.array(DistributionToggles.vel_space_mu_range), {'UNITS':'m/s','LABLAXIS':'V!B&parallel;!N'}],
    }

    outputPath = rf'{DistributionToggles.outputFolder}/distributions.cdf'
    stl.outputDataDict(outputPath, data_dict_output)