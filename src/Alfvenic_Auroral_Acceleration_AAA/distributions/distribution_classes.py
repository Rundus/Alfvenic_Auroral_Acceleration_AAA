import numpy as np
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.distributions.distribution_toggles import DistributionToggles
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import ElectrostaticPotentialClasses
from scipy.integrate import solve_ivp
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses2D as WaveFieldsClasses
envDict = ScaleLengthClasses().loadPickleFunctions()

class DistributionClasses:

    def __init__(self):
        self.B_dipole = envDict['B_dipole']
        self.dB_dipole_dmu = envDict['dB_dipole_dmu']
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]

    # The
    def equations_of_motion(self, t, S, deltaT, uB):
        # State Vector - [mu, chi, vel_mu, vel_chi]

        # --- Coordinates ---
        # dmu/dt
        DmuDt = S[2] / self.h_factors[0](S[0], S[1])

        # dchi/dt
        DchiDt = S[3] / self.h_factors[1](S[0], S[1])

        # --- Velocity ---
        # dv_mu/dt

        # magnetic mirroring only
        DvmuDt_mirror = - (uB/stl.m_e) * (self.dB_dipole_dmu(S[0],S[1])/self.h_factors[0](S[0],S[1]))

        # inverted-V only
        # DvmuDt_inV = (stl.q0/stl.m_e)*ElectrostaticPotentialClasses().invertedVEField([S[0],S[1],S[2]])

        # wave fields + mirroring only
        DvmuDt_Alfven =  - (stl.q0 / stl.m_e) * WaveFieldsClasses().field_generator(time=SimToggles.RK45_tspan[1] -deltaT-t,
                                                                                    eval_pos=[S[0],S[1]],
                                                                                    type='epara')
        DvmuDt = DvmuDt_mirror + DvmuDt_Alfven

        # dv_chi/dt
        DvchiDt = 0

        return [DmuDt, DchiDt, DvmuDt, DvchiDt]


    # An event is a function where the RK45 method determines event(t,y)=0
    def escaped_upper(self, t, S, deltaT, uB):

        alt = stl.Re*stl.m_to_km*(ScaleLengthClasses.r_muChi(S[0],SimToggles.chi0) - 1)

        # top boundary checker
        top_boundary_checker = alt - DistributionToggles.upper_termination_altitude

        return top_boundary_checker
        # return lower_boundary_checkerz

    escaped_upper.terminal = True

    def escaped_lower(self, t, S, deltaT, uB):
        alt = stl.Re * stl.m_to_km * (ScaleLengthClasses.r_muChi(S[0], SimToggles.chi0) - 1)

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
                         method=SimToggles.RK45_method,
                         rtol=SimToggles.RK45_rtol,
                         atol=SimToggles.RK45_atol,
                         # t_eval=SimToggles.RK45_Teval,
                         events=(self.escaped_lower,self.escaped_upper),
                         args=tuple([deltaT,uB])
                         )
        T = soln.t
        particle_mu = soln.y[0, :]
        particle_chi = soln.y[1, :]
        vel_Mu = soln.y[2, :]
        vel_chi = soln.y[3, :]
        # print(soln.message)
        return [T, particle_mu, particle_chi, vel_Mu, vel_chi]



    def Maxwellian(self, n, Te, vel_para, vel_perp): # returns the maxwellian distribution for a given temperature, density and particle velocity
        """
        :param n: Plasma Density in [m^-3]
        :type n : float

        :param Te: Plasma Temperature in [eV]
        :type Te: float

        :param vel_para: Particle Velocity parallel to the background geomagnetic field in [m/s]
        :type vel_para: float

        :param vel_perp: Particle Velocity parallel to the background geomagnetic field in [m/s]
        :type vel_perp: float

        :return: Plasma Distribution Function in [m^-6 s^-3] evaluated at vel_para, vel_perp
        """

        return n*np.sqrt(np.power(stl.m_e/(2*np.pi*Te*stl.q0),3)) * np.exp(-0.5*stl.m_e*(np.square(vel_perp) + np.square(vel_para))/(stl.q0*Te))


