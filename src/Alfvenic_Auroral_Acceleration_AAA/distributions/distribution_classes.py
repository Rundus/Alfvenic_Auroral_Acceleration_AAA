import numpy as np
import spaceToolsLib as stl


class DistributionClasses:



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


