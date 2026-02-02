import numpy as np
import spaceToolsLib as stl

class SimClasses:


    # Convert output to geophysical coordinates
    def r_muChi(mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            distance along geomagnetic field line, measure from earth's center in [km]
        '''

        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.square(zeta) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)

        r = u / chi  # in R_E from earth's center

        return r

    def theta_muChi(mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            distance along geomagnetic field line in [m]
        '''
        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.square(zeta) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)
        return np.degrees(np.arcsin(np.sqrt(u)))

    def to_Vel(self,Energy_eV):
        return np.sqrt(2*Energy_eV*stl.q0/stl.m_e)

    def to_EeV(self, Vel):
        return (0.5*stl.m_e*np.square(Vel))/stl.q0