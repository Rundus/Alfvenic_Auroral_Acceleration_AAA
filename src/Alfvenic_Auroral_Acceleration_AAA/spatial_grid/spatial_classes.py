import numpy as np
import spaceToolsLib as stl
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

class SpatialClasses:




    # Convert output to geophysical coordinates
    def r_muChi(self,mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            distance from earth's center in [km]
        '''

        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.square(zeta) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)

        r = u / chi  # in R_E from earth's center

        return r

    def theta_muChi(self, mu, chi):
        '''
        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :return:
            colatitude
        '''
        zeta = np.power(mu / chi, 4)
        c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
        c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
        gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.square(zeta) + 256 * np.power(zeta, 3))) ** (1 / 3)
        w = - c1 / gamma + gamma / (c2 * zeta)
        u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)
        return np.degrees(np.arcsin(np.sqrt(u)))


    def mu_from_field_line_distance(self,mu0,chi0,z_para_target):
        '''
        Finds the final mu value for a given field-line distance starting at (mu0,chi0), S = int_u0^uf (h_mu)dmu. Done by finite integrating
        the differential length element between u0 and uf, where uf is a variable that's solved for by minimizing (S-z_para_target)

        :param mu:
            mu coordinate value
        :param chi:
            chi coordinate value
        :param z_para_target:
            target distance along geomagnetic field line to determine the uf from
        :return:
            [m] Distance along geomagnetic field line
        '''

        # Get the mu scale factor
        from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
        envDict = EnvironmentExpressionsClasses().loadPickleFunctions()

        def field_align_arc_length(mu, mu0, chi0, z_para_target):

            # Determine
            z_para = quad(
                # func=self.h_mu,
                func=envDict['h_mu'],
                a=mu0,
                b=mu,
                args=(chi0,)
            )
            return np.abs(z_para[0] - z_para_target * stl.m_to_km)

        uf = minimize_scalar(field_align_arc_length,
                             args=(mu0, chi0,z_para_target,),
                             bounds=(mu0,0))
        return uf.x