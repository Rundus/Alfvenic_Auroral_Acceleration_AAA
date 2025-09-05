
# --- imports ---
import spaceToolsLib as stl
import numpy as np

class Ti:

    def shroeder2021(self, simAlt):
        '''
        :param 1D array simAlt:
            simulation altitudes in [m]
        :return:
            Ion Temperature in [eV] for each altitude
        :rtype:
            1D array
        '''
        # convert altitude to km
        z = simAlt / stl.m_to_km

        # --- Ionosphere Temperature Profile ---
        # ASSUMES Ions and electrons have same temperature profile
        T0 = 1  # Temperature at the Ionosphere (in eV)
        T1 = 0.0135  # (in eV)
        h0 = 2000  # scale height (in km)
        T_iono = T1 * np.exp(z / h0) + T0
        deltaZ = 0.3 * stl.Re  # (in km)
        T_ps = 2000  # temperature of plasma sheet (in eV)
        z_ps = 3.75 * stl.Re  # height of plasma sheet (in meters)
        w = 0.5 * (1 - np.tanh((z - z_ps) / deltaZ))  # models the transition to the plasma sheet

        # determine the overall temperature profile
        return np.array([T_iono[i] * w[i] + T_ps * (1 - w[i]) for i in range(len(z))])


class Te:
    def shroeder2021(self, simAlt):
        '''
        :param 1D array simAlt:
            simulation altitudes in [m]
        :return:
            Ion Temperature in [eV] for each altitude
        :rtype:
            1D array
        '''

        # convert altitude to km
        z = simAlt / stl.m_to_km

        # --- Ionosphere Temperature Profile ---
        # ASSUMES Ions and electrons have same temperature profile
        T0 = 1  # Temperature at the Ionosphere (in eV)
        T1 = 0.0135  # (in eV)
        h0 = 2000   # scale height (in km)
        T_iono = T1 * np.exp(z / h0) + T0
        deltaZ = 0.3 * stl.Re # (in km)
        T_ps = 2000  # temperature of plasma sheet (in eV)
        z_ps = 3.75 * stl.Re  # height of plasma sheet (in meters)
        w = 0.5 * (1 - np.tanh((z - z_ps) / deltaZ))  # models the transition to the plasma sheet

        # determine the overall temperature profile
        return np.array([T_iono[i] * w[i] + T_ps * (1 - w[i]) for i in range(len(z))])




class ne:

    # def chaston2002(self, simAlt):

    def kletzingTorbert1994(self, simAlt):
        '''
        returns density for altitude "z [km]" in m^-3
        :param 1D array simAlt:
            simulation altitudes in [m]
        :return:
            plasma density in [m^-3] for each altitude
        :rtype:
            1D array
        '''
        h = 0.06 * (stl.Re / stl.m_to_km)  # in km from E's surface
        n0 = 6E4
        n1 = 1.34E7
        z0 = 0.05 * (stl.Re / stl.m_to_km)  # in km from E's surface
        return (stl.cm_to_m ** 3) * np.array([(n0 * np.exp(-1 * ((alt / stl.m_to_km) - z0) / h) + n1 * ((alt / stl.m_to_km) ** (-1.55))) for alt in simAlt])

    def tanaka2005(self,simAlt):
        ##### TANAKA FIT #####
        # --- determine the density over all altitudes ---
        # Description: returns density for altitude "z [km]" in m^-3
        n0 = 24000000
        n1 = 2000000
        z0 = 600  # in km from E's surface
        h = 680  # in km from E's surface
        H = -0.74
        a = 0.0003656481654202569

        def fitFunc(x, n0, n1, z0, h, H, a):
            return a * (n0 * np.exp(-1 * (x - z0) / h) + n1 * (x ** (H)))

        return (stl.cm_to_m ** 3) * np.array([fitFunc(alt / stl.m_to_km, n0, n1, z0, h, H, a) for alt in simAlt])  # calculated density (in m^-3)

class ni:

    def Chaston2002(self, simAlt):
        '''
        returns density for altitude z in m^-3
        :param 1D array simAlt:
            simulation altitudes in [m]
        :return:
            two ion density array: (1) Oxygen and (2) Hydrogen in [m^-3]
        :rtype:
            1D arrays
        '''

        # convert altitude to km
        z = simAlt/stl.m_to_km

        # oxygen
        n_0 = 400 *z*np.exp(-z/(175)) * np.power(stl.cm_to_m,3)

        # hydrogen
        n_M_H = (0.1 + 10/np.sqrt(400*z/(stl.Re)))
        n_I_H = 100 * z*np.exp(-z/280)
        n_H = (n_I_H + n_M_H)* np.power(stl.cm_to_m,3)

        return n_0, n_H

class ion_composition:

    def Chaston2006(self, simAlt):
        '''
                ratio of oxygen ions to total ion density
                :param 1D array simAlt:
                    simulation altitudes in [m]
                :return:
                     (n_O/n_i) for each altitude
                :rtype:
                    1D array
                '''
        # convert altitude to km
        z = simAlt / stl.m_to_km

        z_i = 2370  #
        h_i = 1800# height of plasma sheet (in meters)
        return 0.5 * (1 - np.tanh((z - z_i) / h_i))

