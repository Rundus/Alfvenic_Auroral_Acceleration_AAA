


class DistributionClasses:



    def Maxwellian(self, n, T, vel_para, vel_perp): # returns the maxwellian distribution for a given temperature, density and particle velocity
        """
        :param n: Plasma Density in [m^-3]
        :type n : float

        :param T: Plasma Temperature in [eV]
        :type T: float

        :param vel_para: Particle Velocity parallel to the background geomagnetic field
        :type vel_para: float

        :param vel_perp: Particle Velocity parallel to the background geomagnetic field
        :type vel_perp: float

        :return: Plasma Distribution Function in [m^-6 s^-3] evaluated at vel_para, vel_perp
        """


