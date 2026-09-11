
import numpy as np
import spaceToolsLib as stl

class FieldParticleCorrelationClasses:
    def to_Vel(self, Energy_eV):
        return np.sqrt(2 * Energy_eV * stl.q0 / stl.m_e)

    def to_EeV(self, Vel):
        return (0.5 * stl.m_e * np.square(Vel)) / stl.q0