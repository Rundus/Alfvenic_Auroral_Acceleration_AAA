import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.sim_classes import SimClasses
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EvironmentExpressionsClasses
import spaceToolsLib as stl
from glob import glob
envDict = EvironmentExpressionsClasses().loadPickleFunctions()
data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/*.cdf*')[0])


class ElectrostaticPotentialClasses:
    def invertedVEField(self, eval_pos):
        mu_alt = stl.Re*(SimClasses.r_muChi(eval_pos[0],eval_pos[1])-1)
        if (mu_alt >= WaveFieldsToggles.inV_Zmin) and (mu_alt <=WaveFieldsToggles.inV_Zmax):
            return (WaveFieldsToggles.inV_Volts) /((WaveFieldsToggles.inV_Zmax - WaveFieldsToggles.inV_Zmin)*stl.m_to_km) # Note, this E-Field is postive --> along mu
        else:
            return 0

class WaveFieldsClasses2D: # for parallel and perp only

    def __init__(self):
        self.k_vectors = np.array([data_dict_wavescale['k_mu'][0], data_dict_wavescale['k_perp'][0]]).T
        self.wave_pos_vector = np.array([data_dict_wavescale['mu_w'][0], data_dict_wavescale['chi_w'][0], data_dict_wavescale['phi_w'][0]]).T
        self.h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]
        self.lmb_e = envDict['lambda_e']
        self.V_A = envDict['V_A']

    def Potential_phi(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h = inputs
        return (WaveFieldsToggles.Phi_0/2)*np.sin(k[0]*h[0]*(eval_pos[0]-wave_pos[0]))

    def EField_phi(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h= inputs
        return -1*(k[1]*WaveFieldsToggles.Phi_0 / (4*np.pi)) * (np.sin(k[0] * h[0] * (eval_pos[0]-wave_pos[0])))

    def BField_perp(self,inputs):
        tme_idx, eval_pos, wave_pos, k, h = inputs
        E_perp = self.EField_phi(inputs)
        return E_perp/(self.V_A(wave_pos[0],wave_pos[1])*np.sqrt(1 + np.power(k[0]*self.lmb_e(wave_pos[0],wave_pos[1]),2)))

    def EField_mu(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h = inputs
        E_perp = self.EField_phi(inputs)
        k_perp = k[1]
        return (k[0]*k_perp*np.square(self.lmb_e(wave_pos[0],wave_pos[1])))*E_perp/(1 + np.square(self.lmb_e(wave_pos[0],wave_pos[1])*k_perp))

    def InWaveChecker(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h = inputs

        # determine the range where you're within the wave, else zero
        lambda_w = 2*np.pi/np.array(k[0])
        delta_w = (lambda_w/2)/np.array(h[0]) # delta = (lambda/2)/ scale_factor

        # check if you're within wave size
        mu_checker = np.all([eval_pos[0] > wave_pos[0] - delta_w, eval_pos[0] < wave_pos[0] + delta_w])

        if mu_checker:
            return True
        else:  # return the field value
            return False

    def field_generator(self, time, eval_pos, **kwargs):

        which = kwargs.get('type')

        # get the specifics of the wave at the chosen time
        tme_idx = np.abs(data_dict_wavescale['time'][0] - time).argmin()
        wave_pos = self.wave_pos_vector[tme_idx]
        h = np.array([self.h_factors[0](wave_pos[0], wave_pos[1]), self.h_factors[1](wave_pos[0], wave_pos[1])])
        k = self.k_vectors[tme_idx]

        # create the inputs
        inputs = [tme_idx, eval_pos, wave_pos, k, h]

        if self.InWaveChecker(inputs):
            if which.lower() == 'potential':
                return self.Potential_phi(inputs)
            elif which.lower() == 'eperp':
                return self.EField_phi(inputs)
            elif which.lower() == 'epara':
                return self.EField_mu(inputs)
            elif which.lower() == 'bperp':
                return self.BField_perp(inputs)
        else:
            return 0


