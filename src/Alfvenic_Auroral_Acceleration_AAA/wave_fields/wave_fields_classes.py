import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
import spaceToolsLib as stl
from glob import glob

envDict = ScaleLengthClasses().loadPickleFunctions()
data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/*.cdf*')[0])
k_vectors = np.array([data_dict_wavescale['k_mu'][0], data_dict_wavescale['k_chi'][0], data_dict_wavescale['k_phi'][0]]).T
wave_pos_vector = np.array([data_dict_wavescale['mu_w'][0],data_dict_wavescale['chi_w'][0],data_dict_wavescale['phi_w'][0]]).T
h_factors = [envDict['h_mu'], envDict['h_chi'], envDict['h_phi']]

class WaveFieldsClasses:

    def Potential_phi(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h, lmb_e = inputs
        return (WaveFieldsToggles.Phi_0/2)*(h[2]*(eval_pos[2]-wave_pos[2])*np.sin(k[0]*h[0]*(eval_pos[0]-wave_pos[0])))

    def EField_phi(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h, lmb_e = inputs
        return -1*(k[2]*WaveFieldsToggles.Phi_0 / (4*np.pi)) * (np.sin(k[0] * h[0] * (eval_pos[0]-wave_pos[0])))

    def EField_mu(self, inputs):
        tme_idx, eval_pos, wave_pos, k, h, lmb_e = inputs
        E_perp = self.EField_phi(inputs)
        k_perp = np.sqrt(k[1]**2 + k[2]**2)
        return (k[0]*k_perp*np.square(lmb_e))*E_perp/(1 + np.square(lmb_e*k_perp))

    def InWaveChecker(self,inputs):
        tme_idx, eval_pos, wave_pos, k, h, lmb_e = inputs

        # determine the range where you're within the wave, else zero
        lambda_w = 2*np.pi/np.array(k)
        delta_w = (lambda_w/2)/np.array(h) # delta = (lambda/2)/ scale_factor

        # check if you're within wave size
        mu_checker = np.all([eval_pos[0] > wave_pos[0] - delta_w[0], eval_pos[0] < wave_pos[0] + delta_w[0]])
        chi_checker = np.all([eval_pos[1] > wave_pos[1] - delta_w[1], eval_pos[1] < wave_pos[1] + delta_w[1]])
        phi_checker = np.all([eval_pos[2] > wave_pos[2] - delta_w[2], eval_pos[2] < wave_pos[2] + delta_w[2]])
        # print([wave_pos[2] - delta_w[2],wave_pos[2],wave_pos[2] + delta_w[2],eval_pos[2]])

        if np.all([mu_checker, chi_checker, phi_checker]):
            return True
        else:  # return the field value
            return False

    def field_generator(self, time, eval_pos,**kwargs):

        which = kwargs.get('type')

        # get the specifics of the wave at the chosen
        tme_idx = np.abs(data_dict_wavescale['time'][0] - time).argmin()
        wave_pos = wave_pos_vector[tme_idx]
        h = np.array([h_factors[0](wave_pos[0], wave_pos[1]), h_factors[1](wave_pos[0], wave_pos[1]), h_factors[2](wave_pos[0], wave_pos[1])])
        k = k_vectors[tme_idx]
        lmb_e = data_dict_plasma['lambda_e'][0][tme_idx]

        # create the inputs
        inputs = [tme_idx, eval_pos, wave_pos, k, h, lmb_e]

        if self.InWaveChecker(inputs):
            if which.lower() == 'potential':
                return self.Potential_phi(inputs)
            elif which.lower() == 'eperp':
                return self.EField_phi(inputs)
            elif which.lower() == 'epara':
                return self.EField_mu(inputs)
        else:
            return 0



