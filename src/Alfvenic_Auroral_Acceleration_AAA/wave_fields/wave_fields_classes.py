import spaceToolsLib as stl
from glob import glob
import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
from src.Alfvenic_Auroral_Acceleration_AAA.scale_length.scale_length_classes import ScaleLengthClasses
envDict = ScaleLengthClasses().loadPickleFunctions()
data_dict_wavescale = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/scale_length/*.cdf*')[0])
data_dict_plasma = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/plasma_environment/*.cdf*')[0])

class WaveFieldsClasses:

    def PhiShape(self,x,Phi0,z0,k):
        return (Phi0/2)*(1+np.cos(k*(x-z0)))

    def Potential_Shape_phi(self,x,Phi0,wave_pos,k):
        h_phi=envDict['h_phi'](wave_pos[0], wave_pos[1])
        return (Phi0/2)*(1+np.sin((k*h_phi)*(x-wave_pos[2])))
        # return 1

    def InWaveChecker(self,tme_idx, mu,chi,phi,wave_pos):
        mu_w, chi_w, phi_w = wave_pos

        # determine the range where you're within the wave, else zero
        lambda_mu = data_dict_wavescale['lambda_mu'][0][tme_idx] # delta = (lambda/2)/ scale_factor
        lambda_chi = data_dict_wavescale['lambda_chi'][0][tme_idx]
        lambda_phi = data_dict_wavescale['lambda_phi'][0][tme_idx]

        delta_mu = np.abs((lambda_mu / 2) / envDict['h_mu'](mu_w, chi_w))
        delta_chi = (lambda_chi / 2) / envDict['h_chi'](mu_w, chi_w)
        delta_phi = (lambda_phi / 2) / envDict['h_phi'](mu_w, chi_w)

        # check if you're within wave size

        mu_checker = np.all([mu < mu_w + delta_mu, mu > mu_w - delta_mu])
        # print(mu - delta_mu, mu_w, mu + delta_mu,mu_checker)
        chi_checker = np.all([chi_w - delta_chi < chi, chi < chi_w + delta_chi])
        phi_checker = np.all([phi_w - delta_phi<phi, phi < phi_w + delta_phi])
        print(phi - delta_phi, phi_w, phi + delta_phi, phi_checker)

        if np.all([mu_checker,chi_checker,phi_checker]):
            return True
        else:  # return the field value
            return False

    def Potential_phi(self,t,mu,chi,phi):
        tme_idx = np.abs(data_dict_wavescale['time'][0] - t).argmin()
        wave_pos = [data_dict_wavescale['mu_w'][0][tme_idx],data_dict_wavescale['chi_w'][0][tme_idx],data_dict_wavescale['phi_w'][0][tme_idx]]
        k_phi = data_dict_wavescale['k_phi'][0][tme_idx]

        if self.InWaveChecker(tme_idx,mu,chi,phi,wave_pos):
            return self.Potential_Shape_phi(phi, WaveFieldsToggles.Phi_0, wave_pos, k_phi)
        else:
            return 0



