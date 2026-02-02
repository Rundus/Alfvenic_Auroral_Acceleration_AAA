import numpy as np

from Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_toggles import WaveFieldsToggles
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_toggles import SimToggles
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
import spaceToolsLib as stl
from glob import glob
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()
data_dict_ray_eqns = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/ray_equations/*.cdf*')[0])


class ElectrostaticPotentialClasses:
    def invertedVEField(self, eval_pos):
        mu_alt = stl.Re*(SimClasses.r_muChi(eval_pos[0],eval_pos[1])-1)
        if (mu_alt >= WaveFieldsToggles.inV_Zmin) and (mu_alt <=WaveFieldsToggles.inV_Zmax):
            return (WaveFieldsToggles.inV_Volts) /((WaveFieldsToggles.inV_Zmax - WaveFieldsToggles.inV_Zmin)*stl.m_to_km) # Note, this E-Field is postive --> along mu
        else:
            return 0

class WaveFieldsClasses: # for parallel and perp only

    def __init__(self):
        self.lmb_e = envDict['lambda_e']
        self.V_A = envDict['V_A']

    def field_generator(self, time, eval_pos, **kwargs):

        if (time < 0) or (time>np.max(data_dict_ray_eqns['time'][0])): # always return a zero value if you try to evaluate beyond where the wave is
            return 0
        else:
            which = kwargs.get('type')

            # --- Interpolate ---
            # Get wave position at chosen time
            wave_pos = np.array([
                                    np.interp(time, data_dict_ray_eqns['time'][0], data_dict_ray_eqns['mu_w'][0]),
                                    np.interp(time, data_dict_ray_eqns['time'][0], data_dict_ray_eqns['chi_w'][0]),
                                    np.interp(time, data_dict_ray_eqns['time'][0], data_dict_ray_eqns['phi_w'][0]),
                                 ])

            # Get wave k-vector at chosen time
            k = np.array([
                np.interp(time, data_dict_ray_eqns['time'][0], data_dict_ray_eqns['k_mu'][0]),
                np.interp(time, data_dict_ray_eqns['time'][0], data_dict_ray_eqns['k_perp'][0]),
            ])

            # form the h-factors and k vectors for this specific time
            h = np.array([envDict['h_mu'](wave_pos[0], wave_pos[1]), envDict['h_chi'](wave_pos[0], wave_pos[1])])

            # create the inputs
            inputs = [eval_pos, wave_pos, k, h]

            # Return the desired parameter at this time
            if self.InWaveChecker(inputs):
                if which.lower() == 'potential':
                    return self.Potential_phi(inputs)
                elif which.lower() == 'eperp':
                    return self.EField_perp(inputs)
                elif which.lower() == 'emu':
                    return self.EField_mu(inputs)
                elif which.lower() == 'bperp':
                    return self.BField_perp(inputs)
            else:
                return 0

    def Potential_phi(self, inputs):
        eval_pos, wave_pos, k, h = inputs
        return (WaveFieldsToggles.Phi_0)*np.sin(k[0]*h[0]*(eval_pos[0]-wave_pos[0]))

    def EField_perp(self, inputs):
        eval_pos, wave_pos, k, h= inputs
        Eperp_val  = -1*(k[1]*WaveFieldsToggles.Phi_0 / (2*np.pi)) * (np.sin(k[0] * h[0] * (eval_pos[0]-wave_pos[0])))
        eval_z = (SimClasses.r_muChi(eval_pos[0],eval_pos[1])-1)*stl.Re
        weight = 0.5*(np.tanh((eval_z - 500)/1000) - np.tanh((eval_z - 12000)/1000))
        return Eperp_val*weight
        # return weight

    def BField_perp(self, inputs):
        eval_pos, wave_pos, k, h = inputs
        E_perp = self.EField_perp(inputs)
        return E_perp/(self.V_A(wave_pos[0],wave_pos[1])*np.sqrt(1 + np.power(k[0]*self.lmb_e(wave_pos[0],wave_pos[1]),2)))

    def EField_mu(self, inputs):
        eval_pos, wave_pos, k, h = inputs
        E_perp = self.EField_perp(inputs)
        k_perp = k[1]
        return (k[0]*k_perp*np.square(self.lmb_e(wave_pos[0],wave_pos[1])))*E_perp/(1 + np.square(self.lmb_e(wave_pos[0],wave_pos[1])*k_perp))

    def InWaveChecker(self, inputs):
        eval_pos, wave_pos, k, h = inputs

        # determine the range where you're within the wave, else zero
        lambda_w = 2*np.pi/np.array(k[0])
        delta_w = (lambda_w/2)/np.array(h[0]) # delta = (lambda/2)/ scale_factor

        # check if you're within wave size
        mu_checker = np.all([eval_pos[0] > wave_pos[0] - delta_w, eval_pos[0] < wave_pos[0] + delta_w])

        if mu_checker:
            return True
        else:  # return the field value
            return False




