from timebudget import timebudget
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *

@timebudget
def verify_system_energy():
    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    import scipy
    from src.Alfvenic_Auroral_Acceleration_AAA.wave_fields.wave_fields_classes import WaveFieldsClasses as WaveFieldsClasses
    from src.Alfvenic_Auroral_Acceleration_AAA.simulation.sim_classes import SimClasses
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scipy.integrate import simpson

    # --- Load the needed data ---
    data_dict_wave_fields = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/wave_fields/wave_fields_characteristics.cdf')[0])

    # prepare the output
    data_dict_output = {
        'time': [np.array(deepcopy(data_dict_wave_fields['time'][0])),deepcopy(data_dict_wave_fields['time'][1])],
        'z': deepcopy(data_dict_wave_fields['z']),
        'dz': deepcopy(data_dict_wave_fields['z']),
        'E_perp': [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'mV/m', 'LABLAXIS': 'E!B&perp;!N', 'VAR_TYPE': 'data'}],
        'E_mu': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'mV/m', 'LABLAXIS': 'E!B&mu;!N', 'VAR_TYPE': 'data'}],
        'B_perp': [[],{'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'nT', 'LABLAXIS': 'B!B&perp;!N', 'VAR_TYPE': 'data'}],
        'B_mu': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'nT', 'LABLAXIS': 'B!B&mu;!N', 'VAR_TYPE': 'data'}],
        'Az': [[], {'DEPEND_0': 'time','DEPEND_1':'z', 'UNITS': 'Wb/m', 'LABLAXIS': 'A!Bz;!N', 'VAR_TYPE': 'data'}],
        'Phi': [[], {'DEPEND_0': 'time', 'DEPEND_1': 'z', 'UNITS': 'V', 'LABLAXIS': '&Phi;', 'VAR_TYPE': 'data'}],
        'System_Energy':[[],{'DEPEND_0':'time','UNITS':'eV','LABLAXIS':'Total Energy','VAR_TYPE':'data'}],
        'System_Energy_Emu': [[], {'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Total Energy (E!B&mu;!N)', 'VAR_TYPE': 'data'}],
        'System_Energy_Eperp': [[], {'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Total Energy (E!B&perp;!N)', 'VAR_TYPE': 'data'}],
        'System_Energy_Bperp': [[], {'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Total Energ (B!B&perp;!N)y', 'VAR_TYPE': 'data'}],
        'System_Energy_Total': [[], {'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Total Energy', 'VAR_TYPE': 'data'}],
        'Input_Energy':[[],{'DEPEND_0':'time','UNITS':'eV','LABLAXIS':'Input Energy','VAR_TYPE':'data'}],
        'Input_Energy_cumsum': [[], {'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Input Energy', 'VAR_TYPE': 'data'}],
        'Input_Energy_Theory': [[],{'DEPEND_0': 'time', 'UNITS': 'eV', 'LABLAXIS': 'Input Energy', 'VAR_TYPE': 'data'}],
    }

    # ====================================
    # 9. Calculate the total System energy
    # ====================================
    # Use Poynting's theorem to calculate the total wave energy: Energy(t) = integral( dr^3 (|E(t)|^2 + |B(t)|^2)/8pi )
    E_mu = (1E-3)*data_dict_wave_fields['E_mu'][0]
    B_mu = (1E-9)*data_dict_wave_fields['B_mu'][0]
    E_perp = (1E-3) * data_dict_wave_fields['E_perp'][0]
    B_perp = (1E-9) * data_dict_wave_fields['B_perp'][0]
    Fields = [E_mu,E_perp,B_mu,B_perp]
    lambda_perp = 2*np.pi/data_dict_wave_fields['k_perp'][0]
    crossArea = (np.pi*(lambda_perp/4)**2) # assume cylindrical cross area
    z = data_dict_wave_fields['z'][0]
    data_dict_output['dz'][0] = np.diff(data_dict_wave_fields['z'][0], prepend=data_dict_wave_fields['z'][0][0])

    # ignore the boundary points, multiply by the cross-sectional area then integrate
    input_energy = [[],[],[],[]]
    for idx in range(len(Fields)):
        Fields[idx][np.isnan(Fields[idx])] = 0
        Fields[idx][np.isinf(Fields[idx])] = 0
        Fields[idx] = crossArea*np.square(Fields[idx])
        input_energy[idx] = Fields[idx][:,-1]*(z[-1]-z[-2])
        Fields[idx] = simpson(y=Fields[idx],x=z,axis=1)


    factor = (np.pi/stl.u0) * (np.pi**2 - 4)/(16)
    E_tot_Bperp = np.sum(factor*np.square(B_perp)*data_dict_output['dz'][0]/(np.square(data_dict_wave_fields['k_perp'][0])),axis=1)
    factor = np.pi*stl.ep0 * (np.pi ** 2 - 4) / (16)
    E_tot_Eperp = np.sum(factor * np.square(E_perp) * data_dict_output['dz'][0] / (np.square(data_dict_wave_fields['k_perp'][0])), axis=1)
    E_tot_Epara = np.sum(((np.power(np.pi,3)*stl.ep0)/(8*np.square(data_dict_wave_fields['k_perp'][0])))*np.square(E_mu)*data_dict_output['dz'][0],axis=1)
    data_dict_output['System_Energy_Total'][0] = E_tot_Bperp + E_tot_Eperp + E_tot_Epara
    data_dict_output['System_Energy_Bperp'][0] = E_tot_Bperp
    data_dict_output['System_Energy_Eperp'][0] = E_tot_Eperp
    data_dict_output['System_Energy_Emu'][0] = E_tot_Epara
    data_dict_output['System_Energy'][0] = 0.5*stl.ep0*(Fields[0] + Fields[1] ) + (0.5/stl.u0)*(Fields[2] + Fields[3])
    data_dict_output['Input_Energy'][0] = 0.5 * stl.ep0 * (input_energy[0] + input_energy[1]) + (0.5 / stl.u0) * (input_energy[2] + input_energy[3])
    data_dict_output['Input_Energy_cumsum'][0] = np.cumsum(np.array(0.5 * stl.ep0 * (input_energy[0] + input_energy[1]) + (0.5 / stl.u0) * (input_energy[2] + input_energy[3])))

    Phi0_driver = 100  # in eV
    freq_driver = 4  # in Hz
    data_dict_output['Input_Energy_Theory'][0] = crossArea[-1]*(z[-1]-z[-2])*np.array([2*Phi0_driver/(np.pi*freq_driver) for i in range(len(data_dict_wave_fields['time'][0]))])
    data_dict_output['dz'][0] = np.diff(data_dict_wave_fields['z'][0],prepend=data_dict_wave_fields['z'][0][0])

    # ==================================================
    # 10. OUTPUT DATA
    # ==================================================
    outputPath = rf'{WaveFieldsToggles.outputFolder}/tests/wave_fields_system_energy.cdf'
    stl.outputDataDict(outputPath, data_dict_output)


verify_system_energy()
