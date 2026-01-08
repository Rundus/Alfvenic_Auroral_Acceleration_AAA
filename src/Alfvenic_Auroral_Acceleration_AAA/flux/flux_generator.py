
def flux_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from copy import deepcopy

    # --- File-specific imports ---
    from glob import glob
    from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.flux.flux_toggles import FluxToggles
    from itertools import product

    # --- Load the wave simulation data ---
    data_dict_distribution = stl.loadDictFromFile(glob(rf'{SimToggles.sim_data_output_path}/distributions/*.cdf*')[0])

    #####################################
    # --- CALCULATE DIFFERENTIAL FLUX ---
    #####################################

    # [1] determine the Energy Grid and Pitch Angle Grid
    Energies = []
    Pitches = []
    for mu_idx in range(len(data_dict_distribution['v_mu_range_eV'][0])):
        for perp_idx in range(len(data_dict_distribution['v_perp_range_eV'][0])):
            vperp = data_dict_distribution['v_perp_range'][0][perp_idx]
            v_mu = data_dict_distribution['v_mu_range'][0][mu_idx]
            Energies.append(0.5*(stl.m_e/stl.q0)*(np.square(vperp) + np.square(v_mu)))
            Pitches.append(180+np.degrees(np.arctan2(-vperp,v_mu)))

    # find every unique pitch angle and Energy - create a grid based off of these
    uEnergies = sorted(list(set(Energies)))
    uPitches = sorted(list(set(Pitches)))

    # for each pitch angle, find where in the uPitches it belongs
    Pitches_idxs = np.array([np.abs(uPitches - val).argmin() for val in Pitches])
    Energies_idxs = np.array([np.abs(uEnergies - val).argmin() for val in Energies])

    # [2] Rebin the Distribution data in terms of Pitch Angle and Energy
    sizes = [len(data_dict_distribution['time_eval'][0]),  len(uPitches),len(uEnergies)]
    Distribution_rebin = np.zeros(shape=(sizes))

    for tmeIdx in range(sizes[0]):
        for mu_idx in range(len(data_dict_distribution['v_mu_range_eV'][0])):
            for perp_idx in range(len(data_dict_distribution['v_perp_range_eV'][0])):
                idx = mu_idx*(len(data_dict_distribution['v_perp_range_eV'][0])) + perp_idx
                ptch_idx = Pitches_idxs[idx]
                engy_idx = Energies_idxs[idx]
                Distribution_rebin[tmeIdx][ptch_idx][engy_idx] = data_dict_distribution['Distribution'][0][tmeIdx][mu_idx][perp_idx]

    # [3] Bin the data to be similar to the Electrostatic analyzers. Average distribution function.
    Pitch_Angle_ESA = [0 + 10*i for i in range(19)]
    Energies_ACESII = [13678.4, 11719.21, 10040.64, 8602.5, 7370.34, 6314.67, 5410.2, 4635.29,
             3971.37, 3402.54, 2915.18, 2497.64, 2139.89, 1833.39, 1570.79, 1345.8,
             1153.04, 987.89, 846.39, 725.16, 621.29, 532.3, 456.06, 390.74,
             334.77, 286.82, 245.74, 210.54, 180.39, 154.55, 132.41, 113.45,
             97.2, 83.28, 71.35, 61.13, 52.37, 44.87, 38.44, 32.94,
             28.22, 24.18, 20.71, 17.75, 15.21, 13.03, 11.16, 9.56, 8.19]
    Energies_ESA = Energies_ACESII[::-1]
    Distribution_ESA = [[[[] for eval in Energies_ESA] for pval in Pitch_Angle_ESA] for tval in range(sizes[0])]

    for tmeIdx, ptchIdx, engyIdx in product(*[range(val) for val in sizes]):
        Eidx = np.abs(Energies_ESA - uEnergies[engyIdx]).argmin()
        Pidx = np.abs(Pitch_Angle_ESA-uPitches[ptchIdx]).argmin()
        Distribution_ESA[tmeIdx][Pidx][Eidx].append(Distribution_rebin[tmeIdx][ptchIdx][engyIdx])

    # average everything together
    sizes_ESA = [sizes[0],len(Pitch_Angle_ESA),len(Energies_ESA)]
    for TmeIdx, PtchIdx, EngyIdx in product(*[range(thing) for thing in sizes_ESA]):
        if len(Distribution_ESA[TmeIdx][PtchIdx][EngyIdx]) >= 1:
            if len(np.nonzero(Distribution_ESA[TmeIdx][PtchIdx][EngyIdx])[0]) >=0:
                Data = np.array(Distribution_ESA[TmeIdx][PtchIdx][EngyIdx])
                Distribution_ESA[TmeIdx][PtchIdx][EngyIdx] = np.mean(Data[np.nonzero(Data)])
            else:
                Distribution_ESA[TmeIdx][PtchIdx][EngyIdx] = 0
        else:
            Distribution_ESA[TmeIdx][PtchIdx][EngyIdx] = 0

    ###########################
    # --- OUTPUT EVERYTHING ---
    ###########################

    # --- prepare the output ---
    data_dict_output = {
        'time': [deepcopy(data_dict_distribution['time_eval'][0]), {'DEPEND_0':'time','UNITS': 's', 'LABLAXIS': 'Time','VAR_TYPE':'data'}],
        'Energy': [np.array(Energies_ESA), {'UNITS': 'eV', 'LABLAXIS': 'Energy', 'VAR_TYPE': 'support_data'}],
        'Pitch_Angle': [np.array(Pitch_Angle_ESA), {'UNITS': 'deg', 'LABLAXIS': 'Pitch Angle', 'VAR_TYPE': 'support_data'}],
        'Distribution_Function': [np.array(Distribution_ESA), {'DEPEND_0':'time','DEPEND_2':'Energy','DEPEND_1':'Pitch_Angle','UNITS':'m!A-6!Ns!A-3!N','LABLAXIS':'Distribution Function','VAR_TYPE':'data'}],
        # 'Differential_Number_Flux': [np.array([]), {'DEPEND_0': 'time', 'DEPEND_1': 'Pitch_Angle', 'DEPEND_2': 'Energy', 'UNITS': ''}],
    }

    outputPath = rf'{FluxToggles.outputFolder}/flux.cdf'
    stl.outputDataDict(outputPath, data_dict_output)