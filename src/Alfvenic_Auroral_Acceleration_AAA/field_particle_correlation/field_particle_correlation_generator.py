import matplotlib.pyplot as plt
from timebudget import timebudget

@timebudget
def field_particle_correlation_generator():

    # --- general imports ---
    import spaceToolsLib as stl
    import numpy as np
    from glob import glob
    import os
    import re

    # --- File-specific imports ---
    from tqdm import tqdm
    from scipy.stats import binned_statistic
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.FileIO import FileIOClass
    from src.Alfvenic_Auroral_Acceleration_AAA.runners.executable_classes import ExecutableClasses
    run_module = ExecutableClasses().load_configuration_module()
    PlasmaEnvironmentToggles = run_module.PlasmaEnvironmentToggles
    from src.Alfvenic_Auroral_Acceleration_AAA.liouville_mapping.liouville_mapping_classes import LiouvilleMapping
    from src.Alfvenic_Auroral_Acceleration_AAA.field_particle_correlation.field_particle_correlation_classes_claude import FPCClasses

    # --- Delete the old FPC Files ---
    old_files = glob(f'{FileIOClass.path_sim_output}/field_particle_correlation/*.cdf*')
    for old_file in old_files:
        os.remove(old_file)

    # --- Load the liouville mapped data ---
    dist_files = glob(rf'{FileIOClass.path_sim_output}/liouville_mapping/*.cdf*')

    # --- Load the wave environment data ---
    data_dict_waves = stl.loadDictFromFile(glob(rf'{FileIOClass.path_sim_output}/wave_potentials/*.cdf*')[0])

    # --- load the plasma environment data ---
    data_dict_plasma = stl.loadDictFromFile(glob(rf'{FileIOClass.path_sim_output}/plasma_environment/*.cdf*')[0])
    dist_obj = LiouvilleMapping(1000) # altitude doesn't matter here

    # loop over distribution function files
    for i in tqdm(range(len(dist_files))):

        ##########################
        # --- PREPARE THE DATA ---
        ##########################

        # Load the file-specific data
        data_dict = stl.loadDictFromFile(dist_files[i])

        # determine which altitude is in the file
        match = re.search(r"_(\d+(?:\.\d+)?)km\.cdf$", dist_files[i])
        alt_val = float(match.group(1))  # [km] altitude

        # get the Liouville Mapping Data
        f = data_dict['distribution_function'][0].copy()
        energy = data_dict['energy'][0].copy()
        pitch_angle = data_dict['pitch_angle'][0].copy()
        time_waves = data_dict['time_waves'][0].copy()
        time = data_dict['time'][0].copy()

        E_para_obs = -1 * data_dict['E_mu_obs'][0].copy()  # put a -1 sign since we're using parallel coordinates
        mask = np.where(np.abs(E_para_obs) < 1E-9)
        E_para_obs[mask] = 0

        res = FPCClasses.compute_fpc(f=f, pitch_angle=pitch_angle, energy=energy, Epar=E_para_obs, t=time)  # or FPCClasses().compute_fpc(...)

        # get the data
        r = res["full"]
        grid = res["grid"]
        vg = FPCClasses.native_velocity_grid(grid["alpha"], grid["E"], grid["alpha_edges"], grid["E_edges"])
        # vparas, vperps = vg["vpar_c"], vg["vperp_c"]  # centre of each (pitch, energy) bin
        # vg["vpar_e"], vg["vperp_e"]  # bin corners, for pcolormesh(vg["vpar_e"], vg["vperp_e"], r["C_native_tint"])

        FPC_integrate_time = r['C_native_tint']
        FPC_integrate_verp = r['C']
        FPC_integrate_vel = r['C_int_v']
        vparas = res['deposition']['vpar']


        # # --- calculate the delta_f0 term ---
        # # the FPC is more clear if you subtract the ambient plasma distribution
        # # get the cold density from the altitude
        # X,Y = np.meshgrid(pitch_angle,energy)
        # f_0_PS = dist_obj.f_maxwellian(E_eV=Y.T, n0=PlasmaEnvironmentToggles.n0_PS*(stl.cm_to_m**3), Te=PlasmaEnvironmentToggles.Te_PS)
        #
        # n_density_cold_idx = np.abs(data_dict_plasma['alt'][0] - alt_val*stl.m_to_km).argmin()
        # f_0_cold = dist_obj.f_maxwellian(E_eV=Y.T, n0=data_dict_plasma['n_density_cold'][0][n_density_cold_idx], Te=PlasmaEnvironmentToggles.Te_cold)
        # f_0 = f_0_PS + f_0_cold
        # f_0 = f_0[None, :, :]
        # deltaf0 = f.copy() - f_0 # calculate delta_f0
        #
        # # ---- bin-average the wave data onto the particle data ---
        # E_para_obs = -1 * data_dict['E_mu_obs'][0].copy()  # put a -1 sign since we're using parallel coordinates
        # mask = np.where(np.abs(E_para_obs) < 1E-9)
        # E_para_obs[mask] = 0
        # mid = 0.5 * (time[1:] + time[:-1])  # get bins by calculating the deltaT/2 from the particle timestamps
        # edges = np.concatenate(([2 * time[0] - mid[0]], mid, [2 * time[-1] - mid[-1]]))
        # E_para_FPC, _, _ = binned_statistic(time_waves, E_para_obs.copy(), bins=edges)
        #
        # ##############################################
        # # --- CALCULATE FPC (DETECTOR COORDINATES) ---
        # ##############################################
        #
        # distribution = deltaf0.copy()
        #
        # # --- Calculate the instant FPC in detector coordinates, C_Epara(r, v, t) ---
        # FPC_instant = FPCClasses().instant_FPC_detector_coords(energies=energy,
        #                                                        pitch_angle=pitch_angle,
        #                                                        distribution=distribution,
        #                                                        Epara=E_para_FPC)
        #
        # FPC_alternate_instant = FPCClasses().instant_alternate_FPC_detector_coords(energies=energy,
        #                                                                    pitch_angle=pitch_angle,
        #                                                                    distribution=distribution,
        #                                                                    Epara=E_para_FPC)
        #
        # # --- Calculate the time-integrated FPC, C_Epara(r, v) ---
        # FPC = FPCClasses().FPC_detector_coords(FPC_instant=FPC_instant,
        #                                        time_obs=time)
        #
        # FPC_alternate = FPCClasses().FPC_detector_coords(FPC_instant=FPC_alternate_instant,
        #                                                  time_obs=time)
        #
        # # --- Calculate the FPC collapsed over Vperp ---
        #
        # vparas, FPC_vpara = FPCClasses().reduce_to_vpar(pitch_deg=pitch_angle,
        #                                                 energy_eV=energy,
        #                                                 values=FPC)
        #
        # # --- Calculate the FPC collapsed over Vperp integral but NOT time ---
        # vparas = []
        # FPC_vpara_instant = []
        # for idx in range(len(time)):
        #     vpar, vals = FPCClasses().reduce_to_vpar(pitch_deg=pitch_angle,
        #                                              energy_eV=energy,
        #                                              values=FPC_instant[idx])
        #     vparas = vpar
        #     FPC_vpara_instant.append(vals)
        #
        # vparas = np.array(vparas)
        # FPC_vpara_instant = np.array(FPC_vpara_instant)
        #
        # # --- Calculate the FPC over collapsed Vperp and Vpar coordinates ---
        # FPC_total = FPCClasses().total_integral(pitch_deg=pitch_angle,
        #                                         energy_eV=energy,
        #                                         values=FPC_instant)



        # --- get the DAW velocity for comparison ---
        DAW_idx = np.abs(data_dict_waves['alt'][0] - alt_val*stl.m_to_km).argmin()
        DAW_vel_obs = np.array([data_dict_waves['DAW_velocity'][0][DAW_idx]])

        ################
        # --- OUTPUT ---
        ################
        data_dict_output = {
            'time': data_dict['time'].copy(),
            'FPC_integrate_time': [np.array(FPC_integrate_time), {'VAR_TYPE': 'data','DEPEND_0': 'pitch_angle','DEPEND_1': 'energy'}],
            'FPC_integrate_vpara' : [np.array(FPC_integrate_verp),{}],
            'FPC_integrate_velspace': [np.array(FPC_integrate_vel), {}],
            # 'FPC_total': [np.array([FPC_total]), {'VAR_TYPE': 'data'}],
            # 'FPC_instant': [np.array(FPC_instant), {'DEPEND_0': 'time','DEPEND_1': 'pitch_angle','DEPEND_2': 'energy', 'VAR_TYPE': 'data','LABLAXIS':'FPC instant'}],
            # 'FPC_alternate_instant': [np.array(FPC_alternate_instant), {'DEPEND_0': 'time', 'DEPEND_1': 'pitch_angle', 'DEPEND_2': 'energy', 'VAR_TYPE': 'data','LABLAXIS':'FPC alternate instant'}],
            # 'FPC_alternate': [np.array(FPC_alternate), {'DEPEND_0': 'pitch_angle', 'DEPEND_1': 'energy', 'VAR_TYPE': 'data', 'LABLAXIS': 'FPC alternate'}],
            # 'FPC_vpara': [np.array(FPC_vpara), {'DEPEND_0': 'vparas','VAR_TYPE': 'data','LABLAXIS':'FPC Vpara'}],
            # 'FPC_vpara_instant':[np.array(FPC_vpara_instant).T,{'DEPEND_1':'time','DEPEND_0':'vparas','VAR_TYPE': 'data','LABLAXIS':'FPC Vpara instant'}],
            # 'vparas':[np.array(vparas),{'UNITS':'m/s'}],
            # 'E_para_FPC' : [E_para_FPC,data_dict['E_mu_obs'][1].copy()],
            'pitch_angle':data_dict['pitch_angle'].copy(),
            'energy': data_dict['energy'].copy(),
            'distribution':[f,{'DEPEND_0': 'time','DEPEND_1': 'pitch_angle','DEPEND_2': 'energy', 'VAR_TYPE': 'data','UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'simulated distribution functon'}],
            # 'f0':[f_0,{'DEPEND_0': 'pitch_angle','DEPEND_1': 'energy', 'VAR_TYPE': 'data','UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': 'f!B0!N'}],
            # 'delta_f0':[distribution,{'DEPEND_0': 'time','DEPEND_1': 'pitch_angle','DEPEND_2': 'energy', 'VAR_TYPE': 'data','UNITS': 'm!A-6!Ns!A-3!N', 'LABLAXIS': '&delta;f!B0!N'}],
            'DAW_vel_obs':[DAW_vel_obs,{'UNITS':'m/s','VAR_TYPE':'data','LABLAXIS':'DAW Velocity Observed'}]
        }

        data_dict_output['E_para_FPC'][1]['DEPEND_0'] = 'time'
        # save the results
        mapping_alt = int(re.search(r'_(\d+)km', dist_files[i]).group(1))
        outputPath = rf'{FileIOClass.path_sim_output}/field_particle_correlation/FPC_{mapping_alt}km.cdf'
        stl.outputDataDict(outputPath, data_dict_output)