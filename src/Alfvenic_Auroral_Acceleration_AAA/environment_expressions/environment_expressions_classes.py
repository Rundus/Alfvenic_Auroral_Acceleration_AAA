from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles


class EvironmentExpressionsClasses:


    def loadPickleFunctions(self):
        from glob import glob
        import dill

        pickle_files = glob(rf'{SimToggles.sim_root_path}/ray_equations/pickled_expressions/*.pkl*')
        for file_nam in pickle_files:
            func = dill.load(open(file_nam, 'rb'))
            if 'lmb_e.pkl' in file_nam:
                lmb_e = func
            elif 'pDD_lmb_e_mu.pkl' in file_nam:
                pDD_mu_lmb_e = func
            elif 'pDD_lmb_e_chi.pkl' in file_nam:
                pDD_chi_lmb_e = func
            elif 'V_A.pkl' in file_nam:
                V_A = func
            elif 'pDD_V_A_mu.pkl' in file_nam:
                pDD_mu_V_A = func
            elif 'pDD_V_A_chi.pkl' in file_nam:
                pDD_chi_V_A = func
            elif 'h_mu' in file_nam:
                h_mu = func
            elif 'h_chi' in file_nam:
                h_chi = func
            elif 'h_phi.pkl' in file_nam:
                h_phi = func
            elif 'B_dipole.pkl' in file_nam:
                B_dipole = func
            elif 'n_Op.pkl' in file_nam:
                n_Op = func
            elif 'n_Hp.pkl' in file_nam:
                n_Hp = func
            elif 'n_density.pkl' in file_nam:
                n_density = func
            elif 'meff.pkl' in file_nam:
                meff = func
            elif 'rho.pkl' in file_nam:
                rho = func
            elif 'dB_dipole_dmu' in file_nam:
                dB_dipole_dmu = func

        funcs = {'lambda_e': lmb_e,
                 'pDD_lambda_e_mu': pDD_mu_lmb_e,
                 'pDD_lambda_e_chi': pDD_chi_lmb_e,
                 'V_A': V_A,
                 'pDD_V_A_mu': pDD_mu_V_A,
                 'pDD_V_A_chi': pDD_chi_V_A,
                 'h_mu': h_mu,
                 'h_chi': h_chi,
                 'h_phi': h_phi,
                 'B_dipole': B_dipole,
                 'n_Op': n_Op,
                 'n_Hp': n_Hp,
                 'n_density': n_density,
                 'meff': meff,
                 'rho': rho,
                 'dB_dipole_dmu':dB_dipole_dmu}

        return funcs