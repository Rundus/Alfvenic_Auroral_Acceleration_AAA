


class ScaleLengthClasses:

    def loadPickleFunctions(self):
        from glob import glob
        import dill

        pickle_files = glob(r'C:\Users\cfelt\PycharmProjects\Alfvenic_Auroral_Acceleration_AAA\src\Alfvenic_Auroral_Acceleration_AAA\ray_equations\pickled_expressions\*.pkl*')
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
            elif 'scale_dkpara' in file_nam:
                scale_dkpara = func
            elif 'scale_dkperp' in file_nam:
                scale_dkperp = func
            elif 'scale_dmu.pkl' in file_nam:
                scale_dmu = func
            elif 'scale_dchi.pkl' in file_nam:
                scale_dchi = func
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

        funcs = {'lambda_e': lmb_e,
                 'pDD_lambda_e_mu': pDD_mu_lmb_e,
                 'pDD_lambda_e_chi': pDD_chi_lmb_e,
                 'V_A': V_A,
                 'pDD_V_A_mu': pDD_mu_V_A,
                 'pDD_V_A_chi': pDD_chi_V_A,
                 'scale_dkpara': scale_dkpara,
                 'scale_dkperp': scale_dkperp,
                 'scale_dmu': scale_dmu,
                 'scale_dchi': scale_dchi,
                 'B_dipole': B_dipole,
                 'n_Op': n_Op,
                 'n_Hp': n_Hp,
                 'n_density': n_density,
                 'meff': meff,
                 'rho': rho}

        return funcs