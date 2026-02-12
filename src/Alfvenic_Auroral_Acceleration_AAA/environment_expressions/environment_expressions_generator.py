# scale_length_Sympy_expression_Generator
# Description: Use sympy to generate analytic expressions for the AAA simulation
from timebudget import timebudget
import json
from src.Alfvenic_Auroral_Acceleration_AAA.simulation.my_imports import *

@timebudget
def environment_expressions_generator():

    # --- IMPORTS ---
    import spaceToolsLib as stl
    import sympy as sp
    import time
    import dill
    dill.settings['recurse'] = True
    from sympy import lambdify
    start_time = time.time()

    ####################################
    # --- Define the Sympy Variables ---
    ####################################
    B, mu, chi, n, z, u, w, zeta, gamma, rho, theta, R, THETA = sp.symbols('B mu chi n z u w zeta gamma rho theta R, THETA')

    #####################################
    # --- Modified Dipole coordinates ---
    #####################################
    R_coord = u/chi
    Theta_coord = sp.asin(sp.sqrt(u))
    THETA_coord = sp.sqrt(1 + 3*(sp.cos(theta))**2)
    z_coord = (u/chi-1)*stl.Re # returns altitude in kilometers
    u_coord = -0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)
    w_coord = - 2**(7/3) * (3**(-1/3))/gamma + gamma/(((2 ** (1 / 3)) * (3 ** (2 / 3)))*zeta)
    gamma_coord = (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)
    zeta_coord = ((mu/chi)**4)

    ########################################
    # --- Define the physics expressions ---
    ########################################

    # PLASMA NUMBER DENSITY
    if EnvironmentExpressionsToggles.environment_expression_dict['chaston2006']:
        n_Hp_density = (stl.cm_to_m**3)*(0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*(z)*sp.exp(-z/280)) # for z in km
        n_Op_density = (stl.cm_to_m**3)*(400*stl.Re*z*sp.exp(-z/175)) # for z in km
    elif EnvironmentExpressionsToggles.environment_expression_dict['shroeder2021']:
        n_e = (stl.cm_to_m**3)*((6E4)*sp.exp(-(z-318)/383) + (1.34E7)*(z**(-1.55)))
        n_Op_density = n_e*0.5*(1 - sp.tanh((z-2370)/1800))
        n_Hp_density = n_e - n_Op_density
    elif EnvironmentExpressionsToggles.environment_expression_dict['chaston2003_nightside']:

        # parameters for [Oxygen+, H+]
        nM = [0, 1]
        gamma_ = [0,0.5]
        nE = [2000, 400]
        Ealt = [350,800]
        wE = [1100, 1000]
        nF = [1600, 4]
        F0 = [95,95]
        alpha_ = [100,250]
        eta = [1.05,1]

        i=0
        n_mag = nM[i]*((z+stl.Re)/stl.Re)**(-gamma_[i])
        n_E = nE[i]*sp.exp(-1*(z-Ealt[i])**2/(wE[i]**2))
        n_F = nF[i]*(z-F0[i])*sp.exp(-1*((z-F0[i])/alpha_[i])**(eta[i]))
        n_Op_density = (stl.cm_to_m**3)*(n_mag+n_E+n_F)

        i = 1
        n_mag = nM[i] * ((z + stl.Re) / stl.Re) ** (-gamma_[i])
        n_E = nE[i] * sp.exp(-1 * (z - Ealt[i]) ** 2 / (wE[i] ** 2))
        n_F = nF[i] * (z - F0[i]) * sp.exp(-1 * ((z - F0[i]) / alpha_[i]) ** (eta[i]))
        n_Hp_density = (stl.cm_to_m ** 3) * (n_mag + n_E + n_F)
    elif EnvironmentExpressionsToggles.environment_expression_dict['chaston2003_cusp']:
        # parameters for [Oxygen+, H+]
        nM = [0, 1]
        gamma_ = [0, 0.5]
        nE = [2E4, 0]
        Ealt = [110, 0]
        wE = [50, 0]
        nF = [3000, 2]
        F0 = [95, 95]
        alpha_ = [13.5, 800]
        eta = [0.45, 1]

        i = 0
        n_mag = nM[i] * ((z + stl.Re) / stl.Re) ** (-gamma_[i])
        n_E = nE[i] * sp.exp(-1 * (z - Ealt[i]) ** 2 / (wE[i] ** 2))
        n_F = nF[i] * (z - F0[i]) * sp.exp(-1 * ((z - F0[i]) / alpha_[i]) ** (eta[i]))
        n_Op_density = (stl.cm_to_m ** 3) * (n_mag + n_E + n_F)

        i = 1
        n_mag = nM[i] * ((z + stl.Re) / stl.Re) ** (-gamma_[i])
        n_E = nE[i] * sp.exp(-1 * (z - Ealt[i]) ** 2 / (wE[i] ** 2))
        n_F = nF[i] * (z - F0[i]) * sp.exp(-1 * ((z - F0[i]) / alpha_[i]) ** (eta[i]))
        n_Hp_density = (stl.cm_to_m ** 3) * (n_mag + n_E + n_F)

    n_density = (n_Op_density + n_Hp_density)

    # PLASMA MASS DENSITY
    m_Op = stl.ion_dict['O+']
    m_Hp = stl.ion_dict['H+']
    rho_density = (m_Op*n_Op_density + m_Hp*n_Hp_density)

    # ELECTRON SKIN DEPTH
    lmb_e = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * (stl.q0**2)))

    # DIPOLE MAGNETIC FIELD
    B_dipole = (3.12E-5) * ((1/(1 + z/stl.Re))**3) * THETA

    # ALFVEN VELOCITY (MHD)
    V_A = (B/sp.sqrt(stl.u0*(rho))) / sp.sqrt(1 + ((B/sp.sqrt(stl.u0*(rho)))/stl.lightSpeed)**(2))

    # SCALE FACTOR - mu
    h_mu = 1/(THETA/(2*(stl.Re*stl.m_to_km)*(R**2)*sp.sqrt(sp.cos(theta))))

    # SCALE FACTOR - chi
    h_chi = 1/(THETA*(sp.sin(theta))/((stl.Re*stl.m_to_km)*(R**2)))

    # SCALE FACTOR - phi
    h_phi = (stl.Re*stl.m_to_km * (R**2)*sp.sin(theta))

    # Form the expression dictionary - loop through this to replace everything down to the two variables: (mu,chi)
    expression_dict = {
                       'B':[B, B_dipole],
                       'rho':[rho, rho_density],
                       'n':[n, n_density],
                       'R':[R, R_coord],
                       'THETA':[THETA, THETA_coord],
                       'Theta':[theta, Theta_coord],
                       'z':[z, z_coord],
                       'u':[u, u_coord],
                       'w':[w, w_coord],
                       'gamma':[gamma, gamma_coord],
                       'zeta':[zeta, zeta_coord]
    }

    ############
    # Skin Depth
    ############
    stl.prgMsg('Forming SkinDepth Formula')
    for key, item in expression_dict.items():
        lmb_e = lmb_e.subs({item[0]:item[1]})
    stl.Done(start_time)

    stl.prgMsg('Differentiating Lmb_e w.r.t. mu')
    diff_lmb_e_mu = sp.diff(lmb_e,mu)
    stl.Done(start_time)

    stl.prgMsg('Differentiating Lmb_e w.r.t. chi')
    diff_lmb_e_chi = sp.diff(lmb_e,chi)
    stl.Done(start_time)

    #################
    # Alfven Velocity
    #################
    stl.prgMsg('Forming Alfven Velocity Formula')
    for key, item in expression_dict.items():
        V_A = V_A.subs({item[0]:item[1]})
    stl.Done(start_time)

    stl.prgMsg('Differentiating V_A w.r.t. mu')
    diff_V_A_mu =sp.diff(V_A, mu)
    stl.Done(start_time)

    stl.prgMsg('Differentiating V_A w.r.t. chi')
    diff_V_A_chi = sp.diff(V_A, chi)
    stl.Done(start_time)

    ################################
    # RAY EQUATION h_mu scale factor
    ################################
    stl.prgMsg('Forming h_mu Scale Formula')
    for key, item in expression_dict.items():
        h_mu = h_mu.subs({item[0]:item[1]})
    stl.Done(start_time)

    #################################
    # RAY EQUATION h_chi scale factor
    #################################
    stl.prgMsg('Forming h_chi Scale Formula')
    for key, item in expression_dict.items():
        h_chi = h_chi.subs({item[0]:item[1]})
    stl.Done(start_time)

    #################################
    # RAY EQUATION h_phi scale factor
    #################################
    stl.prgMsg('Forming h_phi Scale Formula')
    for key, item in expression_dict.items():
        h_phi = h_phi.subs({item[0]:item[1]})
    stl.Done(start_time)

    #############################
    # EXTRA ENVIRONMENT FUNCTIONS
    #############################

    # B-Dipole
    B_dipole_function = B_dipole
    for key, item in expression_dict.items():
        B_dipole_function = B_dipole_function.subs({item[0]:item[1]})

    # dB-Dipole/dMu
    dB_dipole_dmu_function = sp.diff(B_dipole_function, mu)

    # n_Op
    n_Op_function = n_Op_density
    for key, item in expression_dict.items():
        n_Op_function = n_Op_function.subs({item[0]:item[1]})

    # n_Hp
    n_Hp_function = n_Hp_density
    for key, item in expression_dict.items():
        n_Hp_function = n_Hp_function.subs({item[0]:item[1]})

    # total density
    n_total_function = n_Op_density + n_Hp_density
    for key, item in expression_dict.items():
        n_total_function = n_total_function.subs({item[0]:item[1]})

    # weighted ion mass
    rho_function = m_Op*n_Op_function + m_Hp*n_Hp_function
    m_eff_function = rho_function/(m_Op + m_Hp)

    ###############################################
    # --- CONVERT EVERYTHING TO LAMBDA FUNCTION ---
    ###############################################
    func_B_dipole = lambdify([mu, chi], B_dipole_function, modules="numpy")
    func_nOp = lambdify([mu, chi], n_Op_function, modules="numpy")
    func_nHp = lambdify([mu, chi], n_Hp_function, modules="numpy")
    func_n_density = lambdify([mu, chi], n_total_function, modules="numpy")
    func_meff = lambdify([mu, chi], m_eff_function, modules="numpy")
    func_rho = lambdify([mu, chi], rho_function, modules="numpy")
    func_lmb_e = lambdify([mu, chi], lmb_e, modules="numpy")
    func_pDD_mu_lmb_e = lambdify([mu, chi], diff_lmb_e_mu, modules="numpy")
    func_pDD_chi_lmb_e = lambdify([mu, chi], diff_lmb_e_chi, modules="numpy")
    func_V_A = lambdify([mu, chi], V_A, modules="numpy")
    func_pDD_mu_V_A = lambdify([mu, chi], diff_V_A_mu, modules="numpy")
    func_pDD_chi_V_A = lambdify([mu, chi], diff_V_A_chi, modules="numpy")
    func_pDD_mu_Bgeo = lambdify([mu, chi], dB_dipole_dmu_function, modules="numpy")
    func_h_mu = lambdify([mu, chi], h_mu, modules="numpy")
    func_h_chi = lambdify([mu, chi], h_chi, modules="numpy")
    func_h_phi = lambdify([mu, chi], h_phi, modules="numpy")
    funcs = {'lmb_e': func_lmb_e,
             'pDD_lmb_e_mu': func_pDD_mu_lmb_e,
             'pDD_lmb_e_chi': func_pDD_chi_lmb_e,
             'V_A': func_V_A,
             'pDD_V_A_mu': func_pDD_mu_V_A,
             'pDD_V_A_chi': func_pDD_chi_V_A,
             'h_mu': func_h_mu,
             'h_chi': func_h_chi,
             'h_phi': func_h_phi,
             'B_dipole':func_B_dipole,
             'n_Op':func_nOp,
             'n_Hp':func_nHp,
             'n_density':func_n_density,
             'meff':func_meff,
             'rho':func_rho,
             'dB_dipole_dmu':func_pDD_mu_Bgeo}

    ###################
    # PICKLE EVERYTHING
    ###################
    folder = rf'{SimToggles.sim_root_path}/environment_expressions/pickled_expressions/'
    for key, funct in funcs.items():
        file = open(folder+f'{key}.pkl','wb')
        dill.dump(funct, file)
        file.close()


    #######################################
    # --- INDICATE WHICH MODEL WAS USED ---
    #######################################
    # create a JSON file that specifies which density model was used to generate the pickle files
    which_density_model = [key for key in EnvironmentExpressionsToggles.environment_expression_dict.keys() if EnvironmentExpressionsToggles.environment_expression_dict[key]][0]
    config_dict = {'density_model':which_density_model}
    folder_path = f'{SimToggles.sim_root_path}/environment_expressions/pickled_expressions/'
    outpath = f'{folder_path}/model_config.json'
    with open(outpath, 'w') as outfile:
        json.dump(config_dict, outfile, indent=3)
