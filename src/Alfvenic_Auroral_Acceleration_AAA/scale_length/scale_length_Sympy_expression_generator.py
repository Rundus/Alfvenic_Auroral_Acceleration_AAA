# AAA_numerical_differentiation
# Description: Perform the numerical differentiation of the
# electron skin depth and check it with your own .cdf file containing the same

# --- IMPORTS ---
import numpy as np
import spaceToolsLib as stl
import sympy as sp
import time
import pickle
import dill
dill.settings['recurse'] = True
from sympy import lambdify
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
start_time = time.time()

####################################
# --- Define the Sympy Variables ---
####################################
B, mu, chi, n, z, u, w, zeta, gamma, rho, theta, R, THETA = sp.symbols('B mu chi n z u w zeta gamma rho theta R, THETA')

########################################
# --- Define the physics expressions ---
########################################

# Modified Dipole coordinates
R_coord = u/chi
Theta_coord = sp.asin(sp.sqrt(u))
THETA_coord = sp.sqrt(1 + 3*(sp.cos(theta*(sp.pi/180)))**2)
z_coord = (u/chi-1)*stl.Re
u_coord = -0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)
w_coord = - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)
gamma_coord = (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)
zeta_coord = ((mu/chi)**4)

# PLASMA NUMBER DENSITY
n_density = (stl.cm_to_m**3)*(400*stl.Re*z*sp.exp(-z/175) + 0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280))

# PLASMA MASS DENSITY
m_Op = stl.ion_dict['O+']
m_Hp = stl.ion_dict['H+']
rho_density = (stl.cm_to_m**3)*(m_Op*400*stl.Re*z*sp.exp(-z/175) + m_Hp*(0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280)))

# ELECTRON SKIN DEPTH
lmb_e = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * stl.q0*stl.q0))

# DIPOLE MAGNETIC FIELD
B_dipole = (3.12E-5) * (1/(1 + z/stl.Re)) * THETA

# ALFVEN VELOCITY (MHD)
V_A = B/sp.sqrt(stl.u0*(rho))

# Scale Factor on dk_parallel/dt
scale_dkpara = THETA/(2*(stl.Re*stl.m_to_km)*(R**2)*sp.sqrt(sp.cos(theta*(sp.pi/180))))

# Scale Factor on dk_perp/dt
scale_dkperp = THETA*(sp.sin(theta*(sp.pi/180)))/((stl.Re*stl.m_to_km)*(R**2))

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

##############################################
# RAY EQUATION scale factor on dk_para/dt term
##############################################
stl.prgMsg('Forming K_para Scale Formula')
for key, item in expression_dict.items():
    scale_dkpara = scale_dkpara.subs({item[0]:item[1]})
stl.Done(start_time)

##############################################
# RAY EQUATION scale factor on dk_perp/dt term
##############################################
stl.prgMsg('Forming K_para Scale Formula')
for key, item in expression_dict.items():
    scale_dkperp = scale_dkpara.subs({item[0]:item[1]})
stl.Done(start_time)

##########################################
# RAY EQUATION scale factor on dmu/dt term
##########################################
scale_dmu = scale_dkpara

###########################################
# RAY EQUATION scale factor on dchi/dt term
###########################################
scale_dchi = -1*scale_dkperp



###############################################
# --- CONVERT EVERYTHING TO LAMBDA FUNCTION ---
###############################################
func_lmb_e = lambdify([mu, chi], lmb_e, modules="numpy")
func_pDD_mu_lmb_e = lambdify([mu, chi], diff_lmb_e_mu, modules="numpy")
func_pDD_chi_lmb_e = lambdify([mu, chi], diff_lmb_e_chi, modules="numpy")
func_V_A = lambdify([mu, chi], V_A, modules="numpy")
func_pDD_mu_V_A = lambdify([mu, chi], diff_V_A_mu, modules="numpy")
func_pDD_chi_V_A = lambdify([mu, chi], diff_V_A_chi, modules="numpy")
func_scale_dkpara = lambdify([mu, chi], scale_dkpara, modules="numpy")
func_scale_dkperp = lambdify([mu, chi], scale_dkperp, modules="numpy")
func_scale_dmu = lambdify([mu, chi], scale_dmu, modules="numpy")
func_scale_dchi = lambdify([mu, chi], scale_dchi, modules="numpy")

funcs = {'lmb_e':func_lmb_e,
         'pDD_mu_lmb_e':func_pDD_mu_lmb_e,
         'pDD_chi_lmb_e':func_pDD_chi_lmb_e,
         'V_A':func_V_A,
         'pDD_mu_V_A':func_pDD_mu_V_A,
         'pDD_chi_V_A':func_pDD_chi_V_A,
         'scale_dkpara':func_scale_dkpara,
         'scale_dkperp':func_scale_dkperp,
         'scale_dmu':func_scale_dmu,
         'scale_dmu':func_scale_dmu}

###################
# PICKLE EVERYTHING
###################
folder = rf'{SimToggles.sim_root_path}\scale_length\pickled_expressions\\'
for key, funct in funcs.items():
    file = open(folder+f'{key}.pkl','wb')
    dill.dump(funct, file)
    file.close()


# ##################
# # PRINT EVERYTHING
# ##################
#
# # lambda_e
# print('Lambda_e:',end='\n')
# print(lmb_e)
# print('\n\n')
#
# print('pDD_mu Lambda_e:',end='\n')
# print(diff_lmb_e_mu)
# print('\n\n')
#
# print('pDD_chi Lambda_e:',end='\n')
# print(diff_lmb_e_chi)
# print('\n\n')
#
# # V_A
# print('V_A:',end='\n')
# print(V_A)
# print('\n\n')
#
# print('pDD_mu V_A:',end='\n')
# print(diff_V_A_mu)
# print('\n\n')
#
# print('pDD_chi V_A:',end='\n')
# print(diff_V_A_chi)
# print('\n\n')
#
# print('scale dk_para/dt Term:',end='\n')
# print(scale_dkpara)
# print('\n\n')
#
# print('scale dk_perp/dt Term:',end='\n')
# print(scale_dkperp)
# print('\n\n')
#
# print('scale dmu/dt Term:',end='\n')
# print(scale_dmu)
# print('\n\n')
#
# print('scale dchi/dt Term:',end='\n')
# print(scale_dchi)
# print('\n\n')

