# AAA_numerical_differentiation
# Description: Perform the numerical differentiation of the
# electron skin depth and check it with your own .cdf file containing the same

# --- IMPORTS ---
import numpy as np
import spaceToolsLib as stl
import sympy as sp
import time
from sympy import lambdify
start_time = time.time()

# --- TOGGLES ---

# Longest distance, furthest from equator
mu0 = -0.3233202914138799
chi0 = 0.12047

# --- DATA ---
# get the spatial environment data
data_dict_spatial = stl.loadDictFromFile(r'C:\Data\physicsModels\alfvenic_auroral_acceleration_AAA\spatial_environment\spatial_environment.cdf')

#####################
# ELECTRON SKIN DEPTH
#####################
stl.prgMsg('Forming SkinDepth Formula')
mu, chi, n, z, u, w, zeta, gamma = sp.symbols('mu chi n z u w zeta gamma')
lmb_e = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * stl.q0*stl.q0))
lmb_e = lmb_e.subs({n:(stl.cm_to_m*stl.cm_to_m*stl.cm_to_m)*(400*stl.Re*z*sp.exp(-z/175) + 0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280))})
lmb_e = lmb_e.subs({z:(u/chi-1)*stl.Re})
lmb_e = lmb_e.subs({u:-0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)})
lmb_e = lmb_e.subs({w: - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)})
lmb_e = lmb_e.subs({gamma: (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)})
lmb_e = lmb_e.subs({zeta: ((mu/chi)**4)})
stl.Done(start_time)

stl.prgMsg('Differentiating Lmb_e w.r.t. mu')
diff_lmb_e_mu =sp.diff(lmb_e,mu)
stl.Done(start_time)

stl.prgMsg('Differentiating Lmb_e w.r.t. chi')
diff_lmb_e_chi = sp.diff(lmb_e,chi)
stl.Done(start_time)

#################
# Alfven Velocity
#################
stl.prgMsg('Forming Alfven Velocity Formula')
B0 = (3.12E-5) # magnetic moment of earth
m_Op =stl.ion_dict['O+']
m_Hp =stl.ion_dict['H+']
B, mu, chi, n, z, u, w, zeta, gamma, rho, theta = sp.symbols('B mu chi n z u w zeta gamma rho theta')
V_A = B/sp.sqrt(stl.u0*(rho)) # define the function
V_A = V_A.subs({B:(B0/((1 + z/stl.Re)**3))*sp.sqrt(1 + 3*sp.cos(theta * (sp.pi)/180))})
V_A = V_A.subs({rho:(stl.cm_to_m*stl.cm_to_m*stl.cm_to_m)*(m_Op*400*stl.Re*z*sp.exp(-z/175) + m_Hp*(0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280)))}) # add the density
V_A = V_A.subs({z:(u/chi-1)*stl.Re})
V_A = V_A.subs({theta: sp.asin(sp.sqrt(u))})
V_A = V_A.subs({u:-0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)})
V_A = V_A.subs({w: - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)})
V_A = V_A.subs({gamma: (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)})
V_A = V_A.subs({zeta: ((mu/chi)**4)})
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
mu, chi, u, w, zeta, gamma, theta, R = sp.symbols('mu chi u w zeta gamma theta R')
scale_dkpara = sp.sqrt(1 + 3 * sp.cos(theta*(sp.pi/180)))/(2*(stl.Re*stl.m_to_km)*(R**2)*sp.sqrt(sp.cos(theta*(sp.pi/180))))
scale_dkpara = scale_dkpara.subs({R:(u/chi)})
scale_dkpara = scale_dkpara.subs({theta: sp.asin(sp.sqrt(u))})
scale_dkpara = scale_dkpara.subs({u:-0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)})
scale_dkpara = scale_dkpara.subs({w: - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)})
scale_dkpara = scale_dkpara.subs({gamma: (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)})
scale_dkpara = scale_dkpara.subs({zeta: ((mu/chi)**4)})


##############################################
# RAY EQUATION scale factor on dk_perp/dt term
##############################################
mu, chi, u, w, zeta, gamma, theta, R = sp.symbols('mu chi u w zeta gamma theta R')
scale_dkperp = (sp.sin(theta*(sp.pi)/180)*sp.sqrt(1 + 3 * sp.cos(theta*(sp.pi/180))))/((stl.Re*stl.m_to_km)*(R**2))
scale_dkperp = scale_dkperp.subs({R:(u/chi)})
scale_dkperp = scale_dkperp.subs({theta: sp.asin(sp.sqrt(u))})
scale_dkperp = scale_dkperp.subs({u:-0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)})
scale_dkperp = scale_dkperp.subs({w: - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)})
scale_dkperp = scale_dkperp.subs({gamma: (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)})
scale_dkperp = scale_dkperp.subs({zeta: ((mu/chi)**4)})


##########################################
# RAY EQUATION scale factor on dmu/dt term
##########################################
mu, chi, u, w, zeta, gamma, theta, R = sp.symbols('mu chi u w zeta gamma theta R')
scale_dmu = scale_dkpara


###########################################
# RAY EQUATION scale factor on dchi/dt term
###########################################
scale_dchi = -1*scale_dkperp

##################
# PRINT EVERYTHING
##################

# lambda_e
print('Lambda_e:',end='\n')
print(lmb_e)
print('\n\n')

print('pDD_mu Lambda_e:',end='\n')
print(diff_lmb_e_mu)
print('\n\n')

print('pDD_chi Lambda_e:',end='\n')
print(diff_lmb_e_chi)
print('\n\n')

# V_A
print('V_A:',end='\n')
print(V_A)
print('\n\n')

print('pDD_mu V_A:',end='\n')
print(diff_V_A_mu)
print('\n\n')

print('pDD_chi V_A:',end='\n')
print(diff_V_A_chi)
print('\n\n')

print('scale dk_para/dt Term:',end='\n')
print(scale_dkpara)
print('\n\n')

print('scale dk_perp/dt Term:',end='\n')
print(scale_dkperp)
print('\n\n')

print('scale dmu/dt Term:',end='\n')
print(scale_dmu)
print('\n\n')

print('scale dchi/dt Term:',end='\n')
print(scale_dchi)
print('\n\n')

