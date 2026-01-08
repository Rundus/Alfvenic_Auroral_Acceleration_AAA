# AAA_numerical_differentiation
# Description: Perform the numerical differentiation of the
# electron skin depth and check it with your own .cdf file containing the same

# --- IMPORTS ---
import numpy as np
import spaceToolsLib as stl

# --- TOGGLES ---

# Longest distance, furthest from equator
mu0 = -0.3233202914138799
chi0 = 0.12047


# --- DATA ---
# get the spatial environment data
data_dict_spatial = stl.loadDictFromFile(r'C:\Data\physicsModels\alfvenic_auroral_acceleration_AAA\spatial_environment\spatial_environment.cdf')


# FUNCTIONS
# def r_muChi(mu, chi):
#     '''
#     :param mu:
#         mu coordinate value
#     :param chi:
#         chi coordinate value
#     :return:
#         distance along geomagnetic field line, measure from earth's surface in [km]
#     '''
#
#     zeta = np.power(mu/chi,4)
#     c1 = 2**(7/3) * (3**(-1/3))
#     c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
#     gamma = (9*zeta + np.sp.sqrt(3) * np.sqrt(27*np.power(zeta,2) + 256*np.power(zeta,3)))**(1/3)
#     w = - c1/gamma + gamma/(c2*zeta)
#     u = -0.5*np.sqrt(w) + 0.5*np.sqrt(2/(zeta*np.sqrt(w))- w)
#
#     r = u/chi # in R_E from earth's center
#
#     z = (r-1)*stl.Re
#
#     return z

# def theta_muChi(mu,chi):
#     '''
#     :param mu:
#         mu coordinate value
#     :param chi:
#         chi coordinate value
#     :return:
#         distance along geomagnetic field line in [m]
#     '''
#     zeta = np.power(mu / chi, 4)
#     c1 = 2 ** (7 / 3) * (3 ** (-1 / 3))
#     c2 = 2 ** (1 / 3) * (3 ** (2 / 3))
#     gamma = (9 * zeta + np.sqrt(3) * np.sqrt(27 * np.power(zeta, 2) + 256 * np.power(zeta, 3))) ** (1 / 3)
#     w = - c1 / gamma + gamma / (c2 * zeta)
#     u = -0.5 * np.sqrt(w) + 0.5 * np.sqrt(2 / (zeta * np.sqrt(w)) - w)
#     return np.degrees(np.arcsin(np.sqrt(u)))
#
# def n0(z):
#     return (np.power(stl.cm_to_m,3))*400*stl.Re*z*np.exp(-z/175)
#
# def nH(z):
#     nm = 0.1 + 10*np.sqrt(stl.Re/(400*z))
#     nI = 100*z*np.exp(-z/280)
#     return (np.power(stl.cm_to_m, 3))*(nm+nI)
#
# def ntotal(z):
#     return n0(z) + nH(z)
#
# # def lambda_e(ne):
# #     return np.sqrt((stl.lightSpeed**2)*stl.m_e*stl.ep0/(ne*np.power(stl.q0,2)))
#
#
# def lambda_e(mu, chi):
#     z = r_muChi(mu, chi)
#     ne = ntotal(z)
#     return np.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (ne * np.power(stl.q0, 2)))

# --- ATTEMPT 1 : SYMPY ---
import sympy as sp
import time
mu, chi, n, z, u, w, zeta, gamma= sp.symbols('mu chi n z u w zeta gamma')

start_time = time.time()

stl.prgMsg('Define function')
f = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * stl.q0*stl.q0))
stl.Done(start_time)

stl.prgMsg('Insert 1')
f = f.subs({n:(stl.cm_to_m*stl.cm_to_m*stl.cm_to_m)*(400*stl.Re*z*sp.exp(-z/175) + 0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280))})
stl.Done(start_time)

stl.prgMsg('Insert 2')
f = f.subs({z:(u/chi-1)*stl.Re})
stl.Done(start_time)

stl.prgMsg('Insert 3')
f = f.subs({u:-0.5*sp.sqrt(w) + 0.5*sp.sqrt(2/(zeta*sp.sqrt(w))- w)})
stl.Done(start_time)

stl.prgMsg('Insert 4')
f = f.subs({w: - 2**(7/3) * (3**(-1/3))/gamma + gamma/(2 ** (1 / 3) * (3 ** (2 / 3))*zeta)})
stl.Done(start_time)

stl.prgMsg('Insert 5')
f = f.subs({gamma: (9*zeta + sp.sqrt(3) * sp.sqrt(27*(zeta**2) + 256*(zeta**3)))**(1/3)})
stl.Done(start_time)

stl.prgMsg('Insert 6')
f = f.subs({zeta: ((mu/chi)**4)})
stl.Done(start_time)
print(f)

stl.prgMsg('Differentiating w.r.t. mu')
diff_mu =sp.diff(f,mu)
stl.Done(start_time)

stl.prgMsg('Differentiating w.r.t. chi')
diff_chi = sp.diff(f,chi)
stl.Done(start_time)

stl.prgMsg('Derivatives at mu0, chi0')
from sympy import lambdify
pDD_mu = lambdify([mu,chi],diff_mu)
pDD_chi = lambdify([mu,chi],diff_chi)
print('diff mu', pDD_mu(mu0,chi0))
print('diff chi', pDD_chi(mu0,chi0))
stl.Done(start_time)