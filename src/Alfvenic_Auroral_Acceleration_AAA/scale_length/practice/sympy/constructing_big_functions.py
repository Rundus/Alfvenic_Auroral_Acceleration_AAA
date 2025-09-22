import spaceToolsLib as stl
import sympy as sp


# --- Constructing Big Functions ---


# [1] Define the
mu, chi, n, z, u, w, zeta, gamma = sp.symbols('mu chi n z u w zeta gamma')
lmb_e = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * stl.q0*stl.q0))