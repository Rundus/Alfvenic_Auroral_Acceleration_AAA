import spaceToolsLib as stl
import sympy as sp


# --- Constructing Big Functions ---
# [1] Define all the sympy variables
mu, chi, n, z, u, w, zeta, gamma = sp.symbols('mu chi n z u w zeta gamma')

# [2] Define the Large expression you want to insert the many subexpressions
lmb_e = sp.sqrt((stl.lightSpeed ** 2) * stl.m_e * stl.ep0 / (n * stl.q0*stl.q0))

# [3] Define the sub-expressions
density = (stl.cm_to_m*stl.cm_to_m*stl.cm_to_m)*(400*stl.Re*z*sp.exp(-z/175) + 0.1 + 10*sp.sqrt(stl.Re/(400*z)) + 100*z*sp.exp(-z/280))

# [4] Begin inserting the subexpressions
lmb_e = lmb_e.subs({n:density})

print(lmb_e)

lmb_e = lmb_e.subs({zeta:100})
print(lmb_e)



