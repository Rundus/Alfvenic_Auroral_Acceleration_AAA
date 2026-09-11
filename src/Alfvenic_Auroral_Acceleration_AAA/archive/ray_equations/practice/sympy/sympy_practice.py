# --- sympy_practice ---
# Description: Practice using sympy for a function with two varibles

import sympy as sp
import numpy as np


# Example 1:
# x = Symbol('x')
# y = x**2 + 1
# yprime = y.diff(x)
# print(yprime)

# Example 2: Partial differentiation
x, y = sp.symbols('x y')
# f = x**2 + y + y**3
# pDD_x = sp.diff(f,x)
# pDD_y = sp.diff(f,y)
# print(pDD_x)
# print(pDD_y)

# Example 3: Construct sympy function through substitution and build-up of several sub-functions
# a = x**2
# b = y + y**3
# f = a+b
# print(f)
# F = f.subs({x:x**3})
# print(F)
# print(sp.diff(F,y))


# Example 4: Using greek letters for symbols
mu, chi = sp.symbols('mu chi')
a = sp.sqrt(mu)
b = chi + chi**3
f = a+b
print(f)
