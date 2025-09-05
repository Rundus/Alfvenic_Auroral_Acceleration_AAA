import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt





def my_lorenz(t, S, sigma, rho, beta):
    # put your code here
    x,y,z = S[0], S[1], S[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    dS = [dx, dy, dz]
    return dS

s = np.array([1, 2, 3])
dS = my_lorenz(0, s, 10, 28, 8/3)
print(dS) # Should be [10, 23, -6]

# --- Run the Solver and Plot it ---
def my_lorenz_solver(t_span, s0, sigma, rho, beta):
    p = (sigma, rho, beta)

    # Note: my_lorenz(t, S, sigma, rho, beta)
    soln = solve_ivp(fun=my_lorenz,
              t_span=t_span,
              y0=s0,
              method='RK45',
              args=p,
             rtol=1E-7,
              atol=1E-7)
    T = soln.t
    X = soln.y[0, :]
    Y = soln.y[1, :]
    Z = soln.y[2, :]
    return [T, X, Y, Z]

sigma = 10
rho = 28
beta = 8/3
t0 = 0
tf = 50
s0 = np.array([0, 1, 1.05])

[T, X, Y, Z] = my_lorenz_solver([t0, tf], s0, sigma, rho, beta)

from mpl_toolkits import mplot3d

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.plot3D(X, Y, Z)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()