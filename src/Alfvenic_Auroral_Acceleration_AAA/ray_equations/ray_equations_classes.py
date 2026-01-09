import numpy as np
from src.Alfvenic_Auroral_Acceleration_AAA.environment_expressions.environment_expressions_classes import EnvironmentExpressionsClasses
from scipy.integrate import solve_ivp
from src.Alfvenic_Auroral_Acceleration_AAA.ray_equations.ray_equations_toggles import RayEquationToggles
from src.Alfvenic_Auroral_Acceleration_AAA.sim_toggles import SimToggles
envDict = EnvironmentExpressionsClasses().loadPickleFunctions()


class RayEquationsClasses:

    def __init__(self):
        self.V_A = envDict['V_A']
        self.h_mu = envDict['h_mu']
        self.h_chi = envDict['h_chi']
        self.h_phi = envDict['h_phi']
        self.lmb_e = envDict['lambda_e']
        self.pDD_mu_lmb_e = envDict['pDD_lambda_e_mu']
        self.pDD_chi_lmb_e = envDict['pDD_lambda_e_chi']
        self.pDD_mu_V_A = envDict['pDD_V_A_mu']
        self.pDD_chi_V_A = envDict['pDD_V_A_chi']
        self.B_dipole = envDict['B_dipole']
        self.B0 = self.B_dipole(RayEquationToggles.u0_w, RayEquationToggles.chi0_w)

    def calc_k_perp(self, mu, chi, k_perp_0):
        return np.sqrt((self.B_dipole(mu,chi)/self.B0))*k_perp_0

    def ray_equations_ODE(self, t, S, k_perp_0):

        # Initial Conditions
        k_mu, k_chi, k_phi, mu, chi, phi, omega = S[0], S[1], S[2], S[3], S[4], S[5], S[6]

        # Calculate the current k_perp
        k_perp = self.calc_k_perp(mu, chi, k_perp_0)

        ########################
        # --- Ray equation 1 ---
        ########################

        # k_mu
        term1 = (np.square(k_perp) * self.lmb_e(mu,chi)/(1 + np.square(k_perp*self.lmb_e(mu, chi)))) * self.pDD_mu_lmb_e(mu,chi)
        term2 = (1/self.V_A(mu,chi)) * self.pDD_mu_V_A(mu,chi)
        dk_mu = (1/self.h_mu(mu, chi)) * omega * (term1 - term2)

        # k_chi
        dk_chi = 0

        # k_phi
        dk_phi = 0

        ########################
        # --- Ray equation 2 ---
        ########################
        # dmu/dt
        dmu = 1*(1/self.h_mu(mu,chi))*(omega/k_mu)

        # dchi/dt
        dchi = 0

        # dphi/dt
        dphi = 0

        ########################
        # --- Ray Equation 3 ---
        ########################

        # domega/dt
        domega = 0

        return [dk_mu, dk_chi, dk_phi, dmu, dchi, dphi, domega]

    # --- Run the Solver and Plot it ---
    def ray_equation_RK45_solver(self, t_span, s0, k_perp_0):

        # Note: my_lorenz(t, S, sigma, rho, beta)
        soln = solve_ivp(fun=self.ray_equations_ODE,
                         t_span=SimToggles.RK45_tspan,
                         y0=s0,
                         method=RayEquationToggles.RK45_method,
                         rtol=RayEquationToggles.RK45_rtol,
                         atol=RayEquationToggles.RK45_atol,
                         t_eval=RayEquationToggles.RK45_Teval,
                         args=tuple([k_perp_0])
                         )
        T = soln.t
        K_mu = soln.y[0, :]
        K_chi = soln.y[1, :]
        K_phi = soln.y[2, :]
        Mu = soln.y[3, :]
        Chi = soln.y[4, :]
        Phi = soln.y[5,:]
        Omega = soln.y[6,:]
        print(soln.message)
        return [T, K_mu, K_chi, K_phi, Mu, Chi, Phi, Omega]
