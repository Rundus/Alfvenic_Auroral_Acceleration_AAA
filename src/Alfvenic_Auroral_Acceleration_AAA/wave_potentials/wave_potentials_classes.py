import numpy as np
from typing import Callable, Optional
import spaceToolsLib as stl
from tqdm import tqdm

class WaveFieldsClasses: # for parallel and perp only

    # ----------------------------------------------------------------------------
    # Drivers
    # ----------------------------------------------------------------------------
    def tapered_half_sine_pulse(self,Phi_0, f_0):
        """Half sine with a Hann envelope: same oscillation, no kinks at either end."""
        w = 2 * np.pi * f_0/2

        def driver(t):
            if t < 0 or t >= 1.0 / f_0: return 0.0
            return Phi_0 * np.sin(w * t) * np.sin(np.pi * f_0 * t) ** 2

        return driver

    def tapered_sine(self, Phi_0, f_0):
        """Full sine with a Hann envelope: same oscillation, no kinks at either end."""
        w = 2 * np.pi * f_0

        def driver(t):
            if t < 0 or t >= 1.0 / f_0: return 0.0
            return Phi_0 * np.sin(w * t) * np.sin(np.pi * f_0 * t) ** 2

        return driver


    def gaussian_wavepacket(self,Phi_0: float, f_0: float, n_cycles: float = 3.0) -> Callable[[float], float]:
        """Gaussian-enveloped sinusoid, zero-mean, centred at t = n_cycles/(2 f_0)."""
        omega = 2.0 * np.pi * f_0
        t_c = n_cycles / (2.0 * f_0)
        sigma = n_cycles / (4.0 * f_0)

        def drive(t: float) -> float:
            return Phi_0 * np.exp(-0.5 * ((t - t_c) / sigma) ** 2) * np.sin(omega * (t - t_c))

        return drive


    def gaussian_pulse(self,Phi_0: float,  f_0: float, t_centre: Optional[float] = None) -> Callable[[float], float]:
        """Unipolar Gaussian pulse of 1-sigma duration `width_s`.

        Useful for clean travel-time and reflection-coefficient diagnostics, since
        it has a single unambiguous extremum.  `t_centre` defaults to 4*width_s so
        the pulse starts from (numerically) zero.
        """
        t_c = 3 / (2.0 * f_0)
        sigma = 3 / (4.0 * f_0)

        def drive(t: float) -> float:
            return Phi_0 * np.exp(-0.5 * ((t - t_c) / sigma) ** 2)

        return drive

    # ----------------------------------------------------------------------------
    # Hyperbolic PDEs Solver
    # ----------------------------------------------------------------------------
    def solve_hyperbolic(self,z, s, Z, sigma_P, drive, t_end, n_out, cfl):
        """z increasing, ionosphere first.  drive(t) -> Phi_0(t).

        Returns
        -------
        t_out   : (n_out,)      snapshot times
        Phis    : (n_out, N)    electrostatic potential
        As      : (n_out, N)    parallel vector potential
        dA_dt   : (n_out, N)    dA/dt from the scheme (for E_parallel)
        Fbot    : (n_out,)      cumulative energy flux through the ionospheric face
        Ftop    : (n_out,)      cumulative energy flux through the magnetospheric face
        """
        N = z.size # number of points along geomagnetic field line
        zf = np.r_[z[0] - (z[1] - z[0]) / 2, (z[:-1] + z[1:]) / 2, z[-1] + (z[-1] - z[-2]) / 2] # [m] Finite-Volume Face altitudes
        dz = np.diff(zf) # Spatial Gradient along geomagnetic field line
        alpha, beta = s * Z, s / Z # PDE coefficient terms. alpha = VA^2/(1 + (VA/c)^2), beta = 1/(1 + (k_perp*lambda_e)^2)
        ZL, ZR = Z[:-1], Z[1:]
        zfi = zf[1:-1]  # interior faces
        Phi, A = np.zeros(N), np.zeros(N)

        def faces(q):
            sl = np.zeros(N)
            back = (q[1:-1] - q[:-2]) / (z[1:-1] - z[:-2])
            fwd = (q[2:] - q[1:-1]) / (z[2:] - z[1:-1])
            cen = (q[2:] - q[:-2]) / (z[2:] - z[:-2])

            def mm(a, b):
                """Minmod: returns whichever of the two slopes is smaller in magnitude,
                or zero if they disagree in sign (a local extremum). Arguments are
                candidate reconstruction slopes, not the Riemann invariants a/b in rhs().
                """
                return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

            sl[1:-1] = mm(mm(2 * back, 2 * fwd), cen)  # MC limiter
            return q[:-1] + sl[:-1] * (zfi - z[:-1]), q[1:] + sl[1:] * (zfi - z[1:])

        def rhs(Phi, A, t):
            PhiL, PhiR = faces(Phi)
            AL, AR = faces(A)
            a = PhiL + ZL * AL  # right-going invariant, from the left cell
            b = PhiR - ZR * AR  # left-going  invariant, from the right cell
            Af, Pf = np.empty(N + 1), np.empty(N + 1)
            Af[1:-1] = (a - b) / (ZL + ZR)  # exact Riemann solution at the face
            Pf[1:-1] = (ZR * a + ZL * b) / (ZL + ZR)

            g = stl.u0 * sigma_P * Z[0]  # = Sigma_P / Sigma_A
            Pf[0] = (Phi[0] - Z[0] * A[0]) / (1.0 + g)  # only W^- reaches this face
            Af[0] = -1 * stl.u0 * sigma_P * Pf[0]

            d = drive(t)  # matched -> transparent
            Pf[-1] = (Phi[-1] + Z[-1] * A[-1] + d) / 2.0  # only W^+ reaches this face
            Af[-1] = (Pf[-1] - d) / Z[-1]

            ### Get the flux at top/bottom of simulation for energy normalization ###
            flux_bottom = Pf[0]*Af[0]
            flux_top = Pf[-1] * Af[-1]

            return (-alpha * np.diff(Af) / dz,
                    -beta * np.diff(Pf) / dz,
                    flux_bottom,
                    flux_top)

        dt = cfl * np.min(dz / s)
        t_out = np.linspace(0.0, t_end, n_out)
        Phis, As, dA_dt = (np.zeros((n_out, N)) for _ in range(3))
        Fbot, Ftop = np.zeros(n_out), np.zeros(n_out)
        _, dA_dt[0],_,_ = rhs(Phi, A, 0.0)

        t = 0.0
        acc_bot, acc_top = 0.0, 0.0 # accumulate the flux at the top/bottom of the simulation
        for k in tqdm(range(1, n_out)):
            while t < t_out[k] - 1e-15:
                h = min(dt, t_out[k] - t)
                kp, ka,fb1,ft1 = rhs(Phi, A, t)  # SSP-RK2
                P1, A1 = Phi + h * kp, A + h * ka
                kp2, ka2, fb2,ft2 = rhs(P1, A1, t + h)
                Phi, A = 0.5 * (Phi + P1 + h * kp2), 0.5 * (A + A1 + h * ka2)
                acc_bot += 0.5 * h * (fb1 + fb2)
                acc_top += 0.5 * h * (ft1 + ft2)
                t += h

            _, dA_dt[k],_,_ = rhs(Phi, A, t)  # rate at the snapshot state
            Phis[k], As[k] = Phi, A
            Fbot[k], Ftop[k] = acc_bot, acc_top

        return t_out, Phis, As, dA_dt, Fbot, Ftop

