"""
alfven_solver.py
================

Finite-volume solver for the field-aligned dispersive (inertial) Alfven wave
system

    (1)   dPhi/dt  +  alpha(z) dA/dz  =  0,     alpha = V_A^2 / (1 + V_A^2/c^2) = v_A^2
    (2)   dA/dt    +  beta(z)  dPhi/dz =  0,    beta  = 1 / (1 + k_perp^2 lambda_e^2)

with boundary conditions

    magnetosphere (z = z_max):   A - mu0 Sigma_A Phi = -mu0 Sigma_A Phi_drive(t)
    ionosphere    (z = z_min):   A + mu0 Sigma_P Phi = 0

    Sigma_A = 1 / (mu0 v_A (1 + k_perp^2 lambda_e^2)^{1/2}),
    v_A     = V_A / (1 + V_A^2/c^2)^{1/2}.

Everything is SI.  z is the field-aligned coordinate, increasing from the
ionosphere (z[0]) to the magnetosphere (z[-1]).

Method
------
The system is the telegrapher's equation for a transmission line with
position-dependent inductance and capacitance per unit length:

    (1/alpha) dPhi/dt + dA/dz = 0            "C dV/dt + dI/dz = 0"
    (1/beta)  dA/dt  + dPhi/dz = 0           "L dI/dt + dV/dz = 0"

so it is in conservation form with conserved variables q = (Phi/alpha, A/beta)
and flux F = (A, Phi).  We solve it with a Godunov finite-volume scheme using
the *exact* Riemann solver for a jump in material properties across each cell
face.  This is the standard treatment of acoustics in heterogeneous media
(LeVeque, "Finite Volume Methods for Hyperbolic Problems", ch. 9).

Two quantities fully parameterise the medium:

    s = sqrt(alpha*beta) = v_A / (1 + k_perp^2 lambda_e^2)^{1/2}   (wave speed)
    Z = sqrt(alpha/beta) = v_A (1 + k_perp^2 lambda_e^2)^{1/2}     (impedance)

and note Z = 1/(mu0 Sigma_A) exactly, so alpha = s*Z and beta = s/Z.

Why not a characteristic / method-of-lines formulation
------------------------------------------------------
Writing w+- = sqrt(alpha) A +- sqrt(beta) Phi and advecting them at +-s is only
valid for a *uniform* medium.  With alpha = alpha(z), beta = beta(z) the exact
equations pick up source terms:

    dw+/dt + s dw+/dz = +P w+ + Q w-
    dw-/dt - s dw-/dz = -P w- - Q w+

    P = s'/2,      Q = (s/2) d(ln Z)/dz.

P is WKB amplitude transport; Q couples the two characteristics and is what
produces partial reflection off the Alfven-speed gradient (the ionospheric
Alfven resonator).  Dropping them is a leading-order error on an auroral field
line, where Z varies by several orders of magnitude.  The finite-volume scheme
below reproduces both effects automatically, because the Riemann solver at each
face sees the local impedance mismatch.

Author's note: the flux-tube divergence in eq. (1) is implemented as a plain
d/dz, matching the equations as written.  If you intend
grad_par . (A b_hat) = (1/S) d(S A)/dz for a converging flux tube of area S(z),
that is a modelling change and must be added explicitly -- see notes at the end.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

MU0 = 4.0e-7 * np.pi
C_LIGHT = 299_792_458.0

__all__ = [
    "FieldLineMedium",
    "AlfvenSolver",
    "SolverResult",
    "half_sine_pulse",
    "gaussian_wavepacket",
    "gaussian_pulse",
]


# ----------------------------------------------------------------------------
# Medium
# ----------------------------------------------------------------------------
class FieldLineMedium:
    """Precomputed wave speed and impedance along the field line.

    Parameters
    ----------
    z : (N,) array
        Field-aligned coordinate of the cell centres, metres, strictly
        increasing, ionosphere first.  May be non-uniform.
    V_A : (N,) array
        Uncorrected Alfven speed B/sqrt(mu0 rho), m/s.
    lambda_e : (N,) array
        Electron inertial length c/omega_pe, m.
    k_perp : (N,) array
        Perpendicular wavenumber, 1/m.
    """

    def __init__(self, z, V_A, lambda_e, k_perp):
        z = np.asarray(z, dtype=float)
        V_A = np.asarray(V_A, dtype=float)
        lambda_e = np.asarray(lambda_e, dtype=float)
        k_perp = np.asarray(k_perp, dtype=float)

        if z.ndim != 1 or z.size < 3:
            raise ValueError("z must be a 1-D array with at least 3 points")
        if not np.all(np.diff(z) > 0):
            raise ValueError("z must be strictly increasing (ionosphere -> magnetosphere)")
        for name, arr in (("V_A", V_A), ("lambda_e", lambda_e), ("k_perp", k_perp)):
            if arr.shape != z.shape:
                raise ValueError(f"{name} must have the same shape as z")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(V_A <= 0):
            raise ValueError("V_A must be strictly positive")

        self.z = z
        self.V_A = V_A
        self.lambda_e = lambda_e
        self.k_perp = k_perp

        # relativistically corrected Alfven speed
        self.v_A = V_A / np.sqrt(1.0 + (V_A / C_LIGHT) ** 2)
        # inertial (dispersive) factor  (1 + k_perp^2 lambda_e^2)^{1/2}
        self.inertial = np.sqrt(1.0 + (k_perp * lambda_e) ** 2)

        self.s = self.v_A / self.inertial            # wave speed
        self.Z = self.v_A * self.inertial            # impedance, = 1/(mu0 Sigma_A)
        self.alpha = self.s * self.Z                 # = v_A^2
        self.beta = self.s / self.Z                  # = 1/(1 + k_perp^2 lambda_e^2)
        self.Sigma_A = 1.0 / (MU0 * self.Z)

        # ---- finite-volume geometry: treat z as cell centres, build faces ----
        zf = np.empty(z.size + 1)
        zf[1:-1] = 0.5 * (z[:-1] + z[1:])
        zf[0] = z[0] - 0.5 * (z[1] - z[0])
        zf[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
        self.z_face = zf
        self.dz = np.diff(zf)                        # (N,) cell widths

        if np.any(self.dz <= 0):
            raise ValueError("degenerate cell widths; check the z grid")

    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.z.size

    def cfl_dt(self, cfl: float = 0.4) -> float:
        """Largest stable step for the explicit scheme, cell-by-cell."""
        return float(cfl * np.min(self.dz / self.s))

    def travel_time(self) -> float:
        """One-way Alfven travel time along the line, integral dz/s."""
        return float(np.trapezoid(1.0 / self.s, self.z))

    def summary(self) -> str:
        f = self.travel_time()
        return (
            f"cells                : {self.n_cells}\n"
            f"z range              : {self.z[0]/1e3:.1f} -- {self.z[-1]/1e3:.1f} km\n"
            f"dz  min / max        : {self.dz.min()/1e3:.3f} / {self.dz.max()/1e3:.3f} km\n"
            f"s   min / max        : {self.s.min()/1e3:.1f} / {self.s.max()/1e3:.1f} km/s\n"
            f"Z   min / max        : {self.Z.min():.3e} / {self.Z.max():.3e} Ohm-ish\n"
            f"Sigma_A min / max    : {self.Sigma_A.min():.4f} / {self.Sigma_A.max():.4f} S\n"
            f"one-way travel time  : {f:.4f} s\n"
            f"CFL dt (cfl=0.4)     : {self.cfl_dt(0.4):.3e} s"
        )


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------
def half_sine_pulse(Phi_0: float, f_0: float) -> Callable[[float], float]:
    """Single positive half-sine hump of duration 1/f_0 (your current driver).

    Note this is unipolar: it carries a net DC offset, so the domain retains a
    residual potential after the pulse leaves.  Use `gaussian_wavepacket` if you
    want clean spectral content around f_0.
    """
    omega = 2.0 * np.pi * f_0

    def drive(t: float) -> float:
        if t < 0.0 or t >= 1.0 / f_0:
            return 0.0
        return Phi_0 * np.sin(0.5 * omega * t)

    return drive


def gaussian_wavepacket(Phi_0: float, f_0: float, n_cycles: float = 3.0) -> Callable[[float], float]:
    """Gaussian-enveloped sinusoid, zero-mean, centred at t = n_cycles/(2 f_0)."""
    omega = 2.0 * np.pi * f_0
    t_c = n_cycles / (2.0 * f_0)
    sigma = n_cycles / (4.0 * f_0)

    def drive(t: float) -> float:
        return Phi_0 * np.exp(-0.5 * ((t - t_c) / sigma) ** 2) * np.sin(omega * (t - t_c))

    return drive


def gaussian_pulse(Phi_0: float, width_s: float, t_centre: Optional[float] = None) -> Callable[[float], float]:
    """Unipolar Gaussian pulse of 1-sigma duration `width_s`.

    Useful for clean travel-time and reflection-coefficient diagnostics, since
    it has a single unambiguous extremum.  `t_centre` defaults to 4*width_s so
    the pulse starts from (numerically) zero.
    """
    t_c = 4.0 * width_s if t_centre is None else t_centre

    def drive(t: float) -> float:
        return Phi_0 * np.exp(-0.5 * ((t - t_c) / width_s) ** 2)

    return drive


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class SolverResult:
    t: np.ndarray            # (M,)      snapshot times
    z: np.ndarray            # (N,)      cell centres
    Phi: np.ndarray          # (M, N)    electrostatic potential, V
    A: np.ndarray            # (M, N)    parallel vector potential, Wb/m
    energy: np.ndarray       # (M,)      integral 0.5(Phi^2/alpha + A^2/beta) dz
    n_steps: int
    dt: float

    def E_parallel(self, medium: "FieldLineMedium") -> np.ndarray:
        """E_par = -dPhi/dz - dA/dt, evaluated by finite differences.

        Returned on the snapshot grid, shape (M, N).  Time differencing is
        second order in the interior of the snapshot sequence and first order
        at its ends, so use a reasonably dense `n_snapshots` if you care about
        this quantity.
        """
        dPhi_dz = np.gradient(self.Phi, medium.z, axis=1)
        dA_dt = np.gradient(self.A, self.t, axis=0)
        return -dPhi_dz - dA_dt


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
class AlfvenSolver:
    """Godunov / MUSCL finite-volume solver for the system above.

    Parameters
    ----------
    medium : FieldLineMedium
    sigma_P : float
        Height-integrated Pedersen conductance of the ionosphere, S.
        sigma_P = 0 gives a perfectly reflecting boundary (r_Phi = +1);
        sigma_P = Sigma_A(z_min) gives a perfectly absorbing one.
    driver : callable t -> float
        Phi_drive(t) in the magnetospheric boundary condition.
    sigma_A_top : float, optional
        Sigma_A used in the magnetospheric BC.  Defaults to the local matched
        value 1/(mu0 Z[-1]), which makes that boundary perfectly transparent to
        outgoing waves.  Override only if you deliberately want a mismatched
        (partially reflecting) driver.
    order : {1, 2}
        1 = Godunov upwind, 2 = MUSCL reconstruction.  Use 2.
    limiter : {'mc', 'minmod', 'vanleer', None}
        Slope limiter for order=2.  Default 'mc'.  `None` uses unlimited central
        slopes, which is the most accurate choice for this linear, shock-free
        system.  Do not use 'minmod' for long propagation distances.
    time_integrator : {'rk2', 'rk3'}
        SSP Runge-Kutta order.  'rk3' costs 50 % more per step and has markedly
        lower phase error; use it for multi-transit runs.
    cfl : float
        Courant number.  <= 1.0 for order=1, <= 0.5 for order=2 with a limiter.
        With limiter=None keep cfl <= 0.4 ('rk2') or <= 0.6 ('rk3').

    Notes on the boundary conditions
    --------------------------------
    Because Sigma_A is defined as the *local* Alfven conductance,
    mu0 Sigma_A = 1/Z, and the magnetospheric condition
    A - mu0 Sigma_A Phi = -mu0 Sigma_A Phi_drive collapses to a perfectly
    non-reflecting boundary driven by Phi_drive.  The downgoing wave it launches
    has amplitude Phi_drive/2, not Phi_drive -- the source has an internal
    impedance equal to the line impedance, so the "voltage" divides.  Set
    Phi_0 = 2*(desired amplitude) if you want a specific injected wave.
    """

    def __init__(
        self,
        medium: FieldLineMedium,
        sigma_P: float,
        driver: Callable[[float], float],
        sigma_A_top: Optional[float] = None,
        order: int = 2,
        limiter: Optional[str] = "mc",
        time_integrator: str = "rk3",
        cfl: float = 0.4,
    ):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if cfl <= 0 or cfl > (1.0 if order == 1 else 0.6):
            raise ValueError("cfl out of the stable range for this order")
        if sigma_P < 0:
            raise ValueError("sigma_P must be non-negative")
        if limiter not in (None, "mc", "minmod", "vanleer"):
            raise ValueError("limiter must be one of None, 'mc', 'minmod', 'vanleer'")
        if time_integrator not in ("rk2", "rk3"):
            raise ValueError("time_integrator must be 'rk2' or 'rk3'")

        self.m = medium
        self.sigma_P = float(sigma_P)
        self.driver = driver
        self.order = order
        self.limiter = limiter
        self.time_integrator = time_integrator
        self.cfl = cfl

        self.sigma_A_top = (
            float(sigma_A_top) if sigma_A_top is not None else float(medium.Sigma_A[-1])
        )

        # cached geometry / coefficients
        self._Z = medium.Z
        self._alpha = medium.alpha
        self._beta = medium.beta
        self._dz = medium.dz
        self._zc = medium.z
        self._zf = medium.z_face

        # inverse sums used by the Riemann solver at interior faces
        ZL, ZR = self._Z[:-1], self._Z[1:]
        self._inv_Zsum = 1.0 / (ZL + ZR)
        self._ZL, self._ZR = ZL, ZR

        # state
        self.Phi = np.zeros(medium.n_cells)
        self.A = np.zeros(medium.n_cells)

    # ------------------------------------------------------------------
    # spatial operator
    # ------------------------------------------------------------------
    @staticmethod
    def _minmod2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

    def _limited_slopes(self, q: np.ndarray) -> np.ndarray:
        """Slopes dq/dz on the (possibly non-uniform) grid.

        Cells 0 and N-1 get zero slope, so the scheme drops to first order in
        the two boundary cells.  That keeps the boundary Riemann problems
        unambiguous and costs nothing in practice.

        Limiter choice matters a lot here.  This system is linear and its
        solutions are smooth wave packets -- there are no shocks -- so a
        strongly compressive limiter is not needed, while `minmod` clips smooth
        extrema and bleeds amplitude at roughly one clip per half wavelength.
        Over a few hundred wavelengths of propagation that is fatal.  `mc` is a
        good default; `None` (unlimited central slopes) is the most accurate and
        is safe for this problem.
        """
        slope = np.zeros_like(q)
        dz_b = self._zc[1:-1] - self._zc[:-2]
        dz_f = self._zc[2:] - self._zc[1:-1]
        back = (q[1:-1] - q[:-2]) / dz_b
        fwd = (q[2:] - q[1:-1]) / dz_f
        cen = (q[2:] - q[:-2]) / (dz_b + dz_f)

        lim = self.limiter
        if lim is None:
            slope[1:-1] = cen
        elif lim == "minmod":
            slope[1:-1] = self._minmod2(back, fwd)
        elif lim == "mc":
            # monotonised central: minmod(2*back, 2*fwd, central)
            slope[1:-1] = self._minmod2(self._minmod2(2.0 * back, 2.0 * fwd), cen)
        elif lim == "vanleer":
            denom = back + fwd
            with np.errstate(divide="ignore", invalid="ignore"):
                vl = np.where(back * fwd > 0.0, 2.0 * back * fwd / denom, 0.0)
            slope[1:-1] = np.nan_to_num(vl)
        else:
            raise ValueError(f"unknown limiter {lim!r}")
        return slope

    def _face_states(self, q: np.ndarray):
        """Reconstruct q to the left and right side of every interior face.

        Returns (qL, qR) each of length N-1, for faces 1..N-1 (i.e. the face
        between cells i and i+1).
        """
        if self.order == 1:
            return q[:-1].copy(), q[1:].copy()
        slope = self._limited_slopes(q)
        zf_int = self._zf[1:-1]
        qL = q[:-1] + slope[:-1] * (zf_int - self._zc[:-1])
        qR = q[1:] + slope[1:] * (zf_int - self._zc[1:])
        return qL, qR

    def _rhs(self, Phi: np.ndarray, A: np.ndarray, t: float):
        """d(Phi)/dt, d(A)/dt from the flux-difference form."""
        N = Phi.size

        # ---- interior faces: exact Riemann solve across the impedance jump ---
        PhiL, PhiR = self._face_states(Phi)
        AL, AR = self._face_states(A)

        # right-going invariant from the left cell, left-going from the right
        a = PhiL + self._ZL * AL
        b = PhiR - self._ZR * AR

        A_star_int = (a - b) * self._inv_Zsum
        Phi_star_int = (self._ZR * a + self._ZL * b) * self._inv_Zsum

        # ---- boundary faces -------------------------------------------------
        A_star = np.empty(N + 1)
        Phi_star = np.empty(N + 1)
        A_star[1:-1] = A_star_int
        Phi_star[1:-1] = Phi_star_int

        # ionosphere (face 0): outgoing left-going invariant from cell 0,
        # closed with  A + mu0 sigma_P Phi = 0
        b0 = Phi[0] - self._Z[0] * A[0]
        g0 = MU0 * self.sigma_P * self._Z[0]          # = Sigma_P / Sigma_A(z_min)
        Phi_star[0] = b0 / (1.0 + g0)
        A_star[0] = -MU0 * self.sigma_P * Phi_star[0]

        # magnetosphere (face N): outgoing right-going invariant from cell N-1,
        # closed with  A - mu0 Sigma_A Phi = -mu0 Sigma_A Phi_drive(t)
        aN = Phi[-1] + self._Z[-1] * A[-1]
        gN = self._Z[-1] * MU0 * self.sigma_A_top     # = 1 when matched
        Phi_d = float(self.driver(t))
        Phi_star[-1] = (aN + gN * Phi_d) / (1.0 + gN)
        A_star[-1] = MU0 * self.sigma_A_top * (Phi_star[-1] - Phi_d)

        # ---- flux differences ----------------------------------------------
        dPhi_dt = -self._alpha * np.diff(A_star) / self._dz
        dA_dt = -self._beta * np.diff(Phi_star) / self._dz
        return dPhi_dt, dA_dt

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _step(self, dt: float, t: float):
        """One SSP Runge-Kutta step (Shu-Osher form)."""
        Phi0, A0 = self.Phi, self.A

        k1p, k1a = self._rhs(Phi0, A0, t)
        Phi1 = Phi0 + dt * k1p
        A1 = A0 + dt * k1a

        k2p, k2a = self._rhs(Phi1, A1, t + dt)

        if self.time_integrator == "rk2":
            self.Phi = 0.5 * (Phi0 + Phi1 + dt * k2p)
            self.A = 0.5 * (A0 + A1 + dt * k2a)
            return

        # SSP-RK3
        Phi2 = 0.75 * Phi0 + 0.25 * (Phi1 + dt * k2p)
        A2 = 0.75 * A0 + 0.25 * (A1 + dt * k2a)
        k3p, k3a = self._rhs(Phi2, A2, t + 0.5 * dt)
        self.Phi = (Phi0 + 2.0 * (Phi2 + dt * k3p)) / 3.0
        self.A = (A0 + 2.0 * (A2 + dt * k3a)) / 3.0

    def energy(self) -> float:
        """Quadratic invariant 0.5 * integral (Phi^2/alpha + A^2/beta) dz."""
        e = 0.5 * (self.Phi**2 / self._alpha + self.A**2 / self._beta)
        return float(np.sum(e * self._dz))

    def set_initial_conditions(self, Phi: Optional[np.ndarray] = None, A: Optional[np.ndarray] = None):
        if Phi is not None:
            self.Phi = np.array(Phi, dtype=float).copy()
        if A is not None:
            self.A = np.array(A, dtype=float).copy()

    def run(
        self,
        t_end: float,
        n_snapshots: int = 200,
        t_start: float = 0.0,
        progress: bool = False,
    ) -> SolverResult:
        """Integrate to `t_end`, recording `n_snapshots` evenly spaced frames.

        The step size is fixed at the CFL limit but is clipped so that snapshot
        times are hit exactly.
        """
        if t_end <= t_start:
            raise ValueError("t_end must exceed t_start")

        dt_max = self.m.cfl_dt(self.cfl)
        t_out = np.linspace(t_start, t_end, n_snapshots)

        Phi_out = np.empty((n_snapshots, self.m.n_cells))
        A_out = np.empty((n_snapshots, self.m.n_cells))
        E_out = np.empty(n_snapshots)

        Phi_out[0] = self.Phi
        A_out[0] = self.A
        E_out[0] = self.energy()

        t = t_start
        n_steps = 0
        for k in range(1, n_snapshots):
            target = t_out[k]
            while t < target - 1e-15 * max(1.0, abs(target)):
                dt = min(dt_max, target - t)
                self._step(dt, t)
                t += dt
                n_steps += 1
            Phi_out[k] = self.Phi
            A_out[k] = self.A
            E_out[k] = self.energy()
            if progress and (k % max(1, n_snapshots // 10) == 0):
                print(f"  t = {t:8.4f} s  ({100*k/(n_snapshots-1):5.1f} %)", flush=True)

        return SolverResult(
            t=t_out,
            z=self.m.z.copy(),
            Phi=Phi_out,
            A=A_out,
            energy=E_out,
            n_steps=n_steps,
            dt=dt_max,
        )
