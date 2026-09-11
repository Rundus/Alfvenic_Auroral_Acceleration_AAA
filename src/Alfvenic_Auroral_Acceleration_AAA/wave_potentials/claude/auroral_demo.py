"""
auroral_demo.py
===============

End-to-end demonstration on a synthetic but realistic auroral field line,
plus a direct comparison against the "advect the Riemann invariants at +-s and
ignore the source terms" scheme, which is what the original code implements.

Run:  python auroral_demo.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alfven_solver import (
    C_LIGHT,
    MU0,
    AlfvenSolver,
    FieldLineMedium,
    gaussian_pulse,
    half_sine_pulse,
)

R_E = 6.371e6
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------------------
# Synthetic auroral field line
# ---------------------------------------------------------------------------
def auroral_profile(n_cells=3000, z_min=1.0e5, z_max=4.0 * R_E, lambda_perp0=4.0e3):
    """Return (medium, extras) for a plausible auroral flux tube.

    The grid is stretched so that the cell size is roughly proportional to the
    local wavelength: fine in the ionosphere where v_A is small, coarse in the
    magnetosphere.  Replace the whole function with your CDF data.
    """
    # stretched grid: uniform in log(1 + z/h)
    h = 3.0e5
    u = np.linspace(np.log(1 + z_min / h), np.log(1 + z_max / h), n_cells)
    z = h * (np.exp(u) - 1.0)

    # --- magnetic field: dipole along the field line (polar approximation) ---
    B = 6.0e-5 * (R_E / (R_E + z)) ** 3

    # --- electron density: ionospheric layers + magnetospheric tail ---
    z_km = z / 1.0e3
    anchors_km = np.array([100, 300, 1000, 2000, 4000, 8000, 15000, 25500])
    anchors_n = np.array([1e11, 5e11, 2e10, 3e9, 3e8, 5e7, 1e7, 3e6])
    n_e = np.exp(np.interp(np.log(z_km), np.log(anchors_km), np.log(anchors_n)))

    # --- mean ion mass: O+ low down, H+ above (smooth transition ~2000 km) ---
    f_H = 0.5 * (1.0 + np.tanh((z_km - 2000.0) / 800.0))
    m_i = (1.0 - f_H) * 16.0 * M_P + f_H * M_P

    rho = n_e * m_i
    V_A = B / np.sqrt(MU0 * rho)

    # --- electron inertial length ---
    omega_pe = np.sqrt(n_e * Q_E**2 / (EPS0 * M_E))
    lambda_e = C_LIGHT / omega_pe

    # --- perpendicular scale, mapped along the flux tube ---
    flux_tube_scaling = np.sqrt(B[0] / B)
    lambda_perp = lambda_perp0 * flux_tube_scaling
    k_perp = 2.0 * np.pi / lambda_perp

    m = FieldLineMedium(z, V_A, lambda_e, k_perp)
    extras = dict(B=B, n_e=n_e, m_i=m_i, lambda_perp=lambda_perp)
    return m, extras


# ---------------------------------------------------------------------------
# The original scheme, for comparison: advect w+- at +-s, no source terms
# ---------------------------------------------------------------------------
def run_uniform_characteristic(medium, sigma_P, driver, t_end, n_snapshots, cfl=0.4):
    """Reproduces the structure of the original code.

    w+ = sqrt(alpha) A + sqrt(beta) Phi   advected at +s
    w- = sqrt(alpha) A - sqrt(beta) Phi   advected at -s
    first-order upwind, boundary values of the incoming characteristic set
    algebraically each step.  No P or Q source terms.
    """
    z = medium.z
    a, b = np.sqrt(medium.alpha), np.sqrt(medium.beta)
    s = medium.s
    N = z.size
    dz_b = np.empty(N)
    dz_b[1:] = np.diff(z)
    dz_b[0] = dz_b[1]
    dz_f = np.empty(N)
    dz_f[:-1] = np.diff(z)
    dz_f[-1] = dz_f[-2]

    wp = np.zeros(N)
    wm = np.zeros(N)

    dt = cfl * np.min(np.minimum(dz_b, dz_f) / s)
    t_out = np.linspace(0.0, t_end, n_snapshots)
    Phi_out = np.empty((n_snapshots, N))
    A_out = np.empty((n_snapshots, N))

    Sig_A = medium.Sigma_A

    def apply_bcs(wp, wm, t):
        # ionosphere:  A + mu0 sigma_P Phi = 0
        cL = (1.0 / a[0] - MU0 * sigma_P / b[0]) / (1.0 / a[0] + MU0 * sigma_P / b[0])
        wp[0] = -wm[0] * cL
        # magnetosphere: A - mu0 Sigma_A Phi = -mu0 Sigma_A Phi_d(t)
        num = -2.0 * MU0 * Sig_A[-1] * driver(t) - wp[-1] * (
            1.0 / a[-1] - MU0 * Sig_A[-1] / b[-1]
        )
        den = 1.0 / a[-1] + MU0 * Sig_A[-1] / b[-1]
        wm[-1] = num / den

    def rhs(wp, wm, t):
        apply_bcs(wp, wm, t)
        dwp = np.zeros(N)
        dwm = np.zeros(N)
        dwp[1:] = -s[1:] * (wp[1:] - wp[:-1]) / dz_b[1:]
        dwm[:-1] = s[:-1] * (wm[1:] - wm[:-1]) / dz_f[:-1]
        return dwp, dwm

    t = 0.0
    Phi_out[0] = 0.0
    A_out[0] = 0.0
    for k in range(1, n_snapshots):
        target = t_out[k]
        while t < target - 1e-15:
            step = min(dt, target - t)
            k1p, k1m = rhs(wp.copy(), wm.copy(), t)
            wp2, wm2 = wp + step * k1p, wm + step * k1m
            k2p, k2m = rhs(wp2.copy(), wm2.copy(), t + step)
            wp = 0.5 * (wp + wp2 + step * k2p)
            wm = 0.5 * (wm + wm2 + step * k2m)
            t += step
        apply_bcs(wp, wm, t)
        A_out[k] = (wp + wm) / (2.0 * a)
        Phi_out[k] = (wp - wm) / (2.0 * b)
    return t_out, Phi_out, A_out


# ---------------------------------------------------------------------------
def main():
    m, extras = auroral_profile(n_cells=3000)
    print(m.summary())
    print()

    f0 = 3.0
    Phi_0 = 250.0
    sigma_P = 1.0  # S

    # resolution check: cells per wavelength at f0
    lam = m.s / f0
    ppw = lam / m.dz
    print(f"cells per wavelength at {f0} Hz:  min {ppw.min():.1f} "
          f"(at z = {m.z[np.argmin(ppw)]/1e3:.0f} km), max {ppw.max():.1f}")
    if ppw.min() < 12:
        print("  WARNING: under-resolved somewhere; refine the grid there.")
    print()

    driver = half_sine_pulse(Phi_0, f0)
    T = m.travel_time()
    t_end = 2.4 * T
    print(f"one-way transit {T:.4f} s, integrating to {t_end:.4f} s")

    sol = AlfvenSolver(m, sigma_P=sigma_P, driver=driver,
                       order=2, limiter="mc", time_integrator="rk3", cfl=0.4)
    res = sol.run(t_end, n_snapshots=400, progress=True)
    print(f"steps taken: {res.n_steps},  dt = {res.dt:.3e} s")
    # the domain is driven and open, so energy is not conserved here; what this
    # shows is how much is left inside after the wave has drained out
    print(f"residual energy at t_end: {res.energy[-1]/res.energy.max():.3e} of peak")
    print()

    # --- comparison against the source-term-free characteristic scheme -------
    print("running the uniform-medium characteristic scheme for comparison...")
    t_u, Phi_u, A_u = run_uniform_characteristic(
        m, sigma_P, driver, t_end, n_snapshots=400
    )
    peak_fv = np.abs(res.Phi).max()
    peak_u = np.abs(Phi_u).max()
    print(f"peak |Phi|:  finite volume {peak_fv:.2f} V,  "
          f"no-source characteristic {peak_u:.2f} V  "
          f"(ratio {peak_u/peak_fv:.3f})")

    # ---------------------------------------------------------------- plots --
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(m.z / R_E, m.V_A / 1e3, label=r"$V_A$")
    ax.loglog(m.z / R_E, m.s / 1e3, "--", label=r"$s=v_A/(1+k_\perp^2\lambda_e^2)^{1/2}$")
    ax.set_xlabel(r"altitude [$R_E$]")
    ax.set_ylabel("speed [km/s]")
    ax.legend(fontsize=8)
    ax.set_title("wave speed profile")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 1])
    ax.loglog(m.z / R_E, m.Sigma_A, label=r"$\Sigma_A$")
    ax.axhline(sigma_P, color="k", ls=":", label=rf"$\Sigma_P={sigma_P}$ S")
    ax.set_xlabel(r"altitude [$R_E$]")
    ax.set_ylabel("conductance [S]")
    ax.legend(fontsize=8)
    ax.set_title("Alfven conductance")
    ax.grid(alpha=0.3, which="both")

    vmax = np.abs(res.Phi).max()
    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(res.t, m.z / R_E, res.Phi.T, shading="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"altitude [$R_E$]")
    ax.set_title(r"$\Phi$ [V] -- finite volume (correct)")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(t_u, m.z / R_E, Phi_u.T, shading="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"altitude [$R_E$]")
    ax.set_title(r"$\Phi$ [V] -- characteristics, no source terms")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[2, 0])
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(frac * (len(res.t) - 1))
        ax.plot(m.z / R_E, res.Phi[k], lw=1.2, label=f"t={res.t[k]:.2f} s")
    ax.set_xscale("log")
    ax.set_xlabel(r"altitude [$R_E$]")
    ax.set_ylabel(r"$\Phi$ [V]")
    ax.set_title("snapshots (finite volume)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(res.t, res.energy / res.energy.max(), lw=1.2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("normalised energy")
    ax.set_title(r"$\frac{1}{2}\int(\Phi^2/\alpha + A^2/\beta)\,dz$")
    ax.grid(alpha=0.3)

    fig.suptitle("Dispersive Alfven wave on an auroral field line", fontsize=13)
    fig.savefig("auroral_demo.png", dpi=130, bbox_inches="tight")
    print("\nwrote auroral_demo.png")


if __name__ == "__main__":
    main()
