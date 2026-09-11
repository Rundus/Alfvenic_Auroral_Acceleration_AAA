"""Verification tests for alfven_solver.py -- each checks against an analytic result."""

import numpy as np
from alfven_solver import (
    MU0,
    AlfvenSolver,
    FieldLineMedium,
    gaussian_pulse,
    gaussian_wavepacket,
    half_sine_pulse,
)

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")


def uniform_medium(N=800, L=1.0e7, vA=1.0e6, kl=0.0):
    z = np.linspace(0.0, L, N)
    V_A = np.full(N, vA)
    lam = np.full(N, 1.0)
    kp = np.full(N, kl)
    return FieldLineMedium(z, V_A, lam, kp)


# ---------------------------------------------------------------------------
# 1. Non-reflecting top boundary: a launched pulse must leave without trace.
# ---------------------------------------------------------------------------
def test_transparent_top():
    m = uniform_medium()
    f0 = 3.0
    drive = gaussian_wavepacket(250.0, f0, n_cycles=3.0)
    # perfectly absorbing ionosphere too -> nothing should remain
    sol = AlfvenSolver(m, sigma_P=m.Sigma_A[0], driver=drive, order=2, limiter=None, cfl=0.4)
    t_cross = m.travel_time()
    res = sol.run(3.0 * t_cross, n_snapshots=80)
    peak = np.abs(res.Phi).max()
    resid = np.abs(res.Phi[-1]).max()
    ok = resid / peak < 2e-3
    report(
        "transparent top + matched ionosphere leaves no residual",
        ok,
        f"peak |Phi| = {peak:.3f} V, final residual = {resid:.3e} V "
        f"({100*resid/peak:.4f} % of peak)",
    )


# ---------------------------------------------------------------------------
# 2. Two-way travel time in a uniform medium with a perfectly reflecting
#    ionosphere (sigma_P = 0).
# ---------------------------------------------------------------------------
def test_travel_time():
    m = uniform_medium(N=1600)
    width = 0.2
    drive = gaussian_pulse(250.0, width)
    sol = AlfvenSolver(m, sigma_P=0.0, driver=drive, order=2, limiter=None, cfl=0.4)
    T = m.travel_time()
    res = sol.run(2.6 * T, n_snapshots=2000)

    t_c = 4.0 * width                 # pulse launch time
    probe = res.Phi[:, -3]
    mask = res.t > t_c + 1.4 * T      # return window only
    t_meas = res.t[np.argmax(probe * mask)]
    t_pred = t_c + 2.0 * T
    err = abs(t_meas - t_pred) / t_pred
    ok = err < 5e-3
    report(
        "two-way travel time, reflecting ionosphere",
        ok,
        f"predicted {t_pred:.5f} s, measured {t_meas:.5f} s, error {100*err:.3f} %",
    )


# ---------------------------------------------------------------------------
# 3. Reflection coefficient at a sharp impedance step:  r = (Z_L - Z_R)/(Z_L + Z_R)
#    for a wave travelling in -z from the high-altitude side.
# ---------------------------------------------------------------------------
def test_impedance_step():
    N = 4000
    L = 1.0e7
    z = np.linspace(0.0, L, N)
    ratio = 4.0
    V_A = np.where(z < 0.5 * L, 1.0e6, ratio * 1.0e6)   # low Z below, high Z above
    m = FieldLineMedium(z, V_A, np.ones(N), np.zeros(N))

    Z_L, Z_R = m.Z[0], m.Z[-1]
    r_exact = (Z_L - Z_R) / (Z_L + Z_R)

    f0 = 5.0
    drive = gaussian_wavepacket(250.0, f0, n_cycles=4.0)
    sol = AlfvenSolver(m, sigma_P=m.Sigma_A[0], driver=drive, order=2, limiter=None, cfl=0.4)

    # time for the pulse to go down to the step and back up
    t_end = 2.2 * np.trapezoid(1.0 / m.s, z)
    res = sol.run(t_end, n_snapshots=1500)

    probe_idx = N - 20
    probe = res.Phi[:, probe_idx]
    t_c = 4.0 / (2.0 * f0)
    down_to_step = np.trapezoid(1.0 / m.s[N // 2:], z[N // 2:])

    incident = np.abs(probe[res.t < t_c + 0.4 * down_to_step]).max()
    reflected = np.abs(probe[res.t > t_c + 1.4 * down_to_step]).max()
    r_meas = reflected / incident
    err = abs(r_meas - abs(r_exact)) / abs(r_exact)
    ok = err < 0.03
    report(
        "reflection coefficient at an impedance step",
        ok,
        f"|r| exact = {abs(r_exact):.4f}, measured = {r_meas:.4f}, error {100*err:.2f} %",
    )


# ---------------------------------------------------------------------------
# 4. Ionospheric reflection coefficient r = (Sigma_A - Sigma_P)/(Sigma_A + Sigma_P)
# ---------------------------------------------------------------------------
def test_ionospheric_reflection():
    m = uniform_medium(N=3000)
    Sig_A = m.Sigma_A[0]
    width = 0.15
    rows = []
    ok_all = True
    for frac in (0.0, 0.25, 0.5, 2.0):
        sigma_P = frac * Sig_A
        drive = gaussian_pulse(250.0, width)
        sol = AlfvenSolver(m, sigma_P=sigma_P, driver=drive, order=2,
                           limiter=None, cfl=0.4)
        T = m.travel_time()
        res = sol.run(2.5 * T, n_snapshots=1500)
        t_c = 4.0 * width
        probe = res.Phi[:, -20]
        launch = res.t < t_c + 0.5 * T
        ret = res.t > t_c + 1.4 * T
        inc = probe[np.argmax(np.abs(probe) * launch)]
        refl = probe[np.argmax(np.abs(probe) * ret)]
        r_meas = refl / inc
        r_exact = (Sig_A - sigma_P) / (Sig_A + sigma_P)
        err = abs(r_meas - r_exact)
        ok_all &= err < 0.03
        rows.append(
            f"Sigma_P/Sigma_A={frac:4.2f}: exact {r_exact:+.4f}, "
            f"measured {r_meas:+.4f}"
        )
    report("ionospheric reflection coefficient", ok_all, "\n       ".join(rows))


# ---------------------------------------------------------------------------
# 5. WKB amplitude transport: in a slowly varying medium a propagating wave
#    must satisfy Phi ~ sqrt(Z), i.e. conserved energy flux Phi^2/Z.
#    This is the test the source-term-free characteristic scheme fails.
# ---------------------------------------------------------------------------
def test_wkb_scaling():
    N = 6000
    L = 2.0e7
    z = np.linspace(0.0, L, N)
    # smooth 25x variation of V_A, slow compared with the wavelength
    V_A = 1.0e6 * np.exp(np.log(25.0) * z / L)
    m = FieldLineMedium(z, V_A, np.ones(N), np.zeros(N))

    f0 = 3.0
    drive = gaussian_wavepacket(250.0, f0, n_cycles=5.0)
    sol = AlfvenSolver(m, sigma_P=m.Sigma_A[0], driver=drive, order=2, limiter=None, cfl=0.4)
    T = np.trapezoid(1.0 / m.s, z)
    res = sol.run(0.98 * T, n_snapshots=900)

    # follow the peak of the downgoing packet: compare amplitude at two probes
    i1, i2 = int(0.80 * N), int(0.20 * N)
    amp1 = np.abs(res.Phi[:, i1]).max()
    amp2 = np.abs(res.Phi[:, i2]).max()
    pred = np.sqrt(m.Z[i2] / m.Z[i1])
    meas = amp2 / amp1
    err = abs(meas - pred) / pred
    ok = err < 0.05
    report(
        "WKB amplitude transport Phi ~ sqrt(Z)",
        ok,
        f"Z ratio {m.Z[i2]/m.Z[i1]:.4f}; predicted amp ratio {pred:.4f}, "
        f"measured {meas:.4f}, error {100*err:.2f} %",
    )


# ---------------------------------------------------------------------------
# 6. Energy conservation for a closed system (both boundaries reflecting,
#    no driver, initial-value problem).
# ---------------------------------------------------------------------------
def test_energy_conservation():
    N = 2000
    L = 1.0e7
    z = np.linspace(0.0, L, N)
    V_A = 1.0e6 * (1.0 + 0.5 * np.sin(2 * np.pi * z / L))
    m = FieldLineMedium(z, V_A, np.ones(N), np.zeros(N))

    Phi0 = 250.0 * np.exp(-((z - 0.5 * L) / (0.03 * L)) ** 2)
    # sigma_P = 0 (reflecting) and sigma_A_top -> 0 (also reflecting), no drive
    sol = AlfvenSolver(m, sigma_P=0.0, driver=lambda t: 0.0,
                       sigma_A_top=1e-30, order=2, limiter=None, cfl=0.4)
    sol.set_initial_conditions(Phi=Phi0, A=np.zeros(N))
    T = np.trapezoid(1.0 / m.s, z)
    res = sol.run(2.0 * T, n_snapshots=200)
    drift = abs(res.energy[-1] - res.energy[0]) / res.energy[0]
    ok = drift < 5e-3
    report(
        "energy conservation, closed domain",
        ok,
        f"relative drift over 2 transit times = {drift:.3e}",
    )


# ---------------------------------------------------------------------------
# 7. Convergence order against a smooth analytic solution (uniform medium,
#    d'Alembert).  MUSCL+minmod should show close to 2nd order.
# ---------------------------------------------------------------------------
def test_convergence():
    L = 1.0e7
    vA = 1.0e6
    sigma = 0.05 * L
    t_end = 2.0

    def exact(z, t):
        # right-going Gaussian in an unbounded uniform medium
        return 250.0 * np.exp(-((z - 0.5 * L - vA * t) / sigma) ** 2)

    rows = []
    errs = {}
    for order in (1, 2):
        e_prev, N_prev = None, None
        line = [f"order {order}:"]
        for N in (400, 800, 1600):
            z = np.linspace(0.0, L, N)
            m = FieldLineMedium(z, np.full(N, vA), np.ones(N), np.zeros(N))
            Phi0 = exact(z, 0.0)
            A0 = Phi0 / m.Z            # pure right-going wave
            sol = AlfvenSolver(m, sigma_P=m.Sigma_A[0], driver=lambda t: 0.0,
                               order=order, cfl=0.4)
            sol.set_initial_conditions(Phi=Phi0, A=A0)
            res = sol.run(t_end, n_snapshots=2)
            ref = exact(z, t_end)
            # ignore a margin near the boundaries
            sl = slice(N // 10, -N // 10)
            err = np.sqrt(np.mean((res.Phi[-1][sl] - ref[sl]) ** 2))
            if e_prev is not None:
                p = np.log(e_prev / err) / np.log(N / N_prev)
                line.append(f"N={N:5d} L2={err:.3e} rate={p:.2f}")
            else:
                line.append(f"N={N:5d} L2={err:.3e}")
            e_prev, N_prev = err, N
        errs[order] = e_prev
        rows.append("  ".join(line))
    ok = errs[2] < 0.2 * errs[1]
    report("convergence under refinement", ok, "\n       ".join(rows))


if __name__ == "__main__":
    print("=" * 78)
    print("Verification suite")
    print("=" * 78)
    for fn in (
        test_transparent_top,
        test_travel_time,
        test_impedance_step,
        test_ionospheric_reflection,
        test_wkb_scaling,
        test_energy_conservation,
        test_convergence,
    ):
        fn()
        print()
    n_ok = sum(ok for _, ok in RESULTS)
    print("=" * 78)
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    print("=" * 78)
