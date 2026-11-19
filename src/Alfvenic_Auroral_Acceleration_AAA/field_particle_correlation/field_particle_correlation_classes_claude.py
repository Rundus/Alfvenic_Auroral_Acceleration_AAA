"""
fpc_methods.py - field-particle correlation (FPC) of a gyrotropic distribution on a
detector-style (pitch angle, energy) grid, with every reduction used in the analysis.

Everything lives in the class FPCClasses. All methods are static or class methods, so
they need no instance and keep no state:

    from fpc_methods import FPCClasses
    res = FPCClasses.compute_fpc(f, pitch_deg, energy_eV, E_par, t)

Vendoring: copy the methods and the four class attributes (QE, ME, Q_ELECTRON,
ENERGY_UNITS) into your own class, or subclass FPCClasses. Class methods call the other
methods through `cls`, so they keep working inside a merged class or a subclass.
Dependencies: numpy; scipy is optional (sparse deposition matrix).

Array conventions (all SI inside)
---------------------------------
f        (T, M, L)  phase-space density f[t, pitch, energy], s^3 m^-6
alpha    (M,)       pitch-angle bin centres, rad, measured from the SAME b-hat as E_par
E        (L,)       energy bin centres, J, increasing (log-spaced detector channels)
Epar     (T,)       parallel electric field at the f sample times, V/m
t        (T,)       f sample times, s
q, mass             species charge (C) and mass (kg); defaults are electrons

For a single snapshot pass f[None] and a length-1 Epar.

Definitions
-----------
Point FPC, native grid (per d^3v, W m^-3 per (m/s)^3):
    C(v_par, v_perp, t) = -q (v_par^2 / 2) (df/dv_par) E_par(t)
    with v = sqrt(2E/m), v_par = v cos(alpha), v_perp = v sin(alpha) and, on the grid,
    df/dv_par = (1/v) [ 2 cos(alpha) df/d(ln E) - sin(alpha) df/d(alpha) ]
Reduced distribution (s m^-4) and reduced FPC (W m^-3 per m/s):
    F(v_par, t) = integral f d^2v_perp = 2 pi integral f v_perp dv_perp
    C(v_par, t) = -q (v_par^2 / 2) (dF/dv_par) E_par(t)
Energy transfer rate (W m^-3), checked against the moments at every time step:
    integral C dv_par = sum C dV = q n u_par E_par = j_par E_par
Time reductions over the whole record: <C> = mean over t; integral C dt by trapezoid.

Method summary
--------------
- Bin edges: midpoints in ln E (outer edges extrapolated, clipped at 0); pitch midpoints
  with the end bins extended to 0 and pi.
- Integrals (moments, sum C dV): exact per-bin weights with f constant in each bin,
  dV = 2 pi (mu_lo - mu_hi) (v_hi^3 - v_lo^3) / 3.
- Native df/dv_par: np.gradient in ln E (edge_order=2) and in alpha (pitch axis padded
  by one mirrored bin beyond 0 and pi), combined by the chain rule. Differences of f by
  default; differences of ln f optional (f > 0 only, NaN elsewhere).
- F(v_par): each bin is split into K x K equal-volume sub-points (even in v^3 and in
  mu = cos alpha), each carrying f dV / K^2, deposited into uniform v_par slabs.
  Conserves density exactly. Then one 1D derivative of F gives the reduced C.
- E_par on a finer time grid than f is averaged over each f sample's window.

Not handled: relativistic speeds (v = sqrt(2E/m) is about 2 % off at 5e7 m/s and 9 % at
1e8 m/s), counting noise or spacecraft potential (simulation output assumed).
"""
import numpy as np

__all__ = ["FPCClasses"]


class FPCClasses:
    """Field-particle correlation methods (static and class methods only)."""

    QE = 1.602176634e-19          # elementary charge, C
    ME = 9.1093837015e-31         # electron mass, kg
    Q_ELECTRON = -QE              # electron charge, C
    ENERGY_UNITS = {"eV": QE, "keV": 1e3 * QE, "J": 1.0}

    # =========================================================================
    # Grid
    # =========================================================================
    @staticmethod
    def edges_from_centers(c, log=False, lo=None, hi=None):
        """Bin edges from bin centres: midpoints, outer edges extrapolated by half a bin.
        log=True takes the midpoints in ln(c) (log-spaced channels). lo/hi clip the edges."""
        x = np.log(c) if log else np.asarray(c, float)
        mid = 0.5 * (x[1:] + x[:-1])
        e = np.concatenate([[2 * x[0] - mid[0]], mid, [2 * x[-1] - mid[-1]]])
        e = np.exp(e) if log else e
        return np.clip(e, -np.inf if lo is None else lo, np.inf if hi is None else hi)

    @classmethod
    def pitch_edges(cls, alpha):
        """Pitch-angle edges (rad): midpoints between centres, end bins extended to the
        poles so the bins tile 0..pi exactly."""
        e = cls.edges_from_centers(alpha)
        e[0], e[-1] = 0.0, np.pi
        return e

    @classmethod
    def energy_edges(cls, E):
        """Energy edges (J): midpoints in ln E, outer edges extrapolated, clipped at 0."""
        return cls.edges_from_centers(E, log=True, lo=0.0)

    @staticmethod
    def bin_volumes(alpha_edges, E_edges, mass=ME):
        """Exact velocity-space volume d^3v of each (alpha, E) bin, shape (M, L), m^3 s^-3."""
        v_e = np.sqrt(2 * E_edges / mass)
        mu_e = np.cos(alpha_edges)
        return 2 * np.pi * np.outer(mu_e[:-1] - mu_e[1:], (v_e[1:]**3 - v_e[:-1]**3) / 3)

    @staticmethod
    def native_velocity_grid(alpha, E, alpha_edges, E_edges, mass=ME):
        """(v_par, v_perp) of the native bins, for plotting C on the native grid.
        Returns centres vpar_c, vperp_c (M, L) and edge meshes vpar_e, vperp_e (M+1, L+1),
        ready for pcolormesh(vpar_e, vperp_e, C)."""
        v_c, v_e = np.sqrt(2 * E / mass), np.sqrt(2 * E_edges / mass)
        return dict(vpar_c=v_c[None, :] * np.cos(alpha)[:, None],
                    vperp_c=v_c[None, :] * np.sin(alpha)[:, None],
                    vpar_e=v_e[None, :] * np.cos(alpha_edges)[:, None],
                    vperp_e=v_e[None, :] * np.sin(alpha_edges)[:, None])

    @staticmethod
    def lowest_resolved_speed(E_edges, mass=ME):
        """Speed of the lowest energy edge: |v_par| below this is not resolved by the grid."""
        return np.sqrt(2 * E_edges[0] / mass)

    # =========================================================================
    # Moments (exact bin weights)
    # =========================================================================
    @staticmethod
    def _moment_integral(f, alpha_edges, E_edges, mass, k, j):
        """2 pi sum_bins f * integral v^(2+k) dv * integral mu^j dmu  ->  shape (T,)."""
        v_e = np.sqrt(2 * E_edges / mass)
        mu_e = np.cos(alpha_edges)
        wv = (v_e[1:]**(3 + k) - v_e[:-1]**(3 + k)) / (3 + k)
        wmu = (mu_e[:-1]**(j + 1) - mu_e[1:]**(j + 1)) / (j + 1)
        return 2 * np.pi * np.einsum("tml,m,l->t", np.nan_to_num(f), wmu, wv)

    @classmethod
    def moments(cls, f, alpha_edges, E_edges, mass=ME):
        """Density n (m^-3), u_par (m/s), T_par and T_perp (J) of a FULL distribution
        f[t, alpha, E]. Each is an array of shape (T,)."""
        I = lambda k, j: cls._moment_integral(f, alpha_edges, E_edges, mass, k, j)
        n = I(0, 0)
        u = I(1, 1) / n
        return dict(n=n, upar=u,
                    Tpar=mass * (I(2, 2) - n * u**2) / n,
                    Tperp=0.5 * mass * (I(2, 0) - I(2, 2)) / n)

    @classmethod
    def linear_moments(cls, f, alpha_edges, E_edges, mass=ME):
        """n = integral f d^3v and n u_par = integral v_par f d^3v. Linear in f, so also
        valid for delta f (which can be negative). Returns (n, nu), each shape (T,)."""
        return (cls._moment_integral(f, alpha_edges, E_edges, mass, 0, 0),
                cls._moment_integral(f, alpha_edges, E_edges, mass, 1, 1))

    # =========================================================================
    # E_par on the f time grid
    # =========================================================================
    @staticmethod
    def epar_on_f_times(Epar, t_f, t_E=None, average=True):
        """E_par (V/m) at the f sample times t_f.

        Epar   E_par samples, already sign-corrected to the b-hat used for alpha
               (e.g. multiply E_mu_obs by -1 before passing it).
        t_E    times of the E_par samples; None (or identical to t_f) = already on t_f.
        average True: mean of E_par over each f sample's window [t - dt/2, t + dt/2],
               clipped at the record ends (exact for E_par linear between samples).
               False: linear interpolation to t_f.
        """
        Epar, t_f = np.asarray(Epar, float), np.asarray(t_f, float)
        if t_E is None or (len(t_E) == len(t_f) and np.allclose(t_E, t_f)):
            return Epar
        t_E = np.asarray(t_E, float)
        if not average:
            return np.interp(t_f, t_E, Epar)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (Epar[1:] + Epar[:-1]) * np.diff(t_E))])
        dt = np.median(np.diff(t_f))
        lo = np.clip(t_f - dt / 2, t_E[0], t_E[-1])
        hi = np.clip(t_f + dt / 2, t_E[0], t_E[-1])
        return (np.interp(hi, t_E, cum) - np.interp(lo, t_E, cum)) / (hi - lo)

    # =========================================================================
    # Native-grid (point) FPC
    # =========================================================================
    @staticmethod
    def native_dfdvpar(f, alpha, E, mass=ME, use_log=False):
        """df/dv_par on the native (alpha, E) grid, shape (T, M, L).

        Pitch axis padded with one mirrored bin beyond 0 and pi (gyrotropy), so the
        alpha derivative is centred everywhere; energy derivative in ln E, edge_order=2.
        use_log=False: differences of f itself (default; required for delta f and for a
                       beam on a large background).
        use_log=True : differences of ln f, then multiplied by f (f > 0 only; NaN where
                       f <= 0). Slightly more accurate for a smooth near-thermal f.
        """
        a_pad = np.concatenate([[-alpha[0]], alpha, [2 * np.pi - alpha[-1]]])
        if use_log:
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.log(np.where(f > 0, f, np.nan))
        else:
            g = f
        g_pad = np.concatenate([g[:, :1], g, g[:, -1:]], axis=1)
        dg_da = np.gradient(g_pad, a_pad, axis=1)[:, 1:-1]
        dg_dlnE = np.gradient(g, np.log(E), axis=2, edge_order=2)
        sa, ca = np.sin(alpha)[None, :, None], np.cos(alpha)[None, :, None]
        d = (2 * ca * dg_dlnE - sa * dg_da) / np.sqrt(2 * E / mass)[None, None, :]
        return f * d if use_log else d

    @classmethod
    def fpc_native(cls, f, alpha, E, Epar, q=Q_ELECTRON, mass=ME, use_log=False):
        """Point FPC on the native grid, C(alpha, E, t) = -q E_par (v_par^2/2) df/dv_par,
        shape (T, M, L), W m^-3 per (m/s)^3. NaN where use_log=True and f <= 0."""
        vpar = np.sqrt(2 * E / mass)[None, None, :] * np.cos(alpha)[None, :, None]
        return (-q * np.asarray(Epar, float)[:, None, None] * 0.5 * vpar**2
                * cls.native_dfdvpar(f, alpha, E, mass, use_log))

    @staticmethod
    def native_energy_rate(C_native, dV):
        """sum over bins of C dV (W m^-3), shape (T,). NaNs are treated as 0."""
        return np.einsum("tml,ml->t", np.nan_to_num(C_native), dV)

    @staticmethod
    def vperp_jacobian(C_native, alpha, E, mass=ME):
        """2 pi v_perp C: the native C weighted so that integral dv_perp over it gives the
        reduced C(v_par). Not the FPC itself; for plots of where in v_perp the energy goes."""
        vperp = np.sqrt(2 * E / mass)[None, :] * np.sin(alpha)[:, None]
        return 2 * np.pi * vperp * C_native

    # =========================================================================
    # Reduced FPC (K x K sub-point deposition)
    # =========================================================================
    @classmethod
    def parallel_thermal_speed(cls, f0, alpha_edges, E_edges, mass=ME):
        """w_par = sqrt(2 T_par / m) of a single distribution f0[alpha, E], m/s."""
        return float(np.sqrt(2 * cls.moments(f0[None], alpha_edges, E_edges, mass)["Tpar"][0] / mass))

    @staticmethod
    def vpar_slab_edges(E_edges, width, mass=ME):
        """Uniform v_par slab edges of the given width (m/s), symmetric about 0 and
        reaching the top energy edge. A common choice is width = 0.1 w_par."""
        vmax = np.sqrt(2 * E_edges[-1] / mass)
        n = int(np.ceil(vmax / width))
        return np.arange(-n, n + 1) * float(width)

    @classmethod
    def make_deposition(cls, alpha_edges, E_edges, vpar_edges, mass=ME, K=16):
        """Precompute the K x K deposition of the native bins into v_par slabs.

        Each bin is split into K x K sub-points spaced evenly in v^3 and in mu = cos alpha,
        so every sub-point has the same volume and carries f dV / K^2. Build this once per
        grid and reuse it for every time step and every correlation mode.

        Returns a dict: W (n_slab x M*L matrix of sub-point counts / K^2; scipy sparse if
        available), dV (M, L), dvpar (n_slab,), vpar (slab centres), vpar_edges, K, and
        lost_fraction (share of sub-points outside the slab range; 0 if the slabs cover
        the grid).
        """
        try:
            import scipy.sparse as sparse
        except ImportError:
            sparse = None
        v_e = np.sqrt(2 * E_edges / mass)
        mu_e = np.cos(alpha_edges)
        M, L = len(alpha_edges) - 1, len(E_edges) - 1
        s = (np.arange(K) + 0.5) / K
        v3 = v_e[:-1]**3 + s[:, None] * (v_e[1:]**3 - v_e[:-1]**3)       # (K, L)
        mu = mu_e[1:] + s[:, None] * (mu_e[:-1] - mu_e[1:])                # (K, M)
        vpar = mu[:, None, :, None] * np.cbrt(v3)[None, :, None, :]        # (K, K, M, L)
        n_slab = len(vpar_edges) - 1
        slab = np.digitize(vpar, vpar_edges) - 1
        ok = (slab >= 0) & (slab < n_slab)
        bin_idx = np.broadcast_to(np.arange(M * L).reshape(M, L), vpar.shape)
        rows, cols = slab[ok], bin_idx[ok]
        if sparse is not None:
            W = sparse.csr_matrix((np.full(rows.size, 1.0 / K**2), (rows, cols)),
                                  shape=(n_slab, M * L))                  # duplicates summed
        else:
            W = np.bincount(rows * (M * L) + cols, minlength=n_slab * M * L
                            ).reshape(n_slab, M * L) / K**2
        return dict(W=W, dV=cls.bin_volumes(alpha_edges, E_edges, mass),
                    dvpar=np.diff(vpar_edges), vpar=0.5 * (vpar_edges[1:] + vpar_edges[:-1]),
                    vpar_edges=np.asarray(vpar_edges), K=K, lost_fraction=1.0 - ok.mean())

    @staticmethod
    def reduced_F(f, deposition):
        """F(v_par, t) = integral f d^2v_perp (s m^-4), shape (T, n_slab), from a
        deposition built by make_deposition. sum F dv_par = sum f dV exactly."""
        T = f.shape[0]
        fdV = (np.nan_to_num(f) * deposition["dV"][None]).reshape(T, -1)   # (T, M*L)
        return np.asarray((deposition["W"] @ fdV.T).T) / deposition["dvpar"][None, :]

    @staticmethod
    def reduced_fpc(F, vpar, Epar, q=Q_ELECTRON):
        """Reduced FPC C(v_par, t) = -q E_par (v_par^2/2) dF/dv_par, shape (T, n_slab),
        W m^-3 per (m/s). One np.gradient across the slab centres."""
        return (-q * np.asarray(Epar, float)[:, None] * 0.5 * vpar[None, :]**2
                * np.gradient(F, vpar, axis=1))

    @staticmethod
    def reduced_energy_rate(C, dvpar):
        """integral C dv_par = sum C dv_par over the slabs (W m^-3), shape (T,)."""
        return (C * dvpar[None, :]).sum(axis=1)

    # =========================================================================
    # What gets correlated, and the time reductions
    # =========================================================================
    @staticmethod
    def correlation_input(f, mode="full", delta_f=None):
        """The distribution that is correlated with E_par.
        "full"     : f itself (default).
        "deltaF"   : f - <f>_t, with the average over the whole record.
        "delta_f0" : delta_f if given (e.g. the simulation's f - f0), else f - f(t=0).
        All later steps are linear in it, so F of the result equals the same operation on F."""
        if mode == "full":
            return f
        if mode == "deltaF":
            return f - f.mean(axis=0, keepdims=True)
        if mode == "delta_f0":
            return np.asarray(delta_f, float) if delta_f is not None else f - f[:1]
        raise ValueError(f"unknown mode {mode!r}; use 'full', 'deltaF' or 'delta_f0'")

    @staticmethod
    def time_average(C):
        """<C> over the whole record (plain mean over the time axis 0)."""
        return C.mean(axis=0)

    @staticmethod
    def time_integral(C, t):
        """integral C dt over the whole record (trapezoid rule on the time axis 0)."""
        trapezoid = getattr(np, "trapezoid", None) or np.trapz       # numpy >= 2.0 / older
        return trapezoid(C, np.asarray(t, float), axis=0)

    # =========================================================================
    # Checks and diagnostics
    # =========================================================================
    @classmethod
    def jpar_epar(cls, g, alpha_edges, E_edges, Epar, q=Q_ELECTRON, mass=ME):
        """j_par E_par = q (n u_par) E_par from the exact moments of g (W m^-3), shape (T,).
        The FPC integral (reduced or native) must match this at every time step."""
        _, nu = cls.linear_moments(g, alpha_edges, E_edges, mass)
        return q * nu * np.asarray(Epar, float)

    @staticmethod
    def energy_check(C_int, jE, floor=0.01):
        """Median of C_int / jE over the time steps where |jE| exceeds floor * max|jE|.
        Should be 1 (reduced route typically within 0.5 %, native within a few %)."""
        s = np.abs(jE) > floor * np.abs(jE).max()
        return float(np.median(C_int[s] / jE[s])) if s.any() else np.nan

    @staticmethod
    def zero_crossing(vpar, C_avg, v_lo=0.0):
        """Zero crossing of a bipolar <C>(v_par) between its largest positive and largest
        negative lobes, using only v_par > v_lo; linear interpolation between slabs.
        If several crossings lie between the lobes, returns the one next to the negative
        lobe. NaN if there is none. For a multi-lobed <C> check the result by eye."""
        m = vpar > v_lo
        v, c = vpar[m], C_avg[m]
        ip, im = int(np.argmax(c)), int(np.argmin(c))
        lo, hi = sorted((ip, im))
        seg = np.where(np.sign(c[lo:hi + 1][1:]) != np.sign(c[lo:hi + 1][:-1]))[0]
        if not seg.size:
            return np.nan
        k = lo + (seg[-1] if ip < im else seg[0])
        return float(v[k] - c[k] * (v[k + 1] - v[k]) / (c[k + 1] - c[k]))

    # =========================================================================
    # Everything in one call
    # =========================================================================
    @classmethod
    def compute_fpc(cls, f, pitch_angle, energy, Epar, t, *, t_E=None, pitch_in_degrees=True,
                    energy_unit="eV", q=Q_ELECTRON, mass=ME, modes=("full",), delta_f=None,
                    K=16, slab_width="auto", vpar_edges=None, deposition=None,
                    native_gradient="linear", epar_average=True):
        """FPC of f[t, pitch, energy] with all reductions, in the same order as the analysis.

        f            (T, M, L) phase-space density, s^3 m^-6
        pitch_angle  (M,) bin centres, degrees (pitch_in_degrees=False for rad), measured
                     from the same b-hat as Epar
        energy       (L,) bin centres in energy_unit ("eV", "keV" or "J")
        Epar         E_par in V/m, sign-corrected; on t, or on t_E (finer grid)
        t            (T,) f sample times, s
        modes        any of "full", "deltaF", "delta_f0" (see correlation_input)
        delta_f      optional (T, M, L) delta f for the "delta_f0" mode
        K            sub-points per bin per axis for F (16 recommended)
        slab_width   v_par slab width in m/s, or "auto" = 0.1 w_par of f[0]
        vpar_edges   explicit slab edges (overrides slab_width)
        deposition   a make_deposition result to reuse (overrides K, slab_width and
                     vpar_edges); build it once when many calls share one grid
        native_gradient  "linear" (differences of f) or "log" (of ln f; full mode only)

        Returns a dict:
          grid      alpha, E (J), alpha_edges, E_edges, dV, vmin (lowest resolved |v_par|)
          moments   n, upar, Tpar, Tperp of the full f, each (T,)
          Epar      E_par on t
          deposition  (see make_deposition), incl. vpar, vpar_edges, dvpar
          <mode>    one dict per mode with
              F (T, S)          reduced distribution
              C (T, S)          reduced FPC, W m^-3 per (m/s)
              C_int_v (T,)      integral C dv_par (the dW_e/dt panel)
              C_tavg (S,)       <C>(v_par) over the record
              C_tint (S,)       integral C dt over the record
              jE (T,)           q n u_par E_par from the moments
              check_reduced     energy_check(C_int_v, jE)
              C_native (T,M,L)  point FPC, per d^3v (NaN -> 0)
              C_native_int (T,) sum C dV
              C_native_tint (M,L) integral C_native dt (the (v_par, v_perp) figure)
              check_native      energy_check(C_native_int, jE)
              native_dropped_bins  share of bins set to 0 (ln f with f <= 0)
              density_conservation (T,)  sum F dv_par / sum g dV (1 to round-off)
        """
        f = np.asarray(f, float)
        t = np.asarray(t, float)
        alpha = np.deg2rad(pitch_angle) if pitch_in_degrees else np.asarray(pitch_angle, float)
        E = np.asarray(energy, float) * cls.ENERGY_UNITS[energy_unit]
        if f.shape[1:] != (alpha.size, E.size):
            raise ValueError(f"f must be (T, {alpha.size}, {E.size}) = (time, pitch, energy); got {f.shape}")
        Ep = cls.epar_on_f_times(Epar, t, t_E, average=epar_average)

        a_edges, E_edges = cls.pitch_edges(alpha), cls.energy_edges(E)
        if deposition is not None:
            dep = deposition
        else:
            if vpar_edges is None:
                width = (0.1 * cls.parallel_thermal_speed(f[0], a_edges, E_edges, mass)
                         if slab_width == "auto" else float(slab_width))
                vpar_edges = cls.vpar_slab_edges(E_edges, width, mass)
            dep = cls.make_deposition(a_edges, E_edges, vpar_edges, mass, K)
        if dep["dV"].shape != (alpha.size, E.size):
            raise ValueError("deposition was built for a different (pitch, energy) grid")
        dV, vc, dvp = dep["dV"], dep["vpar"], dep["dvpar"]

        out = dict(grid=dict(alpha=alpha, E=E, alpha_edges=a_edges, E_edges=E_edges, dV=dV,
                             vmin=cls.lowest_resolved_speed(E_edges, mass)),
                   moments=cls.moments(f, a_edges, E_edges, mass), Epar=Ep, deposition=dep)
        for mode in modes:
            g = cls.correlation_input(f, mode, delta_f)
            use_log = native_gradient == "log" and mode == "full"
            F = cls.reduced_F(g, dep)
            C = cls.reduced_fpc(F, vc, Ep, q)
            jE = cls.jpar_epar(g, a_edges, E_edges, Ep, q, mass)
            Cn = cls.fpc_native(g, alpha, E, Ep, q, mass, use_log=use_log)
            valid = np.isfinite(Cn)
            Cn = np.where(valid, Cn, 0.0)
            r = dict(F=F, C=C, C_int_v=cls.reduced_energy_rate(C, dvp),
                     C_tavg=cls.time_average(C), C_tint=cls.time_integral(C, t), jE=jE,
                     C_native=Cn, C_native_int=cls.native_energy_rate(Cn, dV),
                     C_native_tint=cls.time_integral(Cn, t),
                     native_dropped_bins=float((~valid).mean()))
            r["check_reduced"] = cls.energy_check(r["C_int_v"], jE)
            r["check_native"] = cls.energy_check(r["C_native_int"], jE)
            with np.errstate(divide="ignore", invalid="ignore"):
                r["density_conservation"] = ((F * dvp).sum(axis=1)
                                             / np.einsum("tml,ml->t", np.nan_to_num(g), dV))
            out[mode] = r
        return out


# =============================================================================
# Self-test: drifting bi-Maxwellian on an 18 x 50 grid (10 eV .. 10 keV)
# =============================================================================
if __name__ == "__main__":
    P = FPCClasses
    q, m = P.Q_ELECTRON, P.ME
    n0, Tpar, Tperp, u, Ep = 5e6, 300 * P.QE, 600 * P.QE, 2.0e6, 1e-3
    wpar, wperp = np.sqrt(2 * Tpar / m), np.sqrt(2 * Tperp / m)
    pitch_deg = np.linspace(5, 175, 18)
    energy_eV = np.geomspace(10, 1e4, 50)
    v = np.sqrt(2 * energy_eV * P.QE / m)
    VPAR = v[None, :] * np.cos(np.deg2rad(pitch_deg))[:, None]
    VPERP = v[None, :] * np.sin(np.deg2rad(pitch_deg))[:, None]
    f = (n0 / (np.pi**1.5 * wpar * wperp**2)
         * np.exp(-(VPAR - u)**2 / wpar**2 - VPERP**2 / wperp**2))[None]
    res = P.compute_fpc(f, pitch_deg, energy_eV, np.array([Ep]), np.array([0.0]))
    r, mo = res["full"], res["moments"]
    vc = res["deposition"]["vpar"]
    C_an = (-q * Ep * 0.5 * vc**2 * (-2 * (vc - u) / wpar**2)
            * n0 / (np.sqrt(np.pi) * wpar) * np.exp(-(vc - u)**2 / wpar**2))

    print("self-test, drifting bi-Maxwellian (n 5 cm^-3, T_par 300 eV, T_perp 600 eV, u 2000 km/s)")

    print(f"  n {mo['n'][0] / n0 - 1:+.3%}   u_par {mo['upar'][0] / u - 1:+.3%}   "
          f"T_par {mo['Tpar'][0] / Tpar - 1:+.3%}   T_perp {mo['Tperp'][0] / Tperp - 1:+.3%}")

    print(f"  density conservation {r['density_conservation'][0]:.6f}")

    print(f"  reduced C, max error / max|C| {np.abs(r['C'][0] - C_an).max() / np.abs(C_an).max():.2%}")

    print(f"  integral C dv_par / (q n u_par E_par) {r['check_reduced']:.4f}   "
          f"sum C dV / (q n u_par E_par) {r['check_native']:.4f}")