# Dispersive Alfvén wave solver

Solver for

```
(1)  ∂Φ/∂t + α(z) ∂A/∂z = 0        α = V_A²/(1 + V_A²/c²) = v_A²
(2)  ∂A/∂t + β(z) ∂Φ/∂z = 0        β = 1/(1 + k⊥²λ_e²)
```

with `A + μ₀Σ_P Φ = 0` at the ionosphere and `A − μ₀Σ_A Φ = −μ₀Σ_A Φ_drive(t)`
at the magnetosphere.

## Files

| file | purpose |
|---|---|
| `alfven_solver.py` | the solver |
| `test_alfven_solver.py` | seven analytic verification tests, all passing |
| `auroral_demo.py` | synthetic auroral field line + comparison against your current scheme |
| `wave_potentials_generator_new.py` | drop-in replacement for your pipeline function |

## Quick start

```python
from alfven_solver import FieldLineMedium, AlfvenSolver, half_sine_pulse

medium = FieldLineMedium(z=z, V_A=V_A, lambda_e=lambda_e, k_perp=k_perp)  # all SI
solver = AlfvenSolver(medium, sigma_P=1.0, driver=half_sine_pulse(250.0, 3.0),
                      order=2, limiter="mc", time_integrator="rk3", cfl=0.4)
result = solver.run(t_end=1.46, n_snapshots=400)
# result.Phi, result.A are (n_snapshots, n_cells)
```

---

## The main problem with the current code

Your characteristic decomposition is only valid in a **uniform** medium. Writing
`w± = √α A ± √β Φ` and advecting at `±s` drops source terms that are present
whenever α and β vary with z. The exact equations are

```
∂w₊/∂t + s ∂w₊/∂z = +P w₊ + Q w₋
∂w₋/∂t − s ∂w₋/∂z = −P w₋ − Q w₊

P = ½ ds/dz                    (WKB amplitude transport)
Q = ½ s d(ln Z)/dz,   Z = √(α/β)   (coupling between the two characteristics)
```

I verified this symbolically; the residual of both equations is exactly zero.

`P` is what makes an inward-propagating wave grow as `Φ ∝ √Z`. `Q` is what
produces **partial reflection off the Alfvén-speed gradient** — i.e. the
ionospheric Alfvén resonator. On an auroral field line `Z` varies by a factor of
~45, so neither term is a small correction.

`auroral_demo.py` runs both schemes on the same profile. The corrected solver
shows gradient reflection and a multi-bounce resonator structure; the
source-term-free version shows a single pulse going down and coming back, and
underestimates peak `|Φ|` by about 40%.

## What I did instead

The system is the telegrapher's equation for a transmission line with
position-dependent L and C:

```
(1/α) ∂Φ/∂t + ∂A/∂z = 0        "C ∂V/∂t + ∂I/∂z = 0"
(1/β) ∂A/∂t + ∂Φ/∂z = 0        "L ∂I/∂t + ∂V/∂z = 0"
```

so it *is* in conservation form, with conserved variables `(Φ/α, A/β)` and flux
`(A, Φ)`. I solve it with a finite-volume Godunov scheme using the exact Riemann
solver for a material jump at each cell face (LeVeque ch. 9, acoustics in
heterogeneous media). The Riemann solver sees the local impedance mismatch, so
both source-term effects appear automatically and correctly — no source terms to
discretise, and the scheme conserves the quadratic invariant
`½∫(Φ²/α + A²/β)dz` to within truncation error.

Two numbers parameterise everything:

```
s = √(αβ) = v_A/(1 + k⊥²λ_e²)^½     wave speed
Z = √(α/β) = v_A(1 + k⊥²λ_e²)^½     impedance,  and Z = 1/(μ₀Σ_A) exactly
```

That last identity is worth noticing (see below).

## Verification results

```
[PASS] transparent top + matched ionosphere leaves no residual
       final residual = 5.4e-31 V
[PASS] two-way travel time, reflecting ionosphere        error 0.012 %
[PASS] reflection coefficient at an impedance step       |r| 0.6000 vs 0.6005
[PASS] ionospheric reflection coefficient
       Σ_P/Σ_A=0.00: exact +1.0000, measured +0.9976
       Σ_P/Σ_A=0.25: exact +0.6000, measured +0.5985
       Σ_P/Σ_A=0.50: exact +0.3333, measured +0.3325
       Σ_P/Σ_A=2.00: exact −0.3333, measured −0.3325
[PASS] WKB amplitude transport Φ ~ √Z                    error 0.10 %
[PASS] energy conservation, closed domain                drift 2.0e-03
[PASS] convergence under refinement                      rate ~1.7 (order 2)
```

---

## Two physics points about your boundary conditions

**1. Your magnetospheric boundary is perfectly transparent, and injects Φ₀/2.**

Since `Σ_A` is defined as the *local* Alfvén conductance, `μ₀Σ_A = 1/Z`, and the
coefficient `1/√α − μ₀Σ_A/√β` in your BC is **identically zero**. The condition
collapses to a non-reflecting boundary driven by `Φ_drive`. That is a good design
— outgoing waves leave cleanly — but be aware the downgoing wave it launches has
amplitude `Φ_drive/2`, not `Φ_drive`. The source has an internal impedance equal
to the line impedance, so the "voltage" divides. Set `Phi_0 = 500` if you want a
250 V wave.

**2. Your ionospheric BC reduces to the textbook reflection coefficient.**

`A + μ₀Σ_P Φ = 0` gives `r_Φ = (Σ_A − Σ_P)/(Σ_A + Σ_P)` evaluated at `z_min`.
Your algebra for this was correct. With `Σ_P = 1 S` and `Σ_A(z_min) ≈ 2 S` on the
demo profile, `r ≈ +0.32`.

---

## Bug list in the current code

**Will not run at all** — these are undefined names:

- `simMUs`, `simChis` → `simMu`, `simChi`
- `simAlts` → `simAlt`
- `lambda_e` → `data_dict_plasma['lambda_e'][0]`
- `N_alt` → never defined
- `SIGMA_P`, `SIGMA_A` → `WavePotentialsToggles.SIGMA_P`, and `Sigma_A` is an array
- `Phi0` → `WaveFieldsClasses.wave_Phi0`
- `WavePotentialsToggles.freq_driver` in `wave_potentials_classes.py` → the toggle
  is named `f_0`

**Numerics:**

- **Missing source terms** (above) — the leading-order error.
- **Boundary conditions applied to a local copy inside `rhs`.** `wp[0]` and
  `wm[-1]` are overwritten each call, but the corresponding entries of the state
  vector `U` still evolve under `d[0] = d[1]` / `d[-1] = d[-2]`, which is an
  unconstrained extrapolation rule. The BC never actually constrains the state;
  it only constrains what `rhs` sees. `solve_ivp`'s error controller is then
  policing a quantity that is not part of the solution. Use ghost cells or
  compute the boundary flux directly, as the new solver does.
- **`deltaT_stable` and `N_points_time` are computed and never used.** `solve_ivp`
  picks its own steps.
- **`rtol=1e-10, atol=1e-15` is wasted effort.** Your spatial discretisation is
  first-order upwind, so spatial error dominates by many orders of magnitude.
  You are paying for 10-digit time accuracy on top of a 2-digit space solution.
  An explicit fixed-step SSP-RK at CFL ≈ 0.4 is both faster and more accurate per
  unit cost for a hyperbolic system.
- **First-order upwind is very diffusive.** Over a few hundred wavelengths it
  will erase your pulse. Relatedly: I initially used a minmod limiter in the new
  solver and it destroyed the wave packets too, because minmod clips smooth
  extrema roughly once per half wavelength. Default is now `'mc'`; for this
  linear shock-free system `limiter=None` is more accurate and perfectly safe.
- **Python loops in `dw_plus_dz`/`dw_minus_dz`** — called thousands of times.
  Vectorised now.
- **`np.diff(z, append=z[-1])`** puts a zero in the last slot of `dz_fwd` (and
  `dz_bwd[0]`). Harmless as written but fragile.

**Units and bookkeeping:**

- `stl.m_to_km` is used to build both `lambda_perp` (from a value documented in
  km) and `z` (from `simAlts`). Those are conversions in opposite directions, so
  at most one can be right. Pin this down before trusting any output — a factor
  of 10⁶ in `dz/s` will not announce itself.
- `DAW_velocity` is set to `V_A/(1+k⊥²λ_e²)^½`, but the actual characteristic
  speed uses the **relativistically corrected** `v_A`, which is what your `alpha`
  already contains. Minor on this profile (~0.2%) but internally inconsistent,
  and it grows wherever `V_A/c` is not small.
- `DAW_velocity_eV` is declared and never filled.
- `Phi_0` is documented as the pulse amplitude but the driver is
  `Φ₀ sin(ωt/2)` for `t < 1/f` — a **unipolar half-sine**. It carries a DC
  component, so the domain retains a residual potential after the pulse leaves.
  If that is deliberate, fine; if you want clean spectral content at `f₀`, use
  `gaussian_wavepacket` instead.

---

## Things to check before you trust a production run

1. **Resolution.** You need ≳12 cells per wavelength *everywhere*, and the binding
   constraint is at the ionospheric end where `s` is smallest. At `f₀ = 3 Hz` and
   `s = 400 km/s`, λ ≈ 130 km. The solver prints a warning if your grid violates
   this; `auroral_demo.py` shows a stretched grid that satisfies it.
2. **Grid direction.** The solver requires `z` strictly increasing, ionosphere
   first. It raises if not.
3. **The flux-tube divergence.** Equation (1) has `∇∥·**A**`, which I implemented
   as a plain `∂A/∂z` to match the equations as written. If you intend
   `(1/S)∂(S A)/∂z` for a converging flux tube of area `S(z)`, that is a modelling
   change and needs to go in explicitly — it will alter the amplitude transport.
   Worth deciding deliberately rather than by default, given that you are already
   applying flux-tube scaling to `λ⊥`.
4. **`E∥`.** `result.E_parallel(medium)` computes `−∂Φ/∂z − ∂A/∂t` by finite
   differences on the snapshot grid, so use a dense `n_snapshots` if that is the
   quantity you actually care about.
