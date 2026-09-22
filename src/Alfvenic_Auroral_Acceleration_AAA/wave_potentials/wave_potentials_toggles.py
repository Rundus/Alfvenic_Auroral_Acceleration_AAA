
class WavePotentialsToggles:

    # ====================
    # === WAVE TOGGLES ===
    # ====================
    # Initial Electric Wave Field Strength - At the initial position
    Phi_0 = -1*2*150  # Amplitude of the potential pulse in the perpendicular direction [in Volts]. Note: The 2* comes
    # from the conversion between a CHARACTERSITIC and ACTUAL potential. On RHS boundary: Φ = Z (W⁺ − W⁻)/2
    # which we specify W⁺ =0, W⁻ = W⁻ = −Φ₀/(v_A \sqrt{1+\lambda k_{\perp}}^{2}), so Φ = Z · (0 + Φ₀/Z)/2 = Φ₀/2
    # Note: The -1* out front is to flip from parallel electric field to modified dipole coordinate electric field

    # Perpendicular Scale at The Ionosphere
    Lambda_perp0 = 4 # [km] Perpendicular scale of wave at Z_min (ionosphere). Mapping using flux tube scaling.

    # Wave Frequeuency
    f_0 = 4 # [Hz] Frequency of injected wave

    driver_dict = {
        'gaussian_pulse':0,
        'gaussian_wavepacket':0,
        'tapered_half_sine_pulse':1,
        'tapered_sine_pulse':0
    }

    # ===========================
    # === BOUNDARY CONDITIONS ===
    # ===========================
    SIGMA_P = 1 # [S] Pedersen Conductance in Ionosphere

    # =============================
    # === RK45 Time Integration ===
    # =============================
    t_start, t_end = 0.0, 5 #[seconds] Time from z_max the wave is allowed to propogate
    n_out = 600  # number of time-points to store for output
    cfl = 0.4