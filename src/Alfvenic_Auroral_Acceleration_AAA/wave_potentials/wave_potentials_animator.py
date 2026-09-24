"""
wave_potentials_animator.py
============================

Reads a `data_dict_output` dict (the format produced by
`wave_potentials_generator()`) and writes a four-panel .mp4 animation:

    row 1: Phi (left axis)      and  A_para (right axis), same panel
    row 2: E_perp (left axis)   and  B_perp (right axis), same panel
    row 3: E_par    [mV/m]
    row 4: normalized total system energy vs time (not altitude -- see note
           in section 3, function `_build_energy_panel`)

Rows 1-3 share an altitude x-axis, in Earth radii. Row 4 has its own time
axis, since the quantity it shows is a single number per instant, not a
profile along the field line. The title carries Sigma_P, f_0, lambda_perp0,
Phi_0, and a running "t = ..." clock.

Every entry of `data_dict_output` is expected in the form
    {'key': [np.array(DATA), {...attrs...}]}
so the array for a given key is always `data_dict_output['key'][0]`.

Run this file directly (`python wave_potentials_animator.py`) after filling in
section 1 below.
"""

import matplotlib
matplotlib.use("Agg")  # headless: render frames without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
import spaceToolsLib as stl
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import WavePotentialsToggles

# =============================================================================
# 1. LOAD YOUR DATA HERE
# =============================================================================
# Put the dict from wave_potentials_generator() here. For example:
#
#     from wave_potentials_generator import wave_potentials_generator
#     data_dict_output = wave_potentials_generator()
#
# or, if you already saved it to a pickle:
#
#     import pickle
#     with open("data_dict_output.pkl", "rb") as f:
#         data_dict_output = pickle.load(f)
#
# or paste/build it directly. Whatever the source, it must end up as a dict
# of {key: [np.array(DATA), {attrs}]}, matching the schema this script reads
# in section 3 below.


data_dict_output = stl.loadDictFromFile(f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.cdf')

# =============================================================================
# 2. OUTPUT SETTINGS
# =============================================================================
OUTPUT_PATH = f'{RunToggles.sim_data_output_path}/wave_potentials/wave_potentials.mp4'
FPS = 20                   # display frame rate of the .mp4 (not a physical rate)
DPI = 150
FIGSIZE = (9, 12)
STRIDE = 1                  # use every Nth time snapshot; 1 = full resolution

# Altitude axis, in Earth radii. This assumes 'alt' is stored in metres -- if
# that turns out to be wrong, change ALTITUDE_SCALE (see the note at the
# bottom of this file: the units of 'alt' were never independently confirmed).
R_EARTH_M = stl.Re*stl.m_to_km
ALTITUDE_SCALE = 1.0 / R_EARTH_M
ALTITUDE_LABEL = r"Altitude [$R_E$]"
ALTITUDE_LOG = False         # log-scaled x-axis; often clearer over multi-R_E ranges

# Font sizes
TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 13


# =============================================================================
# 3. ANIMATION LOGIC -- shouldn't need to touch this
# =============================================================================
def _scalar(entry) -> float:
    """Pull a plain float out of a [array_or_value, attrs] entry."""
    return float(np.asarray(entry[0]).reshape(-1)[0])


def _array(d: dict, key: str) -> np.ndarray:
    if key not in d:
        raise KeyError(
            f"data_dict_output is missing '{key}'. See the note at the bottom "
            f"of this file -- the generator needs a small patch to save this."
        )
    return np.asarray(d[key][0])


def animate_wave_potentials(
    data_dict_output: dict,
    output_path: str,
    fps: int = 20,
    dpi: int = 150,
    figsize=(9, 12),
    altitude_scale: float = 1.0,
    altitude_label: str = "Altitude [m]",
    altitude_log: bool = False,
    stride: int = 1,
    title_fontsize: int = 16,
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
) -> str:
    """Render the four-panel animation and write it to `output_path`."""
    if not output_path.lower().endswith(".mp4"):
        raise ValueError("output_path must end in '.mp4'")

    # ---- pull out data -----------------------------------------------------
    z = _array(data_dict_output, "alt") * altitude_scale
    t = _array(data_dict_output, "time")
    Phi = _array(data_dict_output, "Phi")
    A = _array(data_dict_output, "A_para")
    E_perp = _array(data_dict_output, "E_perp") * 1e3   # V/m -> mV/m
    B_perp = _array(data_dict_output, "B_perp") * 1e9   # T   -> nT
    E_par = -1*_array(data_dict_output, "E_para") * 1e3    # V/m -> mV/m. The -1 is to convert from modified dipole to field-aligned
    E_norm = _array(data_dict_output, "system_energy_normalized")

    for name, arr in (("Phi", Phi), ("A_para", A), ("E_perp", E_perp),
                      ("B_perp", B_perp), ("E_para", E_par)):
        if arr.shape != (t.size, z.size):
            raise ValueError(
                f"'{name}' has shape {arr.shape}, expected {(t.size, z.size)} "
                f"(n_time, n_alt). Check that it is indexed [time, altitude]."
            )
    if E_norm.shape != t.shape:
        raise ValueError(
            f"'system_energy_normalized' has shape {E_norm.shape}, expected "
            f"{t.shape} (n_time,). It should be one value per time snapshot."
        )

    t_full = t.copy()          # full-resolution time axis, for the energy curve
    E_norm_full = E_norm.copy()

    t = t[::stride]
    Phi, A, E_perp, B_perp, E_par = (arr[::stride] for arr in (Phi, A, E_perp, B_perp, E_par))
    E_norm = E_norm[::stride]
    n_frames = t.size

    # ---- title parameters ---------------------------------------------------
    sigma_P = WavePotentialsToggles.SIGMA_P
    f_0 = WavePotentialsToggles.f_0
    lambda_0 = WavePotentialsToggles.Lambda_perp0
    phi_0 = WavePotentialsToggles.Phi_0/2

    for label, val in (("SIGMA_P", sigma_P), ("f0", f_0),
                       ("Lambda_perp0", lambda_0), ("Phi_0", phi_0)):
        if val is None:
            raise KeyError(
                f"data_dict_output is missing '{label}', needed for the title. "
                f"See the note at the bottom of this file."
            )

    def suptitle(time_val: float) -> str:
        return (
            rf"$\Sigma_P$={sigma_P:g} S   $f_0$={f_0:g} Hz   "
            rf"$\lambda_{{\perp 0}}$={lambda_0:g} km   $\Phi_0$={phi_0:g} V"
            "\n"
            rf"t = {time_val:.4f} s"
        )

    # ---- figure setup ---------------------------------------------------
    # Rows 1-3 are profiles vs altitude and share that x-axis. Row 4 (energy)
    # is a function of time, not altitude, so it gets an independent x-axis --
    # it is built with its own add_subplot call, without sharex.
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, hspace=0.2)
    ax_wave = fig.add_subplot(gs[0])
    ax_EB = fig.add_subplot(gs[1], sharex=ax_wave)
    ax_Epar = fig.add_subplot(gs[2], sharex=ax_wave)
    ax_energy = fig.add_subplot(gs[3])   # independent x-axis: time, not altitude

    ax_A = ax_wave.twinx()
    ax_Bperp = ax_EB.twinx()

    fig.suptitle(suptitle(t[0]), fontsize=title_fontsize)

    def pad(lo, hi, frac=0.08):
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        return lo - frac * span, hi + frac * span

    phi_lo, phi_hi = pad(Phi.min(), Phi.max())
    A_lo, A_hi = pad(A.min(), A.max())
    Eperp_lo, Eperp_hi = pad(E_perp.min(), E_perp.max())
    Bperp_lo, Bperp_hi = pad(B_perp.min(), B_perp.max())
    Epar_lo, Epar_hi = pad(E_par.min(), E_par.max())

    # ---- row 1: Phi / A_para --------------------------------------------
    (line_phi,) = ax_wave.plot([], [], color="tab:blue", lw=1.4, label=r"$\Phi$")
    (line_A,) = ax_A.plot([], [], color="tab:red", lw=1.4, label=r"$A_\parallel$")
    ax_wave.set_ylabel(r"$\Phi$ [V]", color="tab:blue", fontsize=label_fontsize)
    ax_A.set_ylabel(r"$A_\parallel$ [Wb/m]", color="tab:red", fontsize=label_fontsize)
    ax_wave.tick_params(axis="y", labelcolor="tab:blue", labelsize=tick_fontsize)
    ax_A.tick_params(axis="y", labelcolor="tab:red", labelsize=tick_fontsize)
    ax_wave.set_ylim(phi_lo, phi_hi)
    ax_A.set_ylim(A_lo, A_hi)
    ax_wave.legend(handles=[line_phi, line_A], loc="upper right",
                   fontsize=tick_fontsize, framealpha=0.85)

    # ---- row 2: E_perp / B_perp -------------------------------------------
    (line_Eperp,) = ax_EB.plot([], [], color="tab:green", lw=1.4, label=r"$E_\perp$")
    (line_Bperp,) = ax_Bperp.plot([], [], color="tab:purple", lw=1.4, label=r"$B_\perp$")
    ax_EB.set_ylabel(r"$E_\perp$ [mV/m]", color="tab:green", fontsize=label_fontsize)
    ax_Bperp.set_ylabel(r"$B_\perp$ [nT]", color="tab:purple", fontsize=label_fontsize)
    ax_EB.tick_params(axis="y", labelcolor="tab:green", labelsize=tick_fontsize)
    ax_Bperp.tick_params(axis="y", labelcolor="tab:purple", labelsize=tick_fontsize)
    ax_EB.set_ylim(Eperp_lo, Eperp_hi)
    ax_Bperp.set_ylim(Bperp_lo, Bperp_hi)
    ax_EB.legend(handles=[line_Eperp, line_Bperp], loc="upper right",
                fontsize=tick_fontsize, framealpha=0.85)

    # ---- row 3: E_par --------------------------------------------------
    (line_Epar,) = ax_Epar.plot([], [], color="tab:orange", lw=1.4)
    ax_Epar.set_ylabel(r"$E_\parallel$ [mV/m]", fontsize=label_fontsize)
    ax_Epar.set_ylim(Epar_lo, Epar_hi)
    ax_Epar.set_xlabel(altitude_label, fontsize=label_fontsize)
    ax_Epar.tick_params(axis="both", labelsize=tick_fontsize)

    if altitude_log:
        ax_Epar.set_xscale("log")
    ax_Epar.set_xlim(z.min(), z.max())
    for ax in (ax_wave, ax_EB):
        ax.tick_params(axis="x", labelbottom=False)  # shared x: labels only on bottom row

    # ---- row 4: normalized total energy vs TIME (not altitude) -----------
    # This is a single scalar per instant (integrated over the whole field
    # line), so it cannot share the altitude axis above. The full curve is
    # drawn once; a marker tracks the current frame along it.
    ax_energy.plot(t_full, E_norm_full, color="0.6", lw=1.0, zorder=1)
    (energy_marker,) = ax_energy.plot([t[0]], [E_norm[0]], "o", color="tab:red",
                                      ms=6, zorder=2)
    ax_energy.set_xlabel("Time [s]", fontsize=label_fontsize)
    ax_energy.set_ylabel("Normalized\nsystem energy", fontsize=label_fontsize)
    ax_energy.tick_params(axis="both", labelsize=tick_fontsize)
    ax_energy.set_xlim(t_full.min(), t_full.max())
    ax_energy.set_ylim(*pad(E_norm_full.min(), E_norm_full.max()))
    ax_energy.grid(alpha=0.3)

    for ax in (ax_wave, ax_EB, ax_Epar):
        ax.grid(alpha=0.3)

    fig.subplots_adjust(left=0.11, right=0.89, top=0.90, bottom=0.06, hspace=0.32)

    lines = (line_phi, line_A, line_Eperp, line_Bperp, line_Epar)
    frame_data = (Phi, A, E_perp, B_perp, E_par)

    def update(k: int):
        for line, arr in zip(lines, frame_data):
            line.set_data(z, arr[k])
        energy_marker.set_data([t[k]], [E_norm[k]])
        fig.suptitle(suptitle(t[k]), fontsize=title_fontsize)
        return (*lines, energy_marker, fig._suptitle)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


# =============================================================================
# 4. RUN
# =============================================================================
def animate_wave_potentials_generator():
    if data_dict_output is None:
        raise RuntimeError(
            "data_dict_output is still None -- fill in section 1 at the top "
            "of this file before running."
        )

    path = animate_wave_potentials(
        data_dict_output,
        OUTPUT_PATH,
        fps=FPS,
        dpi=DPI,
        figsize=FIGSIZE,
        altitude_scale=ALTITUDE_SCALE,
        altitude_label=ALTITUDE_LABEL,
        altitude_log=ALTITUDE_LOG,
        stride=STRIDE,
        title_fontsize=TITLE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        tick_fontsize=TICK_FONTSIZE,
    )
    print(f"wrote {path}")
