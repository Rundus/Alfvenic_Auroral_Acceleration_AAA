#!/usr/bin/env python3
"""
esa_velocity_space_movie.py

Convert an electrostatic-analyzer distribution function f(t, pitch angle, energy)
stored in a CDF file into (v_perp, v_parallel) velocity space and animate it.

    One panel per variable in PANELS (by default the distribution function and
    the differential energy flux), side by side and animated together. Each panel:
      - pcolormesh of log10 of the variable drawn on the true instrument bins
        (each energy x pitch-angle bin becomes an annular wedge)
      - black contour lines of the same quantity overlaid on top
      - MATLAB parula colormap with its own fixed colour range for all frames

Conversion
    |v|     = speed from energy (relativistic; identical to 1/2 m v^2 at these energies)
    v_par   = |v| cos(alpha)
    v_perp  = |v| sin(alpha)
    alpha = 0 deg maps to +v_par (field-aligned), alpha = 180 deg to -v_par.

Usage
    Edit CDF_PATH and OUTPUT_PATH at the bottom of this file and run it, or call
    the function from your own code:

        from esa_velocity_space_movie import make_vspace_movie
        make_vspace_movie("/path/to/detector_flux_700km.cdf",
                          "/path/to/esa_vspace.mp4")

Output
    H.264 video in an MP4 container (.mp4). If the output path has a
    different extension it is changed to .mp4.

Requirements
    numpy, matplotlib, cdflib (pip install cdflib), ffmpeg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator

try:
    import cdflib
except ImportError as err:
    raise ImportError("cdflib is required: pip install cdflib") from err


# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------
VPERP_LIM = (0.0, 20000.0)       # km/s
VPAR_LIM = (-10000.0, 30000.0)   # km/s
VPERP_TICK = 4000.0              # km/s between x-axis labels
VPAR_TICK = 2500.0               # km/s between y-axis labels
INVERT_VPAR = True               # True: positive v_par points down the page
DEFAULT_FPS = 4                  # playback frames per second
CONTOUR_STEP = 1.0               # spacing of contour lines in decades of f
CONTOUR_COLOR = "k"
CONTOUR_LW = 0.8
PITCH_SUBDIV = 8                 # sub-wedges per pitch bin so pcolormesh arcs look round
PANEL_WIDTH = 5.0                # figure width per panel, inches
FIG_HEIGHT = 9.5                 # inches

# Colorbar limits as (min, max) in log10 units, the same numbers shown on the
# colorbar ticks. Lowering the min brightens the low end of the colormap.
# Use None for either value to set it automatically from the data
# (min: 1st percentile rounded down to a decade; max: maximum rounded up).
F_CBAR_LIM = (-18, -11.0)       # log10 f   [s^3 m^-6]
DEF_CBAR_LIM = (7.0, 8.5)       # log10 DEF [eV cm^-2 s^-1 sr^-1 eV^-1]

# One panel per CDF variable, drawn side by side with identical axes and styling.
PANELS = [
    {"var": "distribution_function_esa",
     "title": "Distribution function",
     "cbar_label": r"$\log_{10}\,f$  [s$^3$ m$^{-6}$]",
     "cbar_lim": F_CBAR_LIM},
    {"var": "Differential_Energy_Flux",
     "title": "Differential energy flux",
     "cbar_label": r"$\log_{10}$ DEF  [eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]",
     "cbar_lim": DEF_CBAR_LIM},
]

Q_E = 1.602176634e-19            # C
C_LIGHT = 2.99792458e8           # m/s
MASS = {"electron": 9.1093837015e-31, "proton": 1.67262192369e-27}  # kg

# MATLAB parula (64 entries)
_PARULA = [
    [0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538],
]
PARULA = ListedColormap(_PARULA, name="parula")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_distribution(path, varname):
    """Return time, energy [eV], pitch angle [deg], f[time, pitch, energy], units."""
    cdf = cdflib.CDF(path)
    atts = cdf.varattsget(varname)
    f = np.asarray(cdf.varget(varname), dtype=float)

    # Coordinates come from the DEPEND_n attributes rather than hard-coded names
    t_name, d1, d2 = atts["DEPEND_0"], atts["DEPEND_1"], atts["DEPEND_2"]
    t = cdf.varget(t_name)
    ax1, ax2 = np.asarray(cdf.varget(d1), float), np.asarray(cdf.varget(d2), float)
    u1 = str(cdf.varattsget(d1).get("UNITS", "")).lower()

    # Put the array in (time, pitch, energy) order regardless of file layout
    if "deg" in u1:
        pitch, energy = ax1, ax2
    else:
        pitch, energy = ax2, ax1
        f = np.swapaxes(f, 1, 2)

    # Time: epoch types become datetimes, anything else is treated as numeric
    t_type = cdf.varinq(t_name).Data_Type_Description
    if "EPOCH" in t_type or "TT2000" in t_type:
        t = cdflib.cdfepoch.to_datetime(t)
        t_label = [str(x)[:23] for x in t]
    else:
        t_units = cdf.varattsget(t_name).get("UNITS", "")
        t_label = [f"t = {x:6.2f} {t_units}" for x in np.asarray(t, float)]

    # Sanitise fill / negative values; zeros and fills are masked for log scaling
    f = np.where(np.isfinite(f) & (f > 0) & (f < 1e30), f, 0.0)
    return t_label, energy, pitch, f, atts.get("UNITS", "")


# ----------------------------------------------------------------------------
# Energy / pitch angle -> velocity space
# ----------------------------------------------------------------------------
def energy_to_speed_kms(energy_ev, mass_kg):
    """Relativistic speed in km/s for kinetic energy in eV."""
    gamma = 1.0 + energy_ev * Q_E / (mass_kg * C_LIGHT**2)
    return C_LIGHT * np.sqrt(1.0 - 1.0 / gamma**2) / 1e3


def log_edges(centers):
    """Bin edges for log-spaced centers (geometric midpoints, extrapolated at ends)."""
    lc = np.log(centers)
    mid = 0.5 * (lc[1:] + lc[:-1])
    return np.exp(np.concatenate(([lc[0] - (mid[0] - lc[0])], mid,
                                  [lc[-1] + (lc[-1] - mid[-1])])))


def pitch_edges(centers_deg):
    """Bin edges for pitch angle centers, clipped to [0, 180] deg."""
    mid = 0.5 * (centers_deg[1:] + centers_deg[:-1])
    edges = np.concatenate(([centers_deg[0] - (mid[0] - centers_deg[0])], mid,
                            [centers_deg[-1] + (centers_deg[-1] - mid[-1])]))
    return np.clip(edges, 0.0, 180.0)


def subdivide(edges, n):
    """Split each bin into n equal sub-bins (for smooth arcs in pcolormesh)."""
    fine = [np.linspace(edges[i], edges[i + 1], n + 1)[:-1] for i in range(len(edges) - 1)]
    return np.concatenate(fine + [edges[-1:]])


def to_vspace(speed_kms, pitch_deg):
    """Map (pitch, speed) grids to (v_perp, v_par). Output shape (n_pitch, n_speed)."""
    a = np.deg2rad(pitch_deg)[:, None]
    v = speed_kms[None, :]
    return v * np.sin(a), v * np.cos(a)


def pad_to_axis(pitch_deg, logf):
    """Contour grid only: if the outermost pitch bins are not centred at 0 / 180 deg,
    add rows there carrying the adjacent bin's values so contours reach v_perp = 0."""
    if pitch_deg[0] > 0.0:
        pitch_deg = np.concatenate(([0.0], pitch_deg))
        logf = np.ma.concatenate((logf[:, :1], logf), axis=1)
    if pitch_deg[-1] < 180.0:
        pitch_deg = np.concatenate((pitch_deg, [180.0]))
        logf = np.ma.concatenate((logf, logf[:, -1:]), axis=1)
    return pitch_deg, logf


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def draw_contours(ax, vperp, vpar, logf, levels):
    """Black contour lines of log10 f on the grid of bin centres.
    linestyles is forced to solid; matplotlib otherwise dashes negative levels,
    and every log10 f level here is negative."""
    return ax.contour(vperp, vpar, logf, levels=levels, colors=CONTOUR_COLOR,
                      linewidths=CONTOUR_LW, linestyles="solid", zorder=2)


def remove_contours(cs):
    try:
        cs.remove()                      # matplotlib >= 3.8
    except AttributeError:
        for coll in cs.collections:      # older matplotlib
            coll.remove()


def style_axis(ax):
    ax.set_xlim(*VPERP_LIM)
    ax.set_ylim(*(VPAR_LIM[::-1] if INVERT_VPAR else VPAR_LIM))
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.5", lw=0.6, ls="--", zorder=3)
    ax.xaxis.set_major_locator(MultipleLocator(VPERP_TICK))
    ax.yaxis.set_major_locator(MultipleLocator(VPAR_TICK))
    ax.set_xlabel(r"$v_\perp$ [km/s]")
    ax.set_ylabel(r"$v_\parallel$ [km/s]")


def colour_range(x, in_win, panel):
    """log10 colour limits for one panel, fixed for every frame.
    Uses panel["cbar_lim"]; any None entry is filled in from the bins visible in
    the plot window (min: 1st percentile rounded down to a decade, max: maximum
    rounded up to a decade)."""
    lo, hi = panel.get("cbar_lim") or (None, None)
    if lo is None or hi is None:
        visible = x[:, in_win]
        visible = visible[visible > 0]
        if lo is None:
            lo = np.floor(np.log10(np.percentile(visible, 1)))
        if hi is None:
            hi = np.ceil(np.log10(visible.max()))
    lo, hi = float(lo), float(hi)
    if lo >= hi:
        raise ValueError(f"{panel['var']}: colorbar min ({lo}) must be below max ({hi})")
    return lo, hi


# ----------------------------------------------------------------------------
# MP4 output
# ----------------------------------------------------------------------------
def mp4_writer(fps):
    """FFmpeg writer for H.264 video in an MP4 container."""
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg was not found; it is required to write .mp4 files")
    extra = [
        "-pix_fmt", "yuv420p",                                  # plays everywhere
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",  # yuv420p needs even size
        "-crf", "18",                                           # quality (lower = better)
        "-preset", "medium",
        "-movflags", "+faststart",                              # starts playing sooner
    ]
    return animation.FFMpegWriter(fps=fps, codec="h264", bitrate=-1, extra_args=extra)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def make_vspace_movie(cdf_path, output_path, panels=PANELS, species="electron",
                      fps=DEFAULT_FPS, dpi=150, show=False):
    """Animate CDF variables in (v_perp, v_par) space and save the movie as an .mp4.

    Parameters
    ----------
    cdf_path : str
        Path to the input .cdf file.
    output_path : str
        Where to save the animation. Saved as MP4; the extension is set to .mp4.
    panels : list of dict
        One entry per panel; see PANELS at the top of the file.
    species : {"electron", "proton"}
        Particle mass used to convert energy to speed.
    fps : float
        Playback rate in animation frames per second (lower is slower).
    dpi : int
        Resolution of each frame.
    show : bool
        If True, display the animation in a window instead of saving it.

    Returns
    -------
    str or None
        The path of the saved .mp4 file (None when show=True).
    """
    cdf_path = Path(cdf_path).expanduser()
    if not cdf_path.is_file():
        raise FileNotFoundError(f"CDF file not found: {cdf_path}")
    if species not in MASS:
        raise ValueError(f"species must be one of {list(MASS)}")

    # Load every panel's variable; all must share the same time/pitch/energy grid
    t_label, energy, pitch, _, _ = load_distribution(str(cdf_path), panels[0]["var"])
    data = []
    for panel in panels:
        _, e_p, pa_p, x, _ = load_distribution(str(cdf_path), panel["var"])
        if not (np.allclose(e_p, energy) and np.allclose(pa_p, pitch)):
            raise ValueError(f"{panel['var']} is not on the same energy/pitch grid")
        data.append(x)
    nt = data[0].shape[0]
    speed = energy_to_speed_kms(energy, MASS[species])

    # Bin centres (for contours) and bin edges (for pcolormesh), shared by all panels
    vperp_c, vpar_c = to_vspace(speed, pitch)          # true bin centres
    pa_fine = subdivide(pitch_edges(pitch), PITCH_SUBDIV)
    vperp_e, vpar_e = to_vspace(energy_to_speed_kms(log_edges(energy), MASS[species]),
                                pa_fine)
    in_win = ((vperp_c >= VPERP_LIM[0]) & (vperp_c <= VPERP_LIM[1]) &
              (vpar_c >= VPAR_LIM[0]) & (vpar_c <= VPAR_LIM[1]))

    # Figure: one column per panel, same axes, pcolormesh + black contours in each
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(PANEL_WIDTH * len(panels), FIG_HEIGHT),
                             layout="constrained")
    axes = axes[0]
    title = fig.suptitle(t_label[0], fontsize=11)

    layers = []
    for k, (ax, panel, x) in enumerate(zip(axes, panels, data)):
        logx = np.ma.log10(np.ma.masked_less_equal(x, 0.0))
        pitch_k, logx_k = pad_to_axis(pitch, logx)        # grid used for contours
        vperp_k, vpar_k = to_vspace(speed, pitch_k)
        lo, hi = colour_range(x, in_win, panel)
        # Contours stay on whole multiples of CONTOUR_STEP even if the colour floor isn't
        levels = np.arange(np.ceil(lo / CONTOUR_STEP) * CONTOUR_STEP,
                           hi + 0.5 * CONTOUR_STEP, CONTOUR_STEP)

        style_axis(ax)
        if k > 0:
            ax.set_ylabel("")
        ax.set_title(panel.get("title", panel["var"]), fontsize=10)
        mesh = ax.pcolormesh(vperp_e, vpar_e, logx[0].repeat(PITCH_SUBDIV, axis=0),
                             cmap=PARULA, norm=Normalize(vmin=lo, vmax=hi),
                             shading="flat", rasterized=True, zorder=1)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, aspect=35, extend="both", pad=0.02)
        cbar.set_label(panel["cbar_label"])
        layers.append({"ax": ax, "mesh": mesh, "logx": logx, "logx_k": logx_k,
                       "vperp_k": vperp_k, "vpar_k": vpar_k, "levels": levels,
                       "lo": lo, "hi": hi,
                       "cs": draw_contours(ax, vperp_k, vpar_k, logx_k[0], levels)})

    def update(i):
        for L in layers:
            L["mesh"].set_array(L["logx"][i].repeat(PITCH_SUBDIV, axis=0))
            remove_contours(L["cs"])
            L["cs"] = draw_contours(L["ax"], L["vperp_k"], L["vpar_k"],
                                    L["logx_k"][i], L["levels"])
        title.set_text(t_label[i])
        return [L["mesh"] for L in layers] + [title]

    anim = animation.FuncAnimation(fig, update, frames=nt, interval=1000 / fps,
                                   blit=False)

    if show:
        plt.show()
        return None

    out = Path(output_path).expanduser()
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
        print(f"Output extension set to .mp4: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    def progress(i, n):
        print(f"\rRendering frame {i + 1}/{n}", end="", flush=True)

    anim.save(str(out), writer=mp4_writer(fps), dpi=dpi, progress_callback=progress)
    plt.close(fig)
    print(f"\nSaved {out}  ({nt} frames)")
    for panel, L in zip(panels, layers):
        print(f"  {panel['var']}: colour range 1e{L['lo']:g} to 1e{L['hi']:g}")
    return str(out)


if __name__ == "__main__":
    from src.Alfvenic_Auroral_Acceleration_AAA.run_toggles import RunToggles
    from glob import glob

    # Set these two strings, then run the file
    files = glob(f"{RunToggles.sim_data_output_path}/detector_flux/*.cdf")
    CDF_PATH = files[0]
    OUTPUT_PATH = f"{RunToggles.sim_data_output_path}/detector_flux/esa_vspace.mp4"

    make_vspace_movie(CDF_PATH, OUTPUT_PATH)