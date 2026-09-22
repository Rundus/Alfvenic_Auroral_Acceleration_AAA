r"""
Stacked spectrogram plotter for a directory of detector-flux CDF files.

One figure, N rows x 2 columns, where N is the number of .cdf files:
    left  column : Differential_Energy_Flux averaged over pitch angle -> energy vs time
    right column : Differential_Energy_Flux averaged over energy      -> pitch angle vs time

Rows are labelled with the detector altitude parsed from the file name
(detector_flux_<altitude>km.cdf), highest altitude at the top.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import spaceToolsLib as stl

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# DATA_DIR = Path(r"C:/data/alfvenic_auroral_acceleration_AAA/run_0/detector_flux")
DATA_DIR = Path(r"/home/connor/Data/MODELS/alfvenic_auroral_acceleration_AAA/run_0/detector_flux/")

FLUX_KEY = "Differential_Energy_Flux"
TIME_KEY = "time"
ENERGY_KEY = "energy"
PITCH_KEY = "pitch_angle"

LOG_COLOR = True              # flux spans decades; zeros are rendered as gaps
CMAP = "turbo"                # "viridis" for a perceptually uniform map
LOG_ENERGY_AXIS = True

# Fixed colour limits, one per column. Set either to None to fall back to
# CLIM_PERCENTILES for that column. Values outside the range are clamped to
# the end colours (the colorbars carry extend arrows to show this).
ENERGY_COLOR_LIMITS = (1e6, 7e7)    # left column: energy vs time
PITCH_COLOR_LIMITS = (2e6, 3e7)     # right column: pitch angle vs time
CLIM_PERCENTILES = (1.0, 99.9)      # used only where the above is None
SHARED_COLOR_LIMITS = True    # one colour scale per column, so altitudes compare

HIGH_ALTITUDE_FIRST = True    # top row is the highest altitude

FIG_WIDTH = 14.0              # inches
ROW_HEIGHT = 3.0              # inches per file

# Figure output. OUTPUT_DIR is tried first; if it refuses a write (read-only
# mount, synced/virtual drive, controlled-folder-access, missing permissions)
# the script falls back to the directories below rather than dying.
OUTPUT_DIR = DATA_DIR
FIGURE_NAME = "detector_fluxes.png"
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:            # interactive session
    _SCRIPT_DIR = None
FALLBACK_OUTPUT_DIRS = (_SCRIPT_DIR, Path.cwd())

SHOW = False

# Matches "..._2000km.cdf", "..._400km.cdf", "..._1.5km.cdf"
ALTITUDE_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)\s*km", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# File discovery and loading
# --------------------------------------------------------------------------- #
def altitude_from_path(path):
    """Numeric altitude in km parsed from the file name, or None."""
    match = ALTITUDE_PATTERN.search(Path(path).stem)
    return float(match.group(1)) if match else None


def altitude_label(path):
    """Row label, e.g. '400 km'. Falls back to the file stem."""
    altitude = altitude_from_path(path)
    return f"{altitude:g} km" if altitude is not None else Path(path).stem


def cdf_files(directory=DATA_DIR, descending=HIGH_ALTITUDE_FIRST):
    """.cdf files sorted by altitude, not lexicographically.

    A plain glob returns 2000, 4000, 400, 6000 — string ordering. This sorts
    numerically, descending by default so the highest altitude ends up in the
    top row. Files without a parsable altitude go last, alphabetically.
    """
    def key(path):
        altitude = altitude_from_path(path)
        value = altitude if altitude is not None else 0.0
        return (altitude is None, -value if descending else value,
                path.stem.lower())

    return sorted(Path(directory).glob("*.cdf"), key=key)


def load_data_dictionaries(directory=DATA_DIR):
    """{'6000 km': {var: [array, meta], ...}, ...}, highest altitude first."""
    dictionaries = {}
    for path in cdf_files(directory):
        label = altitude_label(path)
        if label in dictionaries:               # duplicate altitudes, if any
            label = f"{label} ({path.stem})"
        dictionaries[label] = stl.loadDictFromFile(str(path))
    return dictionaries


def _writable(directory):
    """True if `directory` exists (or can be created) and accepts a file write."""
    directory = Path(directory)
    probe = directory / ".detector_flux_write_probe.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(probe, "wb") as handle:
            handle.write(b"0")
        probe.unlink()
        return True
    except OSError as exc:
        print(f"cannot write to {directory}: {exc.__class__.__name__}: {exc}")
        return False


def resolve_save_path(output_dir=None, name=None):
    """First writable candidate directory joined with `name`."""
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    name = FIGURE_NAME if name is None else name
    candidates = [output_dir, *FALLBACK_OUTPUT_DIRS]
    for candidate in candidates:
        if candidate is None:
            continue
        if _writable(candidate):
            if Path(candidate).resolve() != Path(output_dir).resolve():
                print(f"falling back to {candidate}")
            return Path(candidate) / name
    raise OSError(f"none of these directories are writable: {candidates}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_ENTITIES = {
    "&parallel;": "\u2225", "&perp;": "\u22a5", "&theta;": "\u03b8",
    "&phi;": "\u03c6", "&alpha;": "\u03b1", "&mu;": "\u03bc",
    "&Delta;": "\u0394", "&deg;": "\u00b0",
}


def _format_idl(text):
    """Render IDL/CDF label codes ('cm!U-2!N') as mathtext ('cm$^{-2}$')."""
    if not isinstance(text, str):
        return text
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r"!(?:U|A|E)(.*?)!N", lambda m: f"$^{{{m.group(1)}}}$", text)
    text = re.sub(r"!(?:B|D|I)(.*?)!N", lambda m: f"$_{{{m.group(1)}}}$", text)
    return text.replace("!C", "\n").replace("!N", "")


def _entry(data_dictionary, key):
    """Return (values, metadata) for `key`, tolerating a missing metadata dict."""
    try:
        entry = data_dictionary[key]
    except KeyError as exc:
        raise KeyError(
            f"variable {key!r} not present; available keys: {sorted(data_dictionary)}"
        ) from exc
    if isinstance(entry, (list, tuple)):
        values = entry[0]
        meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    else:
        values, meta = entry, {}
    return np.asarray(values), meta


def _axis_label(meta, fallback):
    name = None
    for key in ("LABLAXIS", "FIELDNAM", "CATDESC"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    name = _format_idl(name or fallback)
    units = meta.get("UNITS") or meta.get("UNIT")
    if isinstance(units, str) and units.strip():
        return f"{name} [{_format_idl(units.strip())}]"
    return name


def _clean(array, meta):
    """Cast to float; replace fill values and non-finite entries with NaN."""
    values = np.array(array, dtype=float)
    fill = meta.get("FILLVAL", meta.get("FILL_VALUE"))
    if fill is not None:
        try:
            values[values == float(fill)] = np.nan
        except (TypeError, ValueError):
            pass
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1e30] = np.nan
    return values


def _support_axis(array):
    """Collapse a possibly time-varying (2D) energy or pitch-angle table to 1D."""
    values = np.array(array, dtype=float)
    values[np.abs(values) > 1e30] = np.nan
    if values.ndim == 2:
        values = np.nanmedian(values, axis=0)
    return np.squeeze(values)


def _prepare_time(array):
    values = np.asarray(array)
    if values.dtype == object:
        try:
            return np.array(values, dtype="datetime64[ns]")
        except (TypeError, ValueError):
            return values
    return values


def _colour_norm(values, limits):
    if limits is not None:
        vmin, vmax = (float(v) for v in limits)
        return LogNorm(vmin, vmax) if LOG_COLOR else Normalize(vmin, vmax)

    finite = np.asarray(values)[np.isfinite(values)]
    if LOG_COLOR:
        finite = finite[finite > 0.0]
    if finite.size == 0:
        return LogNorm(1.0, 10.0) if LOG_COLOR else Normalize(0.0, 1.0)
    if CLIM_PERCENTILES is None:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        vmin, vmax = (float(v) for v in np.percentile(finite, CLIM_PERCENTILES))
    if not vmax > vmin:
        vmax = vmin * 10.0 if LOG_COLOR and vmin > 0 else vmin + 1.0
    return LogNorm(vmin, vmax) if LOG_COLOR else Normalize(vmin, vmax)


def _draw(ax, x, y, values, norm):
    """Draw a [time, y] array as a spectrogram."""
    grid = np.ma.masked_invalid(values.T)
    if LOG_COLOR:
        grid = np.ma.masked_less_equal(grid, 0.0)
    return ax.pcolormesh(x, y, grid, cmap=CMAP, norm=norm, shading="auto")


def _row_payload(label, data):
    """Extract coordinates and both reductions for one file."""
    flux_values, flux_meta = _entry(data, FLUX_KEY)
    flux = _clean(flux_values, flux_meta)
    if flux.ndim != 3:
        raise ValueError(
            f"{label}: expected {FLUX_KEY} shaped [time, pitch_angle, energy], "
            f"got {flux.shape}"
        )

    time_values, time_meta = _entry(data, TIME_KEY)
    energy_values, energy_meta = _entry(data, ENERGY_KEY)
    pitch_values, pitch_meta = _entry(data, PITCH_KEY)

    time = _prepare_time(time_values)
    energy = _support_axis(energy_values)
    pitch = _support_axis(pitch_values)

    expected = (time.shape[0], pitch.shape[0], energy.shape[0])
    if flux.shape != expected:
        raise ValueError(
            f"{label}: {FLUX_KEY} has shape {flux.shape} but the coordinate "
            f"arrays imply {expected} (time, pitch_angle, energy)"
        )

    # Mean over the collapsed axis, ignoring NaN bins. All-NaN slices return
    # NaN (and raise a RuntimeWarning), which is exactly the gap we want.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        over_pitch = np.nanmean(flux, axis=1)    # [time, energy]
        over_energy = np.nanmean(flux, axis=2)   # [time, pitch angle]

    return {
        "label": label,
        "time": time, "energy": energy, "pitch": pitch,
        "over_pitch": over_pitch, "over_energy": over_energy,
        "time_meta": time_meta, "energy_meta": energy_meta,
        "pitch_meta": pitch_meta, "flux_meta": flux_meta,
    }


# --------------------------------------------------------------------------- #
# Main plotting routine
# --------------------------------------------------------------------------- #
def plot_flux_dictionaries(dictionaries, save_path="auto"):
    """Plot every file as a row. save_path: "auto", an explicit path, or None."""
    if isinstance(dictionaries, (list, tuple)):
        dictionaries = {f"file {i}": d for i, d in enumerate(dictionaries)}
    if not dictionaries:
        raise ValueError(f"no data supplied — are there .cdf files in {DATA_DIR}?")

    rows = [_row_payload(label, data) for label, data in dictionaries.items()]
    n_rows = len(rows)

    # Fixed limits are already per-column, so they bypass the shared/per-row
    # distinction entirely; the shared path only matters when limits are None.
    if SHARED_COLOR_LIMITS or ENERGY_COLOR_LIMITS is not None:
        left_norm = _colour_norm(
            np.concatenate([r["over_pitch"].ravel() for r in rows]), ENERGY_COLOR_LIMITS)
    else:
        left_norm = None

    if SHARED_COLOR_LIMITS or PITCH_COLOR_LIMITS is not None:
        right_norm = _colour_norm(
            np.concatenate([r["over_energy"].ravel() for r in rows]), PITCH_COLOR_LIMITS)
    else:
        right_norm = None

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(FIG_WIDTH, ROW_HEIGHT * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, payload in enumerate(rows):
        ax_left, ax_right = axes[row, 0], axes[row, 1]
        ax_right.sharex(ax_left)

        mesh_left = _draw(
            ax_left, payload["time"], payload["energy"], payload["over_pitch"],
            left_norm if left_norm is not None
            else _colour_norm(payload["over_pitch"], ENERGY_COLOR_LIMITS),
        )
        mesh_right = _draw(
            ax_right, payload["time"], payload["pitch"], payload["over_energy"],
            right_norm if right_norm is not None
            else _colour_norm(payload["over_energy"], PITCH_COLOR_LIMITS),
        )

        units = payload["flux_meta"].get("UNITS") or payload["flux_meta"].get("UNIT")
        suffix = f"\n[{_format_idl(units)}]" if isinstance(units, str) else ""
        fig.colorbar(mesh_left, ax=ax_left, pad=0.01, extend="both").set_label(
            f"Mean over pitch angle{suffix}", fontsize=8)
        fig.colorbar(mesh_right, ax=ax_right, pad=0.01, extend="both").set_label(
            f"Mean over energy{suffix}", fontsize=8)

        ax_left.set_ylabel(_axis_label(payload["energy_meta"], "Energy"), fontsize=9)
        ax_right.set_ylabel(_axis_label(payload["pitch_meta"], "Pitch angle"), fontsize=9)

        energy = payload["energy"]
        if LOG_ENERGY_AXIS and np.all(energy[np.isfinite(energy)] > 0):
            ax_left.set_yscale("log")

        pitch = payload["pitch"][np.isfinite(payload["pitch"])]
        if pitch.size and pitch.min() >= -1 and pitch.max() <= 181:
            ax_right.set_yticks([0, 45, 90, 135, 180])

        label = payload["label"]
        ax_left.set_title(f"{label} — averaged over pitch angle", fontsize=10)
        ax_right.set_title(f"{label} — averaged over energy", fontsize=10)

        time = payload["time"]
        is_datetime = np.issubdtype(np.asarray(time).dtype, np.datetime64)
        is_bottom = row == n_rows - 1
        for ax in (ax_left, ax_right):
            ax.set_xlim(time.min(), time.max())
            ax.tick_params(labelsize=8, labelbottom=is_bottom)
            if is_bottom:
                ax.set_xlabel(_axis_label(payload["time_meta"], "Time"), fontsize=9)
                if is_datetime:
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(20)
                        tick.set_horizontalalignment("right")

    fig.suptitle(f"{FLUX_KEY} — {DATA_DIR.parent.name}", fontsize=12)

    if save_path == "auto":
        save_path = resolve_save_path()
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"saved {save_path}")

    return fig, axes


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data_dictionaries = load_data_dictionaries()
    print(f"{len(data_dictionaries)} file(s): {', '.join(data_dictionaries)}")

    plot_flux_dictionaries(data_dictionaries)
    if SHOW:
        plt.show()