"""Shared utilities for dasfm.steps modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── SEED location-code helpers ───────────────────────────────────────────

def clean_location(val) -> str:
    """Normalise a SEED location code to a 2-digit string.

    Handles the common pandas gotcha where integer location codes
    (``0``, ``10``) become floats (``0.0``, ``10.0``) when NaN is present
    in the column.

    Examples: ``0.0`` → ``"00"``, ``10.0`` → ``"10"``, ``2.0`` → ``"02"``,
    ``NaN`` → ``"00"``, ``""`` → ``"00"``, ``"00"`` → ``"00"``.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "00"
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return "00"
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(2)
    return s


def clean_location_array(series_or_array) -> np.ndarray:
    """Apply :func:`clean_location` element-wise to a pandas Series or array."""
    if isinstance(series_or_array, pd.Series):
        return series_or_array.apply(clean_location).values
    return np.array([clean_location(v) for v in series_or_array], dtype=object)


def station_filename(network: str, station: str, location) -> str:
    """Layer-1 filename for a station: ``{network}_{station}_{location}.h5``."""
    return f"{network}_{station}_{clean_location(location)}.h5"


# ── Inversion type parsing ───────────────────────────────────────────────

# Valid data-type combinations → internal mode name
COMBO_TO_MODE = {
    frozenset(["sta_pol"]): "STA_pol",
    frozenset(["sta_pol", "sta_sp"]): "STA_pol_sp",
    frozenset(["das_pol"]): "DAS_pol",
    frozenset(["das_pol", "das_sp"]): "DAS_pol_sp",
    frozenset(["sta_pol", "das_pol"]): "STA_pol_DAS_pol",
    frozenset(["sta_pol", "das_sp"]): "STA_pol_DAS_sp",
    frozenset(["sta_pol", "das_pol", "das_sp"]): "STA_pol_DAS_pol_sp",
}

VALID_MODES = set(COMBO_TO_MODE.values())

# Backward compatibility: old all-lowercase names → new names
LEGACY_MODE_MAP = {
    "sta_pol": "STA_pol",
    "sta_pol_sp": "STA_pol_sp",
    "das_pol": "DAS_pol",
    "das_pol_sp": "DAS_pol_sp",
    "sta_pol_das_pol": "STA_pol_DAS_pol",
    "sta_pol_das_sp": "STA_pol_DAS_sp",
    "sta_pol_das_pol_sp": "STA_pol_DAS_pol_sp",
}


def parse_inversion_types(types_list):
    """Parse inversion types: support '+' format, new names, and legacy names.

    Invalid combinations are warned and skipped (not raised).

    Examples
    --------
    >>> parse_inversion_types(["sta_pol + das_sp"])
    ["STA_pol_DAS_sp"]
    >>> parse_inversion_types(["sta_pol_das_pol_sp"])  # legacy format
    ["STA_pol_DAS_pol_sp"]
    >>> parse_inversion_types(["das_sp"])  # invalid → skipped with warning
    []
    """
    import warnings
    valid_combos = [" + ".join(sorted(k)) for k in COMBO_TO_MODE]

    modes = []
    for t in types_list:
        t = t.strip()
        if "+" in t:
            parts = frozenset(p.strip() for p in t.split("+"))
            mode = COMBO_TO_MODE.get(parts)
            if mode is None:
                warnings.warn(
                    f"Skipping unsupported combination: {t!r}. "
                    f"Valid combinations:\n  " + "\n  ".join(valid_combos),
                    stacklevel=2)
                continue
            modes.append(mode)
        elif t in VALID_MODES:
            modes.append(t)
        elif t in LEGACY_MODE_MAP:
            modes.append(LEGACY_MODE_MAP[t])
        else:
            warnings.warn(
                f"Skipping unknown inversion type: {t!r}. "
                f"Valid: {sorted(VALID_MODES)}",
                stacklevel=2)
    return modes


# ── Centralized mode configuration ──────────────────────────────────────

ALL_MODES = list(COMBO_TO_MODE.values())

# Modes that use joint (polarity + S/P) solver
JOINT_MODES = {"STA_pol_sp", "STA_pol_DAS_sp", "STA_pol_DAS_pol_sp", "DAS_pol_sp"}

# Display titles
MODE_TITLES = {
    "STA_pol": "STA polarity",
    "STA_pol_sp": "STA pol + STA S/P",
    "STA_pol_DAS_pol": "STA + DAS polarity",
    "STA_pol_DAS_sp": "STA pol + DAS S/P",
    "STA_pol_DAS_pol_sp": "STA pol + DAS pol + DAS S/P",
    "DAS_pol": "DAS polarity",
    "DAS_pol_sp": "DAS pol + DAS S/P",
}

# Plot ordering index
MODE_INDEX = {
    "STA_pol": 1, "STA_pol_sp": 2, "DAS_pol": 3, "DAS_pol_sp": 4,
    "STA_pol_DAS_pol": 5, "STA_pol_DAS_sp": 6, "STA_pol_DAS_pol_sp": 7,
}

# Colors for summary plots
MODE_COLORS = {
    "STA_pol": "#7faed4", "STA_pol_sp": "#a0c4e8",
    "STA_pol_DAS_pol": "#f28e8e", "STA_pol_DAS_sp": "#f2c68e",
    "STA_pol_DAS_pol_sp": "#8ecf8e", "DAS_pol_sp": "#d4a76a",
    "DAS_pol": "#b5a0d4",
}

# Solution key mapping (pol-only vs joint)
MODE_SOL_KEY = {m: ("candidates_joint" if m in JOINT_MODES else "candidates_pol")
                for m in ALL_MODES}


def log_or_print(logger, msg, file_only=False):
    """Log via Logger if available, else print to console."""
    if logger is not None:
        if file_only:
            logger.log(msg)
        else:
            logger.info(msg)
    elif not file_only:
        print(msg)


class Logger:
    """Simple dual-output logger (console + file)."""

    def __init__(self, script: str, log_dir: str = "logs") -> None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.log_f = open(log_path / f"{script}.log", "w", buffering=1, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def info(self, msg: str = "") -> None:
        print(msg)
        print(msg, file=self.log_f)

    def log(self, msg: str = "") -> None:
        print(msg, file=self.log_f)

    def close(self) -> None:
        self.log_f.close()


def resolve_path(p, root):
    """Resolve *p* to an absolute path; relative paths are joined to *root*."""
    p = Path(p)
    return p if p.is_absolute() else root / p


# ── Device normalization (shared by step2a/step2b/step3 dispatchers) ─────

def resolve_device(device):
    """Normalize a user-supplied device argument to a canonical form.

    Accepts every form a user might write at the top of a run script::

        "cuda:3"            single GPU as string
        ["cuda:3"]          single GPU as a one-element list
        ["cuda:0", "cuda:1"]  multiple GPUs as a list
        "cpu"               CPU as string
        ["cpu"]             CPU wrapped in a list
        []                  empty list (treated as CPU)
        None                missing argument (treated as CPU)

    Returns
    -------
    norm_device : str | list[str]
        Canonical form for the dispatcher to forward to backends:
        - single-GPU and CPU collapse to a string,
        - multi-GPU stays as a list.
        Backends can therefore rely on ``isinstance(device, str)`` /
        ``device.startswith("cuda")`` checks without re-handling lists.
    num_gpu : int
        Number of CUDA entries (0 = CPU, 1 = single GPU, >1 = multi-GPU).
    gpu_devices : list[str]
        All CUDA entries as strings (empty when CPU).  The multi-GPU
        backends iterate over this directly.

    The function never raises on a missing/empty/invalid input — it
    silently degrades to ``("cpu", 0, [])`` so unknown forms still get
    a working CPU run instead of an opaque ``AttributeError`` deep
    inside a backend.
    """
    if device is None:
        return "cpu", 0, []
    if isinstance(device, list):
        gpu_devices = [d for d in device if isinstance(d, str) and d.startswith("cuda")]
    elif isinstance(device, str):
        gpu_devices = [device] if device.startswith("cuda") else []
    else:
        gpu_devices = []

    num_gpu = len(gpu_devices)
    if num_gpu == 0:
        norm_device = "cpu"
    elif num_gpu == 1:
        norm_device = gpu_devices[0]
    else:
        norm_device = list(gpu_devices)
    return norm_device, num_gpu, gpu_devices


def plot_ray_param_matrices(fig_dir, T_all, ito_all, az_all,
                                xlabel="Channel index", show_plots=False):
    """Plot traveltime / takeoff / azimuth QC matrices (3 in one row)."""
    import matplotlib.pyplot as plt
    import numpy as np

    data_list = [
        T_all,
        np.degrees(ito_all),
        np.degrees(az_all),
    ]
    labels = [
        "Traveltime [s]",
        "Takeoff [°]",
        "Azimuth [°]",
    ]

    # 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    for ax, data, label in zip(axes, data_list, labels):
        im = ax.imshow(data, aspect="auto", cmap="seismic", origin="upper")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event index")
        ax.set_title(f"{label} matrix")
        fig.colorbar(im, ax=ax, label=label)

    fig.tight_layout()
    out_file = fig_dir / "ray_param_matrices.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")

    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(out_file)))

    plt.close(fig)


# ---------------------------------------------------------------------------
#  step3 helpers
# ---------------------------------------------------------------------------

def find_result_dir(base, prefix):
    """Return next available try directory under base."""
    from pathlib import Path
    base = Path(base)
    n = 1
    while True:
        p = base / f"{prefix}try_dir{n}"
        if not p.exists() or not any(p.glob("inv_sol/*/*.h5")):
            return p
        n += 1


def build_obs(mode, iev, n_ch, Pkic, das_mc_takeoff_deg, das_mc_azimuth_deg,
               sp_ratios, sta_takeoff_deg_t0, sta_az_deg_t0, sta_pol_i,
               sta_sp_takeoff_t0, sta_sp_az_t0, sta_sp_obs_i, HAS_DAS_SRC):
    """Build observation dict for a given mode and event (for plotting)."""
    obs = {}
    use_das_pol = mode in {"DAS_pol", "DAS_pol_sp", "STA_pol_DAS_pol", "STA_pol_DAS_pol_sp"}
    use_das_sp = mode in {"DAS_pol_sp", "STA_pol_DAS_sp", "STA_pol_DAS_pol_sp"}
    use_sta_pol = mode in {"STA_pol", "STA_pol_sp", "STA_pol_DAS_pol", "STA_pol_DAS_sp", "STA_pol_DAS_pol_sp"}
    use_sta_sp = mode == "STA_pol_sp"

    if use_das_pol and n_ch > 0 and HAS_DAS_SRC and Pkic is not None:
        obs["das_az"] = das_mc_azimuth_deg[0][iev]
        obs["das_takeoff"] = das_mc_takeoff_deg[0][iev]
        obs["das_pol"] = Pkic[:, iev]
    if use_das_sp and sp_ratios is not None and n_ch > 0:
        obs["das_sp"] = sp_ratios[:, iev]
        if "das_az" not in obs and HAS_DAS_SRC:
            obs["das_az"] = das_mc_azimuth_deg[0][iev]
            obs["das_takeoff"] = das_mc_takeoff_deg[0][iev]
    if use_sta_pol and sta_takeoff_deg_t0 is not None and sta_pol_i is not None:
        obs["sta_az"] = sta_az_deg_t0
        obs["sta_takeoff"] = sta_takeoff_deg_t0
        obs["sta_pol"] = sta_pol_i
    if use_sta_sp and sta_sp_takeoff_t0 is not None and sta_sp_obs_i is not None:
        obs["sta_sp_az"] = sta_sp_az_t0
        obs["sta_sp_takeoff"] = sta_sp_takeoff_t0
        obs["sta_sp"] = sta_sp_obs_i
    return obs
