"""das_fft — Load windowed DAS data and precomputed FFT from step2a/step2b."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dasfm.io.data import DASData
from dasfm.io.middle_io import load_win_middle


def validate_das_win_dir(das_win_dir) -> None:
    """Pre-flight check: das_win directory exists and contains at least one .h5 file.

    Use :func:`dasfm.io.event_catalog_io.validate_per_event_files` to verify
    that every catalog event has a matching file inside this directory.

    Raises
    ------
    FileNotFoundError
        If the directory is missing, or contains no .h5 files.
    """
    das_win_dir = Path(das_win_dir)
    if not das_win_dir.is_dir():
        raise FileNotFoundError(f"das_win directory not found: {das_win_dir}")
    if not any(das_win_dir.glob("*.h5")):
        raise FileNotFoundError(
            f"das_win directory contains no *.h5 files: {das_win_dir}\n"
            f"  Run step2a_window first."
        )


def validate_das_fft_dir(das_fft_dir, require_lr: bool = False) -> None:
    """Pre-flight check: das_fft directory exists and contains valid FFT cache.

    Parameters
    ----------
    das_fft_dir : str or Path
        Directory containing per-event FFT HDF5 files written by step2a_window.
    require_lr : bool
        If True (e.g. ``polarity_method='hilbert'``), every file must have
        ``freqs_left`` / ``freqs_right`` datasets and ``has_lr=True``.
        Otherwise raise ``FileNotFoundError`` telling the user to re-run
        step2a_window with ``hilbert=True``.

    Raises
    ------
    FileNotFoundError
        If directory missing, no .h5 files, or hybrid mode required but
        ``has_lr`` is False on the spot-checked file.
    KeyError
        If the spot-checked file is missing required datasets / attrs.
    """
    import h5py
    das_fft_dir = Path(das_fft_dir)
    if not das_fft_dir.is_dir():
        raise FileNotFoundError(
            f"das_fft directory not found: {das_fft_dir}\n"
            f"  Run step2a_window first."
        )
    files = sorted(das_fft_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(
            f"das_fft directory contains no *.h5 files: {das_fft_dir}\n"
            f"  Run step2a_window first."
        )
    # Spot-check the first file's schema (datasets + attrs).
    with h5py.File(files[0], "r") as f:
        if "freqs_p" not in f:
            raise KeyError(
                f"{files[0].name}: missing dataset 'freqs_p' "
                f"(re-run step2a_window)."
            )
        for k in ("nfast", "dt", "has_lr"):
            if k not in f.attrs:
                raise KeyError(
                    f"{files[0].name}: missing attr '{k}' "
                    f"(re-run step2a_window after the das_fft refactor)."
                )
        if require_lr:
            if not bool(f.attrs["has_lr"]):
                raise FileNotFoundError(
                    f"polarity_method='hilbert' requires Hilbert L/R FFTs in "
                    f"das_fft, but {files[0].name} has has_lr=False. "
                    f"Re-run step2a_window with hilbert=True."
                )
            for ds in ("freqs_left", "freqs_right"):
                if ds not in f:
                    raise KeyError(
                        f"{files[0].name}: missing dataset '{ds}' "
                        f"(re-run step2a_window with hilbert=True)."
                    )


def load_das_window_single(das_win_dir, event_id, collect_hilbert=False):
    """Load windowed DAS data for a single event.

    Parameters
    ----------
    das_win_dir : str or Path
        Directory containing per-event HDF5 files from step2a.
    event_id : str
        Event ID to load.
    collect_hilbert : bool
        If True, also return Hilbert components (needed by step2b).

    Returns
    -------
    result : dict or None
        None if the event file does not exist.  Otherwise dict with keys:
        "dasdata" : DASData
        "dt" : float
        "n_ch" : int
        "event_id" : str
        If collect_hilbert:
            "p_original", "p_right", "p_left",
            "s_original", "s_right", "s_left",
            "taper_weights_x", "taper_n_x"
    """
    das_win_dir = Path(das_win_dir)
    fpath = das_win_dir / f"{event_id}.h5"
    if not fpath.exists():
        return None

    m = load_win_middle(fpath)
    n = m["p_original"].shape[0]

    dasdata = DASData(
        p_data        = m["p_original"],
        p_shift_index = np.full(n, m["cut_half"], dtype=np.int64),
        p_snr         = np.ones(n, dtype=np.float64),
        p_traveltime  = m["p_traveltime"],
        s_data        = m["s_original"],
        s_shift_index = np.full(n, m["cut_half"], dtype=np.int64),
        s_snr         = np.ones(n, dtype=np.float64),
        s_traveltime  = m["s_traveltime"],
        dt            = m["dt"],
        event_id      = m["event_id"],
        magnitude     = m["magnitude"],
        valid_mask    = m["p_valid"],
        p_valid       = m["p_valid"],
        s_valid       = m["s_valid"],
    )

    result = {
        "dasdata": dasdata,
        "dt": m["dt"],
        "n_ch": n,
        "event_id": event_id,
    }

    if collect_hilbert:
        result.update({
            "p_original": m["p_original"],
            "p_right": m["p_right"],
            "p_left": m["p_left"],
            "s_original": m["s_original"],
            "s_right": m["s_right"],
            "s_left": m["s_left"],
            "taper_weights_x": m["taper_weights_x"],
            "taper_n_x": m["taper_n_x"],
        })

    return result


def load_das_fft_single(fft_dir, event_id):
    """Load pre-computed P-wave FFT for a single event.

    Parameters
    ----------
    fft_dir : str or Path
        Directory containing per-event FFT HDF5 files from step2a.
    event_id : str
        Event ID to load.

    Returns
    -------
    np.ndarray or None
        Complex64 array of shape (n_ch, nfast//2+1), or None if not found.
    """
    import h5py
    fpath = Path(fft_dir) / f"{event_id}.h5"
    if not fpath.exists():
        return None
    with h5py.File(fpath, "r") as f:
        return f["freqs_p"][:].astype(np.complex64)


def load_das_fft_lr_single(fft_dir, event_id):
    """Load pre-computed left/right Hilbert FFT for a single event.

    Both ``freqs_left`` and ``freqs_right`` are written by step2a_window when
    ``hilbert=True`` and live in the same per-event ``{eid}.h5`` as ``freqs_p``.

    Parameters
    ----------
    fft_dir : str or Path
        Directory containing per-event FFT HDF5 files from step2a.
    event_id : str
        Event ID to load.

    Returns
    -------
    dict or None
        ``{"left": ndarray, "right": ndarray}`` complex64 arrays of shape
        ``(n_ch, nfast//2 + 1)``.  Returns None if the file is missing or if
        ``has_lr`` is False (i.e. step2a was run with ``hilbert=False``).
        Caller must handle the None case.
    """
    import h5py
    fpath = Path(fft_dir) / f"{event_id}.h5"
    if not fpath.exists():
        return None
    with h5py.File(fpath, "r") as f:
        if not bool(f.attrs.get("has_lr", False)):
            return None
        return {
            "left":  f["freqs_left"][:].astype(np.complex64),
            "right": f["freqs_right"][:].astype(np.complex64),
        }


def load_das_fft_meta(fft_dir, event_id):
    """Load metadata + p_valid from a das_fft file (no FFT data loaded).

    Used by step2b setup() to read ``dt / has_lr / nfast`` in one shot, and by
    :func:`dasfm.io.polarity_io.build_pol_valid` to populate ``pol_valid`` in
    the output H5 without re-loading any time-domain data.

    Parameters
    ----------
    fft_dir : str or Path
    event_id : str

    Returns
    -------
    dict or None
        ``{"nfast": int, "dt": float, "has_lr": bool, "event_id": str,
            "p_valid": ndarray bool (n_ch,) or None}``.
        None if file missing.
    """
    import h5py
    fpath = Path(fft_dir) / f"{event_id}.h5"
    if not fpath.exists():
        return None
    with h5py.File(fpath, "r") as f:
        return {
            "nfast":    int(f.attrs["nfast"]),
            "dt":       float(f.attrs["dt"]),
            "has_lr":   bool(f.attrs["has_lr"]),
            "event_id": str(f.attrs.get("event_id", event_id)),
            "p_valid":  f["p_valid"][:] if "p_valid" in f else None,
        }


