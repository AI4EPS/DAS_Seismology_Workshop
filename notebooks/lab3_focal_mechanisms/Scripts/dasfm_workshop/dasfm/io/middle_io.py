"""Read / write DAS windowed P/S files (das_win format).

HDF5 layout
-----------
::

    data/                                        (group)
      @event_id, @event_time, @dt
      @magnitude, @latitude, @longitude, @depth_km
      ──────────────────────────────────────────
      P/                                         (sub-group)
        data         (n_ch, n_win)  float32
        traveltime   (n_ch,)        float64
        shift_index  (n_ch,)        int64
        snr          (n_ch,)        float64
      S/                                         (sub-group)
        data         (n_ch, n_win)  float32
        traveltime   (n_ch,)        float64
        shift_index  (n_ch,)        int64
        snr          (n_ch,)        float64
      P_right/  (optional, Hilbert right-going)
        data            (n_ch, n_win)  float32
        taper_weights_x (n_ch,)        float32
        @taper_n_x      int
      P_left/   (optional, Hilbert left-going)
        data         (n_ch, n_win)  float32
      S_right/  (optional)
        data         (n_ch, n_win)  float32
      S_left/   (optional)
        data         (n_ch, n_win)  float32

Public functions
----------------
* :func:`write_win_middle`
* :func:`load_win_middle`
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def write_win_middle(
    out_path: str | Path,
    hilbert_res: dict,
    p_traveltime: np.ndarray,
    s_traveltime: np.ndarray,
    dt: float,
    event_id: str              = "",
    event_time: str            = "",
    magnitude: Optional[float] = None,
    latitude: Optional[float]  = None,
    longitude: Optional[float] = None,
    depth_km: Optional[float]  = None,
    p_valid: Optional[np.ndarray] = None,
    s_valid: Optional[np.ndarray] = None,
) -> None:
    """Save Hilbert-separated P/S windows to an HDF5 file (das_win format).

    Parameters
    ----------
    out_path : str or Path
        Output ``.h5`` file path (parent directories are created automatically).
    hilbert_res : dict
        Required keys: ``p_original``, ``s_original``, ``cut_half``, ``taper_n_x``.
        Optional keys: ``p_left``, ``p_right``, ``s_left``, ``s_right``,
        ``t_cut``, ``taper_mask_x``, ``taper_weights_x``.
    p_traveltime, s_traveltime : np.ndarray, shape (n_ch,)
        Per-channel P/S traveltimes [s].
    dt : float
        Sampling interval [s].
    event_id, magnitude, latitude, longitude, depth_km
        Event metadata stored as HDF5 group attributes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _nan(v):
        return float("nan") if v is None else float(v)

    n_ch     = np.asarray(hilbert_res["p_original"]).shape[0]
    cut_half = int(hilbert_res["cut_half"])
    _shift   = np.full(n_ch, cut_half, dtype=np.int64)
    _snr     = np.ones(n_ch, dtype=np.float64)
    kw       = dict(compression="gzip", compression_opts=4)

    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")

        # ── metadata attributes ──────────────────────────────────────────────
        grp.attrs["event_id"]   = str(event_id)
        grp.attrs["event_time"] = str(event_time)
        grp.attrs["dt"]         = float(dt)
        grp.attrs["magnitude"]  = _nan(magnitude)
        grp.attrs["latitude"]   = _nan(latitude)
        grp.attrs["longitude"]  = _nan(longitude)
        grp.attrs["depth_km"]   = _nan(depth_km)

        # ── P and S sub-groups ───────────────────────────────────────────────
        for phase, orig_key, tt in [
            ("P", "p_original", p_traveltime),
            ("S", "s_original", s_traveltime),
        ]:
            sg = grp.create_group(phase)
            sg.create_dataset("data",        data=np.asarray(hilbert_res[orig_key], dtype=np.float32), **kw)
            sg.create_dataset("traveltime",  data=np.asarray(tt, dtype=np.float64))
            sg.create_dataset("shift_index", data=_shift)
            sg.create_dataset("snr",         data=_snr)

        # ── Pick validity masks ───────────────────────────────────────────────
        if p_valid is not None:
            grp.create_dataset("p_valid", data=np.asarray(p_valid, dtype=bool))
        if s_valid is not None:
            grp.create_dataset("s_valid", data=np.asarray(s_valid, dtype=bool))

        # ── Hilbert components (siblings of P/ and S/) ───────────────────────
        for grp_name, res_key in [
            ("P_left",  "p_left"),
            ("P_right", "p_right"),
            ("S_left",  "s_left"),
            ("S_right", "s_right"),
        ]:
            if hilbert_res.get(res_key) is not None:
                sg = grp.create_group(grp_name)
                sg.create_dataset(
                    "data",
                    data=np.asarray(hilbert_res[res_key], dtype=np.float32),
                    **kw,
                )
                # Store taper info inside P_right/
                if grp_name == "P_right":
                    if hilbert_res.get("taper_weights_x") is not None:
                        sg.create_dataset("taper_weights_x",
                                          data=np.asarray(hilbert_res["taper_weights_x"],
                                                          dtype=np.float32))
                    sg.attrs["taper_n_x"] = int(hilbert_res.get("taper_n_x", 0))

    pass  # progress is reported by the caller via tqdm


def load_win_middle(filepath: str | Path) -> dict:
    """Load a das_win H5 file written by :func:`write_win_middle`
    or in the compatible user-provided format (P/ and S/ sub-groups).

    Returns
    -------
    dict with keys
        ``p_original``, ``p_traveltime``, ``s_original``, ``s_traveltime``
        (required — always present),
        ``p_right``, ``p_left``, ``s_right``, ``s_left``
        (float32 ndarrays or None if Hilbert components are absent),
        ``taper_weights_x`` (float32 ndarray or None),
        and scalar metadata: ``dt``, ``cut_half``, ``taper_n_x``,
        ``event_id``, ``event_time``, ``magnitude``,
        ``latitude``, ``longitude``, ``depth_km``.

    Compatible with both step2a output and user-provided files
    (e.g. ``73482301.h5`` with only ``data/P/`` and ``data/S/``).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"das_win file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        grp   = f["data"]
        attrs = dict(grp.attrs)

        # ── Core P/S data (required) ──────────────────────────────────────
        out = {
            "p_original":   grp["P/data"][:],
            "p_traveltime": grp["P/traveltime"][:],
            "s_original":   grp["S/data"][:],
            "s_traveltime": grp["S/traveltime"][:],
        }

        # ── Pick validity masks ───────────────────────────────────────────
        out["p_valid"] = grp["p_valid"][:] if "p_valid" in grp else None
        out["s_valid"] = grp["s_valid"][:] if "s_valid" in grp else None

        # ── cut_half: from P/shift_index ──────────────────────────────────
        out["cut_half"] = int(grp["P/shift_index"][0])

        # ── Hilbert components (optional) ─────────────────────────────────
        out["p_right"] = grp["P_right/data"][:] if "P_right" in grp else None
        out["p_left"]  = grp["P_left/data"][:]  if "P_left"  in grp else None
        out["s_right"] = grp["S_right/data"][:] if "S_right" in grp else None
        out["s_left"]  = grp["S_left/data"][:]  if "S_left"  in grp else None

        # ── Taper (stored inside P_right/, or legacy top-level) ───────────
        pr = grp.get("P_right")
        if pr is not None and "taper_weights_x" in pr:
            out["taper_weights_x"] = pr["taper_weights_x"][:]
        elif "taper_weights_x" in grp:
            out["taper_weights_x"] = grp["taper_weights_x"][:]
        else:
            out["taper_weights_x"] = None

        if pr is not None and "taper_n_x" in pr.attrs:
            out["taper_n_x"] = int(pr.attrs["taper_n_x"])
        else:
            out["taper_n_x"] = int(attrs.get("taper_n_x", out["cut_half"]))

        # ── dt ────────────────────────────────────────────────────────────
        if "dt" in attrs:
            out["dt"] = float(attrs["dt"])
        elif "dt" in grp["P/data"].attrs:
            out["dt"] = float(grp["P/data"].attrs["dt"])
        else:
            raise KeyError(
                f"'dt' not found in {filepath}. "
                "Expected at data.attrs['dt'] or data/P/data.attrs['dt']."
            )

        # ── Event metadata ────────────────────────────────────────────────
        out["event_id"]   = str(attrs.get("event_id", ""))
        out["event_time"] = str(attrs.get("event_time", ""))
        out["magnitude"]  = float(attrs.get("magnitude", float("nan")))
        out["latitude"]   = float(attrs.get("latitude",  float("nan")))
        out["longitude"]  = float(attrs.get("longitude", float("nan")))
        out["depth_km"]   = float(attrs.get("depth_km",  float("nan")))

    return out


def validate_das_polarity(filepath) -> None:
    """Pre-flight check: DAS polarity HDF5 file exists and has required datasets.

    Required datasets: ``Pkic``, ``sigma_perc_0``, ``event_ids``, ``channel_ids``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required datasets are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAS polarity HDF5 file not found: {filepath}")
    with h5py.File(filepath, "r") as f:
        keys = set(f.keys())
    required = {"Pkic", "sigma_perc_0", "event_ids", "channel_ids"}
    missing = required - keys
    if missing:
        raise KeyError(
            f"DAS polarity file {filepath} missing required datasets: {sorted(missing)}"
        )


def validate_das_sp_ratios(filepath) -> None:
    """Pre-flight check: DAS S/P ratios HDF5 file exists and has required datasets.

    Required datasets: ``sp_ratios``, ``sp_valid``, ``event_ids``, ``channel_ids``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required datasets are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAS S/P ratios HDF5 file not found: {filepath}")
    with h5py.File(filepath, "r") as f:
        keys = set(f.keys())
    required = {"sp_ratios", "sp_valid", "event_ids", "channel_ids"}
    missing = required - keys
    if missing:
        raise KeyError(
            f"DAS S/P ratios file {filepath} missing required datasets: {sorted(missing)}"
        )


def load_das_polarity(filepath):
    """Load DAS polarity from polarity.h5.

    Returns
    -------
    dict with keys: Pkic, pol_valid, sigma_perc_0, event_ids, channel_ids
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        result = {
            "Pkic":         f["Pkic"][:],
            "sigma_perc_0": f["sigma_perc_0"][:],
            "event_ids":    list(f["event_ids"].asstr()[:]),
            "channel_ids":  f["channel_ids"][:],
            "pol_valid":    f["pol_valid"][:] if "pol_valid" in f else None,
        }
    return result


def load_das_sp_ratios(filepath):
    """Load DAS S/P ratios from sp_ratios.h5.

    Returns
    -------
    dict with keys: sp_ratios, sp_valid, event_ids, channel_ids, p_amp, s_amp
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        sp_ratios = f["sp_ratios"][:].copy()
        sp_valid  = f["sp_valid"][:]
        sp_ratios[~sp_valid] = np.nan
        result = {
            "sp_ratios":    sp_ratios,
            "sp_valid":     sp_valid,
            "event_ids":    list(f["event_ids"].asstr()[:]),
            "channel_ids":  f["channel_ids"][:],
            "p_amp":         f["p_amp"][:] if "p_amp" in f else None,
            "s_amp":         f["s_amp"][:] if "s_amp" in f else None,
        }
    return result
