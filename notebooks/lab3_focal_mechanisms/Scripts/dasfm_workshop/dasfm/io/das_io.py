"""DAS data readers for HDF5 files.

Two public functions:

* :func:`load_das_window` — pre-extracted P/S time-window files.
* :func:`load_das_raw`    — continuous raw-trace files.

Both are **pure readers**: they perform only I/O and numerical-safety fixes
(NaN/Inf → 0).  All waveform preprocessing (filtering, demean, detrend) is
handled separately by :mod:`dasfm.io.preprocess`.

----

**Window file layout** (pre-extracted P/S windows)::

    data/                          (group)
      @event_id, @event_time, @latitude, @longitude, @depth_km, @magnitude
      P/
        data         (n_channels, n_samples)  float32
          @dt   float
        shift_index  (n_channels,)  int64
        snr          (n_channels,)  float64
        traveltime   (n_channels,)  float64
          @traveltime_type  str
          @tref             float
      S/  (same structure as P/)
      N/  (optional – pre-P background noise window)
        data  (n_channels, n_samples)  float32

**Raw file layout** (continuous trace)::

    data  (n_channels, n_samples)  float32
      @dt_s, @dx_m, @event_time_index
      @begin_time, @end_time, @event_time, @event_id
      @latitude, @longitude, @depth_km, @magnitude, @magnitude_type
      @unit, @time_before, @time_after

----

Channel QC strategy
-------------------
* NaN / Inf values are **replaced with zero** (always applied).
  Affected channels are flagged ``valid_mask = False``.
* Low-SNR and all-zero channels are **flag-only** — data is never modified.
* Total channel count is always preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from dasfm.io.data import DASData, DASRawData


# ===========================================================================
# Public API
# ===========================================================================

def load_das_window(
    filepath: str | Path,
    channel_coords: Optional[np.ndarray] = None,
    snr_min: float = 0.0,
    mask_zeros: bool = True,
) -> DASData:
    """Load a pre-extracted P/S window HDF5 file into a :class:`~dasfm.io.data.DASData`.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5`` file.
    channel_coords : np.ndarray, optional
        Channel positions in metres, shape ``(n_channels, 3)`` as ``[x, y, z]``.
    snr_min : float
        Flag channels whose P **or** S SNR is below this value
        (default ``0.0`` — no flagging).
    mask_zeros : bool
        Flag channels whose P or S window is entirely zero (default ``True``).

    Returns
    -------
    DASData
        ``valid_mask[i]`` is ``True`` iff channel *i* passed all QC checks.
        ``metadata['n_fixed']`` reports channels that had NaN/Inf zeroed.

    Raises
    ------
    FileNotFoundError
    ValueError
        If P and S channel counts differ.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        grp   = f["data"]
        attrs = dict(grp.attrs)

        p_grp         = grp["P"]
        p_data        = p_grp["data"][:].astype(np.float32)
        p_shift_index = p_grp["shift_index"][:]
        p_snr         = p_grp["snr"][:]
        p_traveltime  = p_grp["traveltime"][:]
        dt = float(p_grp["data"].attrs.get("dt", 0.01))
        metadata = {
            "traveltime_type": str(p_grp["traveltime"].attrs.get("traveltime_type", "")),
            "tref": float(p_grp["traveltime"].attrs.get("tref", 0)),
        }

        s_grp         = grp["S"]
        s_data        = s_grp["data"][:].astype(np.float32)
        s_shift_index = s_grp["shift_index"][:]
        s_snr         = s_grp["snr"][:]
        s_traveltime  = s_grp["traveltime"][:]

        noise_data_raw: Optional[np.ndarray] = None
        if "N" in grp:
            noise_data_raw = grp["N"]["data"][:].astype(np.float32)

    if p_data.shape[0] != s_data.shape[0]:
        raise ValueError(
            f"P and S channel counts differ: {p_data.shape[0]} vs {s_data.shape[0]}"
        )

    p_data, p_fixed = fix_invalid_values(p_data)
    s_data, s_fixed = fix_invalid_values(s_data)
    if noise_data_raw is not None:
        noise_data_raw, _ = fix_invalid_values(noise_data_raw)
    fixed_mask = p_fixed | s_fixed
    metadata["n_fixed"] = int(fixed_mask.sum())

    valid_mask = build_window_qc_mask(
        p_data, p_snr, s_data, s_snr,
        fixed_mask=fixed_mask,
        snr_min=snr_min,
        mask_zeros=mask_zeros,
    )

    return DASData(
        p_data=p_data,
        p_shift_index=p_shift_index,
        p_snr=p_snr,
        p_traveltime=p_traveltime,
        s_data=s_data,
        s_shift_index=s_shift_index,
        s_snr=s_snr,
        s_traveltime=s_traveltime,
        dt=dt,
        channel_coords=channel_coords,
        noise_data=noise_data_raw,
        valid_mask=valid_mask,
        event_id=str(attrs.get("event_id", "")),
        event_time=str(attrs.get("event_time", "")),
        latitude=float(attrs.get("latitude", np.nan)),
        longitude=float(attrs.get("longitude", np.nan)),
        depth_km=float(attrs.get("depth_km", np.nan)),
        magnitude=float(attrs.get("magnitude", np.nan)),
        metadata=metadata,
    )


def load_das_raw(
    filepath: str | Path,
    channel_file: Optional[str | Path] = None,
    channel_coords: Optional[np.ndarray] = None,
    mask_zeros: bool = True,
) -> DASRawData:
    """Load a continuous DAS HDF5 file into a :class:`~dasfm.io.data.DASRawData`.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.h5`` file.
    channel_file : str or Path, optional
        Path to a CSV file with a 1-based ``ichan`` column listing good
        channels (e.g. ``das_info.csv``).  When supplied, only those channels
        are kept; the returned ``n_channels`` equals the number of good
        channels, not the raw file's full channel count.
    channel_coords : np.ndarray, optional
        Channel positions in metres, shape ``(n_channels, 3)`` as ``[x, y, z]``.
        If ``None``, a 1-D along-fibre coordinate array is built from ``dx_m``
        before any channel selection is applied.
    mask_zeros : bool
        Flag channels that are entirely zero after NaN/Inf fixing (default ``True``).

    Returns
    -------
    DASRawData
        ``valid_mask[i]`` is ``True`` iff channel *i* passed all QC checks.
        ``metadata['n_fixed']`` reports how many channels had NaN/Inf zeroed.
        ``metadata['n_channels_raw']`` records the original full channel count
        when *channel_file* is used.

    Raises
    ------
    FileNotFoundError
    KeyError
        If the HDF5 file does not contain a ``data`` dataset.
    ValueError
        If *channel_file* contains indices out of range for the HDF5 data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        if "data" not in f:
            raise KeyError(
                f"Expected dataset 'data' in {filepath}, "
                f"found: {list(f.keys())}"
            )
        ds        = f["data"]
        waveforms = ds[:].astype(np.float32)
        # New format: attrs at file level; legacy: attrs on dataset
        attrs = dict(f.attrs) if len(f.attrs) > 0 else dict(ds.attrs)
        # Read traveltime datasets if present (new combined H5)
        _p_tt = f["p_traveltime"][:] if "p_traveltime" in f else None
        _s_tt = f["s_traveltime"][:] if "s_traveltime" in f else None

    dt             = float(attrs.get("dt_s", attrs.get("dt", 0.01)))
    dx_m           = float(attrs.get("dx_m", np.nan))
    eti            = int(attrs.get("event_time_index", 0))
    begin_time     = str(attrs.get("begin_time",     ""))
    end_time       = str(attrs.get("end_time",       ""))
    event_id       = str(attrs.get("event_id",       "")) or None
    event_time     = str(attrs.get("event_time",     "")) or None
    magnitude_type = str(attrs.get("magnitude_type", "")) or None
    unit           = str(attrs.get("unit",           "microstrain/s"))
    latitude       = float_attr(attrs, "latitude")
    longitude      = float_attr(attrs, "longitude")
    depth_km       = float_attr(attrs, "depth_km")
    magnitude      = float_attr(attrs, "magnitude")
    time_before    = float(attrs.get("time_before", 0.0))
    time_after     = float(attrs.get("time_after",  0.0))

    waveforms, fixed_mask = fix_invalid_values(waveforms)
    valid_mask = ~fixed_mask.copy()
    if mask_zeros:
        valid_mask &= np.any(waveforms != 0, axis=1)

    # build 1-D fibre coordinates from dx_m (before channel selection)
    if channel_coords is None and np.isfinite(dx_m):
        n_ch = waveforms.shape[0]
        along = np.arange(n_ch, dtype=np.float64) * dx_m
        channel_coords = np.column_stack(
            [along, np.zeros(n_ch), np.zeros(n_ch)]
        )

    metadata: dict = {
        "n_fixed": int(fixed_mask.sum()),
        "source_file": str(filepath),
    }
    if _p_tt is not None:
        metadata["p_traveltime"] = _p_tt
    if _s_tt is not None:
        metadata["s_traveltime"] = _s_tt

    # ── channel selection from CSV ─────────────────────────────────────────
    if channel_file is not None:
        good_no = load_good_channels(channel_file)
        n_raw = waveforms.shape[0]
        if good_no.max() >= n_raw:
            raise ValueError(
                f"channel_file contains index {good_no.max() + 1} (1-based) "
                f"but the HDF5 file has only {n_raw} channels."
            )
        waveforms  = waveforms[good_no]
        valid_mask = valid_mask[good_no]
        if channel_coords is not None:
            channel_coords = channel_coords[good_no]
        metadata["n_channels_raw"] = n_raw
        metadata["channel_file"]   = str(channel_file)

    return DASRawData(
        waveforms=waveforms,
        dt=dt,
        dx_m=dx_m,
        event_time_index=eti,
        begin_time=begin_time,
        end_time=end_time,
        event_id=event_id,
        event_time=event_time,
        latitude=latitude,
        longitude=longitude,
        depth_km=depth_km,
        magnitude=magnitude,
        magnitude_type=magnitude_type,
        unit=unit,
        time_before=time_before,
        time_after=time_after,
        channel_coords=channel_coords,
        valid_mask=valid_mask,
        metadata=metadata,
    )


# ===========================================================================
# Pre-flight validators
# ===========================================================================

def validate_das_raw(filepath) -> None:
    """Pre-flight check: a raw DAS HDF5 file exists and contains a 'data' dataset.

    Lightweight version of :func:`load_das_raw` — only opens the file long
    enough to verify the dataset is present.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the HDF5 file does not contain a 'data' dataset.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAS raw HDF5 file not found: {filepath}")
    with h5py.File(filepath, "r") as f:
        if "data" not in f:
            raise KeyError(
                f"Expected dataset 'data' in {filepath}, found: {list(f.keys())}"
            )


def validate_das_geo(filepath) -> None:
    """Pre-flight check: DAS geometry CSV exists and has required columns.

    Required columns: ``index``, ``latitude``, ``longitude``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAS geometry CSV not found: {filepath}")
    import pandas as pd
    header = set(pd.read_csv(filepath, nrows=0).columns)
    required = {"index", "latitude", "longitude"}
    missing = required - header
    if missing:
        raise KeyError(
            f"DAS geometry CSV {filepath} missing required columns: {sorted(missing)}"
        )


def validate_phase_picks(filepath) -> None:
    """Pre-flight check: per-event phase picks CSV exists and has required columns.

    Required columns: ``channel_index``, ``phase_index``, ``phase_type``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Phase picks CSV not found: {filepath}")
    import pandas as pd
    header = set(pd.read_csv(filepath, nrows=0).columns)
    required = {"channel_index", "phase_index", "phase_type"}
    missing = required - header
    if missing:
        raise KeyError(
            f"Phase picks CSV {filepath} missing required columns: {sorted(missing)}"
        )


# ===========================================================================
# Internal helpers
# ===========================================================================

def fix_invalid_values(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN and Inf with 0; return ``(fixed_data, was_fixed_mask)``."""
    data = data.copy()
    bad = ~np.isfinite(data)
    was_fixed = bad.any(axis=1)
    data[bad] = 0.0
    return data, was_fixed


def build_window_qc_mask(
    p_data: np.ndarray,
    p_snr: np.ndarray,
    s_data: np.ndarray,
    s_snr: np.ndarray,
    fixed_mask: np.ndarray,
    snr_min: float,
    mask_zeros: bool,
) -> np.ndarray:
    """Return boolean valid_mask (True = good channel).  Data is never touched."""
    valid = ~fixed_mask.copy()
    if snr_min > 0.0:
        valid &= (p_snr >= snr_min) & (s_snr >= snr_min)
    if mask_zeros:
        valid &= np.any(p_data != 0, axis=1)
        valid &= np.any(s_data != 0, axis=1)
    return valid


def float_attr(attrs: dict, key: str) -> Optional[float]:
    """Return ``float(attrs[key])`` or ``None`` if the key is absent."""
    val = attrs.get(key)
    return None if val is None else float(val)


def load_good_channels(channel_file: str | Path) -> np.ndarray:
    """Read a channel-info CSV and return 0-based good channel indices.

    The CSV must contain an ``index`` column with 0-based channel numbers.
    Other columns (e.g. ``latitude``, ``longitude``, ``elevation_m``) are ignored.

    Parameters
    ----------
    channel_file : str or Path
        Path to the CSV file (e.g. ``das_info.csv``).

    Returns
    -------
    np.ndarray
        Sorted 0-based integer indices of good channels.
    """
    try:
        import pandas as pd
        df = pd.read_csv(channel_file)
        good_no = df["index"].values.astype(int)
    except ImportError:
        raw = np.genfromtxt(channel_file, delimiter=",", names=True)
        good_no = raw["index"].astype(int)
    return np.sort(good_no)
