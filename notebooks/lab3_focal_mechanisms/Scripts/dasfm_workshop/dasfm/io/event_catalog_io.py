"""Seismic catalog reader.

Public function
---------------
* :func:`load_event_catalog` — read a CSV catalog into a ``pandas.DataFrame``.

Expected CSV columns (SKHASH catfile format)
---------------------------------------------
event_id, time, longitude, latitude, depth, magnitude

Legacy column names ``event_time`` and ``depth_km`` are also accepted
and automatically renamed to ``time`` and ``depth``.

Any additional columns are retained as-is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_event_catalog(filepath: str | Path) -> pd.DataFrame:
    """Read a seismic event catalog CSV into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        One row per event.  ``event_id`` is stored as ``str``;
        ``time`` is parsed as timezone-aware ``datetime64[ns, UTC]``.
        Columns use SKHASH names: ``time``, ``depth``.

    Raises
    ------
    FileNotFoundError
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Catalog file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Accept legacy column names
    if "event_time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"event_time": "time"})
    if "depth_km" in df.columns and "depth" not in df.columns:
        df = df.rename(columns={"depth_km": "depth"})

    required = {"event_id", "time", "longitude", "latitude", "depth"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Catalog CSV missing required columns: {sorted(missing)}")

    df["event_id"] = df["event_id"].astype(str)
    df["time"]     = pd.to_datetime(df["time"], utc=True)
    for col in ("longitude", "latitude", "depth"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "magnitude" in df.columns:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")

    return df.reset_index(drop=True)


def validate_event_catalog(filepath: str | Path) -> None:
    """Pre-flight check: catalog file exists and has the columns dasfm steps read.

    Required columns (matching what every step actually accesses):

    - ``event_id`` — universal across all steps
    - ``latitude``, ``longitude`` — used by step1 grid construction
    - ``depth`` or ``depth_km`` — used by step1 depth grid

    Note: this is intentionally looser than :func:`load_event_catalog`,
    which additionally requires ``time``. dasfm steps use ``pd.read_csv``
    directly and don't depend on ``time``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Event catalog not found: {filepath}")
    header = set(pd.read_csv(filepath, nrows=0).columns)
    required = {"event_id", "latitude", "longitude"}
    missing = required - header
    if missing:
        raise KeyError(
            f"Event catalog {filepath} missing required columns: {sorted(missing)}"
        )
    if not ({"depth", "depth_km"} & header):
        raise KeyError(
            f"Event catalog {filepath} missing depth column (need 'depth' or 'depth_km')"
        )


def validate_per_event_files(
    event_ids,
    directory: str | Path,
    suffix: str,
    label: str,
    upstream_step: str | None = None,
) -> None:
    """Pre-flight check: every event ID has its corresponding file on disk.

    Parameters
    ----------
    event_ids : iterable of str
        Event IDs from the catalog.
    directory : str or Path
        Directory expected to contain ``{eid}{suffix}`` for every event.
    suffix : str
        File extension including the dot, e.g. ``".h5"`` or ``".csv"``.
    label : str
        Human-readable label used in the error message (e.g. "raw H5").
    upstream_step : str, optional
        Name of the step that produces these files; mentioned in the
        error message as a hint when files are missing.

    Raises
    ------
    FileNotFoundError
        Listing the missing event IDs (truncated to first 5).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {directory}")
    eids = [str(eid) for eid in event_ids]
    missing = [eid for eid in eids if not (directory / f"{eid}{suffix}").exists()]
    if missing:
        head = missing[:5]
        more = "..." if len(missing) > 5 else ""
        hint = f"  Run {upstream_step} first." if upstream_step else ""
        raise FileNotFoundError(
            f"{label}: missing per-event files in {directory}\n"
            f"  Missing ({len(missing)}/{len(eids)}): {head}{more}{hint}"
        )
