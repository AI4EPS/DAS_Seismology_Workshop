"""P-wave polarity reader.

Public function
---------------
* :func:`load_polarity` — read a per-channel polarity CSV into a
  ``pandas.DataFrame``.

Expected CSV columns
--------------------
index                     : int   — 0-based channel index (matches das_info.csv)
status                    : str   — channel quality flag (e.g. ``good``)
latitude, longitude       : float — channel coordinates [degrees]
elevation_m               : float — channel elevation [m]
azimuth                   : float — ray azimuth at channel [degrees]
dip                       : float — ray dip angle at channel [degrees]
phase_polarity            : float — polarity amplitude (corrected)
phase_type                : str   — phase identifier (e.g. ``P``)
phase_polarity_no_correction : float — polarity amplitude (uncorrected)

Any additional columns are retained as-is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_polarity(filepath: str | Path) -> pd.DataFrame:
    """Read a per-channel polarity CSV into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file (e.g. ``73477486.csv``).

    Returns
    -------
    pandas.DataFrame
        One row per channel.  Numeric columns are coerced to ``float64``;
        ``index`` is stored as ``int``.  Rows are sorted by ``index``.

    Raises
    ------
    FileNotFoundError
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Polarity file not found: {filepath}")

    df = pd.read_csv(filepath)

    required = {"index", "phase_polarity", "phase_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Polarity CSV missing required columns: {sorted(missing)}")

    df["index"] = df["index"].astype(int)

    for col in ("latitude", "longitude", "elevation_m",
                "azimuth", "dip",
                "phase_polarity", "phase_polarity_no_correction"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("index").reset_index(drop=True)
