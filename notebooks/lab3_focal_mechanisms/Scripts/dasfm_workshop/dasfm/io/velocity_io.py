"""velocity_io — Load velocity model files."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_velocity_1d(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column 1-D P-velocity model from a text file.

    Parameters
    ----------
    filepath : str or Path
        Whitespace-delimited text file with columns ``depth_km``  ``vp_km_s``.
        Lines starting with ``#`` or ``%`` are skipped as comments.
        Comma-delimited files are also accepted.

    Returns
    -------
    depth_km : np.ndarray, shape (n,)
        Depth nodes [km], positive downward, monotonically increasing.
    vp_km_s : np.ndarray, shape (n,)
        P-wave velocity [km/s].
    """
    with open(filepath) as _f:
        lines = [l.strip() for l in _f if l.strip()
                 and not l.strip()[0] in ('#', '%')
                 and l.strip()[0].isdigit()]
    import io
    text = "\n".join(lines)
    try:
        data = np.loadtxt(io.StringIO(text))
    except ValueError:
        data = np.loadtxt(io.StringIO(text), delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def validate_velocity_1d(filepath: str | Path) -> None:
    """Pre-flight check: 1-D velocity file exists and is parseable.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no numeric rows.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"1D velocity file not found: {filepath}")
    with open(filepath) as _f:
        for line in _f:
            s = line.strip()
            if not s or s[0] in ("#", "%"):
                continue
            if s[0].isdigit() or s[0] in ("+", "-", "."):
                return  # at least one numeric line — looks valid
    raise ValueError(f"1D velocity file has no numeric rows: {filepath}")
