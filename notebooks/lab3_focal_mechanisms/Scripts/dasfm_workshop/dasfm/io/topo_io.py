"""topo_io.py — Load topography from GeoTIFF or HDF5.

Supports:
- GeoTIFF (.tif, .tiff) — downloaded from https://opentopography.org/
- HDF5 (.h5, .hdf5) — converted with dasfm/tools/convert_topo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_topo(filepath):
    """Load topography from GeoTIFF (.tif/.tiff) or HDF5 (.h5).

    Parameters
    ----------
    filepath : str or Path
        Path to a GeoTIFF or HDF5 topography file.

    Returns
    -------
    lat : (n_lat,) float64 — latitude array (ascending)
    lon : (n_lon,) float64 — longitude array (ascending)
    elevation : (n_lat, n_lon) float32 — elevation in meters
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio is required for GeoTIFF topography. "
                "Install with: pip install rasterio")
        with rasterio.open(filepath) as src:
            elev = src.read(1).astype(np.float32)
            transform = src.transform
            n_lat, n_lon = elev.shape
        lon = transform.c + np.arange(n_lon) * transform.a + transform.a / 2
        lat = transform.f + np.arange(n_lat) * transform.e + transform.e / 2
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            elev = elev[::-1, :]

    elif suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(filepath, "r") as f:
            lon = f["lon"][:]
            lat = f["lat"][:]
            elev = f["elevation"][:]

    else:
        raise ValueError(
            f"Unsupported topography format: {suffix!r}. "
            f"Use GeoTIFF (.tif) or HDF5 (.h5)")

    return lat.astype(np.float64), lon.astype(np.float64), elev.astype(np.float32)


def validate_topo(filepath) -> None:
    """Pre-flight check: topography file exists and has a supported suffix.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the suffix is not one of ``.tif``, ``.tiff``, ``.h5``, ``.hdf5``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Topography file not found: {filepath}")
    suffix = filepath.suffix.lower()
    if suffix not in (".tif", ".tiff", ".h5", ".hdf5"):
        raise ValueError(
            f"Unsupported topography format: {suffix!r} (file: {filepath}). "
            f"Use GeoTIFF (.tif/.tiff) or HDF5 (.h5/.hdf5)."
        )
