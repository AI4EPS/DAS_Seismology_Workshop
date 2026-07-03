"""convert_topo — Convert GeoTIFF topography to dasfm HDF5 format.

Topography data can be downloaded from https://opentopography.org/
(e.g., SRTM GL1 30m, SRTM GL3 90m, or other DEM sources).

Output HDF5 format:
    lon         (n_lon,)        float64   Longitude array (ascending)
    lat         (n_lat,)        float64   Latitude array (ascending)
    elevation   (n_lat, n_lon)  float32   Elevation in meters

Usage:
    1. Edit INPUT_TIF and OUTPUT_H5 below
    2. Run: python convert_topo.py
"""

import h5py
import numpy as np
import rasterio

# ── Edit these paths ──────────────────────────────────────────────────────
INPUT_TIF = "output_SRTMGL1.tif"     # Downloaded GeoTIFF from OpenTopography
OUTPUT_H5 = "input/topo.h5"          # Output HDF5 for dasfm
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with rasterio.open(INPUT_TIF) as src:
        elev = src.read(1).astype(np.float32)  # (n_lat, n_lon)
        transform = src.transform
        n_lat, n_lon = elev.shape

    # Build 1D coordinate arrays from the affine transform (pixel center)
    lon = transform.c + np.arange(n_lon) * transform.a + transform.a / 2
    lat = transform.f + np.arange(n_lat) * transform.e + transform.e / 2

    # dasfm expects lat in ascending order
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        elev = elev[::-1, :]

    # Save
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("lon", data=lon.astype(np.float64))
        f.create_dataset("lat", data=lat.astype(np.float64))
        f.create_dataset("elevation", data=elev)

    print(f"Converted: {INPUT_TIF}")
    print(f"  lon: {lon.shape} [{lon[0]:.4f}, {lon[-1]:.4f}]")
    print(f"  lat: {lat.shape} [{lat[0]:.4f}, {lat[-1]:.4f}]")
    print(f"  elevation: {elev.shape} [{elev.min():.1f}, {elev.max():.1f}] m")
    print(f"  -> {OUTPUT_H5}")
