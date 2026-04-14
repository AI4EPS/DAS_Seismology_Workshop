"""Geographic-to-Cartesian projection and regular-grid builder for traveltime modeling.

Coordinate system convention
-----------------------------
All Cartesian coordinates use a **right-handed, depth-positive** frame:

* ``x``  [km] — **northing**, positive North
* ``y``  [km] — **easting**,  positive East
* ``z``  [km] — **depth**,    positive down

Grid origin rules:

* *Horizontal* — The south-west (minimum-x, minimum-y) corner of the
  computation domain is taken as ``(x, y) = (0, 0)``.  All grid
  coordinates are therefore non-negative.
* *Vertical* — ``z = 0`` is set at the **highest terrain elevation** (``h_max``)
  within the computation domain.  A receiver at elevation ``h_rec`` [km a.s.l.]
  sits at model depth ``z_rec = h_max − h_rec ≥ 0``.  A seismic source
  at ``depth_km`` below sea level sits at ``z_src = h_max + depth_km``.

Topographic DEM input
---------------------
:func:`build_model_grid` accepts a regular-grid DEM as three arrays:

* ``topo_lat``    — 1-D latitude axis  [degrees], ascending
* ``topo_lon``    — 1-D longitude axis [degrees], ascending
* ``topo_elev_m`` — 2-D elevation grid [m], shape ``(nlat, nlon)``

These are typically read directly from a NetCDF / HDF5 file::

    import h5py
    with h5py.File("surface.nc", "r") as f:
        topo_lat   = f["lat"][:]
        topo_lon   = f["lon"][:]
        topo_elev_m = f["elevation"][:]

The function builds a bilinear interpolator from the DEM and:

1. Evaluates it at every node of the horizontal Cartesian grid to find ``h_max``
   (the peak elevation within the domain).
2. Evaluates it at each receiver position to compute ``receiver_z``.
3. Returns a ``topo_fn(x, y) → elevation [km]`` callable in the geo dict for
   use by the FSM forward model.

Flat-Earth projection
---------------------
Uses the equirectangular (plate carrée) approximation centred on the mean
position of all catalog events and receivers::

    x_raw = (lat − lat₀) × R_km × π/180
    y_raw = (lon − lon₀) × cos(lat₀ × π/180) × R_km × π/180

The SW-corner offset is then subtracted so the grid starts at (0, 0).
Accurate to < 0.1 % for domains ≲ 300 km across.

Public functions
----------------
* :func:`build_model_grid`
* :func:`latlon_to_xy`
* :func:`xy_to_latlon`
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from dasfm.utils.step_utils import log_or_print

EARTH_RADIUS_KM = 6371.0   # mean Earth radius [km]


# ===========================================================================
# Low-level coordinate conversion helpers
# ===========================================================================

def latlon_to_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert geographic coordinates to local flat-Earth Cartesian (x, y).

    Convention: **x = northing** (positive North), **y = easting** (positive East).

    Parameters
    ----------
    lat, lon : array-like
        Geographic latitude and longitude [degrees].
    origin_lat, origin_lon : float
        Reference point for the equirectangular projection [degrees].
    x_offset, y_offset : float, optional
        Subtract these from the raw projected coordinates.  Pass
        ``geo["x_offset"]`` and ``geo["y_offset"]`` from
        :func:`build_model_grid` to obtain grid-frame coordinates directly.

    Returns
    -------
    x, y : np.ndarray
        Northing and easting [km], after offset subtraction.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    cos_lat0 = np.cos(np.deg2rad(origin_lat))
    x = (lat - origin_lat) * EARTH_RADIUS_KM * np.pi / 180.0 - x_offset          # northing
    y = (lon - origin_lon) * cos_lat0 * EARTH_RADIUS_KM * np.pi / 180.0 - y_offset  # easting
    return x, y


def xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`latlon_to_xy` — Cartesian (x, y) back to (lat, lon).

    Parameters
    ----------
    x, y : array-like
        Northing and easting [km] in the grid frame.
    origin_lat, origin_lon : float
        Reference point used in the forward projection [degrees].
    x_offset, y_offset : float, optional
        Offsets that were subtracted in :func:`latlon_to_xy`; added back here.

    Returns
    -------
    lat, lon : np.ndarray
        Geographic coordinates [degrees].
    """
    x = np.asarray(x, dtype=np.float64) + x_offset
    y = np.asarray(y, dtype=np.float64) + y_offset
    cos_lat0 = np.cos(np.deg2rad(origin_lat))
    lat = x / (EARTH_RADIUS_KM * np.pi / 180.0) + origin_lat
    lon = y / (cos_lat0 * EARTH_RADIUS_KM * np.pi / 180.0) + origin_lon
    return lat, lon


# ===========================================================================
# Public API
# ===========================================================================

def build_model_grid(
    catalog: pd.DataFrame,
    receiver_lat: np.ndarray,
    receiver_lon: np.ndarray,
    topo_lat: np.ndarray | None = None,
    topo_lon: np.ndarray | None = None,
    topo_elev_m: np.ndarray | None = None,
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    depth_max: float = 15.0,
    margin_km: float = 2.0,
    logger=None,
) -> dict:
    """Project catalog and receiver positions onto a local Cartesian 3-D grid.

    Coordinate convention (see module docstring):

    * x — northing [km], x_grid[0] = 0  (SW corner)
    * y — easting  [km], y_grid[0] = 0  (SW corner)
    * z — depth    [km], z_grid[0] = 0  (model top = highest terrain elevation)

    Parameters
    ----------
    catalog : pd.DataFrame
        Event catalog.  Required columns: ``latitude``, ``longitude``,
        ``depth_km`` (below sea level, positive down).
    receiver_lat : array-like, shape (n_ch,)
        Geographic latitude of each DAS channel [degrees].
    receiver_lon : array-like, shape (n_ch,)
        Geographic longitude of each DAS channel [degrees].
    topo_lat : array-like, shape (nlat,), optional
        Latitude axis of the DEM [degrees].  Must be monotonically increasing
        (or will be flipped automatically).  If None (together with topo_lon
        and topo_elev_m), a flat surface at sea level is assumed (h_max = 0).
    topo_lon : array-like, shape (nlon,), optional
        Longitude axis of the DEM [degrees].  Must be monotonically increasing
        (or will be flipped automatically).
    topo_elev_m : array-like, shape (nlat, nlon), optional
        Terrain elevation [m, positive upward].  Row index corresponds to
        ``topo_lat``, column index to ``topo_lon``.
    dx : float
        Grid spacing in the x (north) direction [km].  Default 0.1.
    dy : float
        Grid spacing in the y (east) direction [km].  Default 0.1.
    dz : float
        Grid spacing in the z (depth) direction [km].  Default 0.1.
    depth_max : float
        Maximum source depth **below sea level** [km].  The z-grid extends
        from 0 (model top = h_max) to ``h_max + depth_max``.  Default 15.0.
    margin_km : float
        Horizontal padding added to all four sides of the bounding box
        enclosing all events and receivers [km].  Default 2.0.

    Returns
    -------
    dict with keys:

    **Projection reference**
        ``origin_lat``, ``origin_lon`` — equirectangular reference [degrees]

        ``x_offset``, ``y_offset`` — SW-corner raw projected northing/easting
        [km].  Pass to :func:`latlon_to_xy` as ``x_offset``/``y_offset`` to
        convert any lat/lon to grid-frame (x, y) directly.

        ``h_max`` — peak terrain elevation within the computation domain [km].

    **Topographic function**
        ``topo_fn`` — callable ``topo_fn(x, y) → elevation [km]`` in the
        grid frame (x = northing, y = easting).  Pass directly to
        :func:`~dasfm.forward.ray_lookup_2d.compute_ray_lookup`.

    **Grid axes** (1-D arrays)
        ``x_grid`` (nx,) — northing [km], x_grid[0] = 0
        ``y_grid`` (ny,) — easting  [km], y_grid[0] = 0
        ``z_grid`` (nz,) — depth    [km], z_grid[0] = 0

    **Grid metadata**
        ``dx``, ``dy``, ``dz`` — grid spacings [km]
        ``nx``, ``ny``, ``nz`` — number of grid nodes per axis

    **Terrain surface index**
        ``surface_iz`` (nx, ny) int64 — index into ``z_grid`` of the terrain
        surface at each horizontal grid node.
        ``z_grid[surface_iz[ix, iy]] ≈ h_max − h(ix, iy)``.
        Nodes with ``iz < surface_iz[ix, iy]`` are above the surface (air).

    **Source positions** (n_ev,)
        ``source_x``, ``source_y``, ``source_z`` — physical [km]
        ``source_ix``, ``source_iy``, ``source_iz`` — nearest grid indices

    **Receiver positions** (n_ch,)
        ``receiver_x``, ``receiver_y``, ``receiver_z`` — physical [km]
        ``receiver_ix``, ``receiver_iy``, ``receiver_iz`` — nearest grid indices
    """
    receiver_lat = np.asarray(receiver_lat, dtype=np.float64).ravel()
    receiver_lon = np.asarray(receiver_lon, dtype=np.float64).ravel()
    n_ch = receiver_lat.size

    ev_lat   = catalog["latitude"].values.astype(np.float64)
    ev_lon   = catalog["longitude"].values.astype(np.float64)
    _depth_col = "depth" if "depth" in catalog.columns else "depth_km"
    ev_depth = catalog[_depth_col].values.astype(np.float64)  # below sea level

    # ── equirectangular reference point (centroid for projection accuracy) ────
    all_lat = np.concatenate([ev_lat, receiver_lat])
    all_lon = np.concatenate([ev_lon, receiver_lon])
    origin_lat = float(np.mean(all_lat))
    origin_lon = float(np.mean(all_lon))

    # ── raw projected coordinates (x = north, y = east, no offset yet) ───────
    ev_x_raw, ev_y_raw = latlon_to_xy(ev_lat,       ev_lon,       origin_lat, origin_lon)
    rx_raw,   ry_raw   = latlon_to_xy(receiver_lat, receiver_lon, origin_lat, origin_lon)

    all_x_raw = np.concatenate([ev_x_raw, rx_raw])
    all_y_raw = np.concatenate([ev_y_raw, ry_raw])

    # ── SW corner: floor to nearest grid-spacing multiple ────────────────────
    x_offset = float(np.floor((all_x_raw.min() - margin_km) / dx) * dx)
    y_offset = float(np.floor((all_y_raw.min() - margin_km) / dy) * dy)

    # ── shift to SW-corner origin (all coordinates ≥ 0) ──────────────────────
    ev_x = ev_x_raw - x_offset
    ev_y = ev_y_raw - y_offset
    rx   = rx_raw   - x_offset
    ry   = ry_raw   - y_offset

    # ── horizontal grid extent (NE corner rounded up) ─────────────────────────
    all_x = np.concatenate([ev_x, rx])
    all_y = np.concatenate([ev_y, ry])
    x_ne  = np.ceil((all_x.max() + margin_km) / dx) * dx
    y_ne  = np.ceil((all_y.max() + margin_km) / dy) * dy

    nx = int(np.round(x_ne / dx)) + 1
    ny = int(np.round(y_ne / dy)) + 1
    x_grid = np.arange(nx, dtype=np.float64) * dx
    y_grid = np.arange(ny, dtype=np.float64) * dy

    # ── Topography handling ──────────────────────────────────────────────────
    _has_topo = (topo_lat is not None and topo_lon is not None
                 and topo_elev_m is not None)

    if _has_topo:
        # ── DEM bilinear interpolator ─────────────────────────────────────────
        t_lat  = np.asarray(topo_lat,   dtype=np.float64).ravel()
        t_lon  = np.asarray(topo_lon,   dtype=np.float64).ravel()
        t_elev = np.asarray(topo_elev_m, dtype=np.float64)

        if t_elev.ndim != 2 or t_elev.shape != (len(t_lat), len(t_lon)):
            raise ValueError(
                f"topo_elev_m shape {t_elev.shape} must be (nlat={len(t_lat)}, nlon={len(t_lon)})"
            )

        # Ensure strictly increasing axes (RegularGridInterpolator requirement)
        if t_lat[0] > t_lat[-1]:
            t_lat  = t_lat[::-1]
            t_elev = t_elev[::-1, :]
        if t_lon[0] > t_lon[-1]:
            t_lon  = t_lon[::-1]
            t_elev = t_elev[:, ::-1]

        # Bilinear DEM interpolator in lat/lon space.
        dem_rgi = RegularGridInterpolator(
            (t_lat, t_lon), t_elev,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # ── DEM coverage check ────────────────────────────────────────────────
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid, indexing="ij")
        lat_grid, lon_grid = xy_to_latlon(
            xx_grid.ravel(), yy_grid.ravel(),
            origin_lat, origin_lon, x_offset, y_offset,
        )

        lat_check = np.concatenate([lat_grid,    receiver_lat])
        lon_check = np.concatenate([lon_grid,    receiver_lon])
        n_grid_pts = lat_grid.size

        lat_lo, lat_hi = float(t_lat[0]),  float(t_lat[-1])
        lon_lo, lon_hi = float(t_lon[0]),  float(t_lon[-1])

        outside = (
            (lat_check < lat_lo) | (lat_check > lat_hi) |
            (lon_check < lon_lo) | (lon_check > lon_hi)
        )
        if outside.any():
            n_grid_out = int(outside[:n_grid_pts].sum())
            n_rec_out  = int(outside[n_grid_pts:].sum())
            warnings.warn(
                f"[build_model_grid] Computation domain exceeds DEM extent: "
                f"grid nodes {n_grid_out}/{n_grid_pts}, "
                f"receivers {n_rec_out}/{len(receiver_lat)} out of range "
                f"(DEM latitude [{lat_lo:.4f}, {lat_hi:.4f}]°, "
                f"longitude [{lon_lo:.4f}, {lon_hi:.4f}]°). "
                f"Out-of-range nodes will use nearest-edge elevation.",
                UserWarning,
                stacklevel=2,
            )

        # ── Clamped DEM query helper ──────────────────────────────────────────
        _lat_lo, _lat_hi = lat_lo, lat_hi
        _lon_lo, _lon_hi = lon_lo, lon_hi
        _dem_rgi = dem_rgi

        def _query_dem(lat_q: np.ndarray, lon_q: np.ndarray) -> np.ndarray:
            """Query DEM [m] with lat/lon clamped to DEM extent."""
            lat_c = np.clip(lat_q, _lat_lo, _lat_hi)
            lon_c = np.clip(lon_q, _lon_lo, _lon_hi)
            return _dem_rgi(np.column_stack([lat_c, lon_c]))

        # ── h_max: peak DEM elevation within the computation domain ──────────
        elev_grid_m = _query_dem(lat_grid, lon_grid)
        h_max = float(elev_grid_m.max()) / 1000.0

        # ── receiver elevations from DEM ──────────────────────────────────────
        rec_elev_m  = _query_dem(receiver_lat, receiver_lon)
        rec_elev_km = rec_elev_m / 1000.0

        # ── topo_fn: (x_grid_array, y_grid_array) → elevation [km] ──────────
        _origin_lat = origin_lat
        _origin_lon = origin_lon
        _x_offset   = x_offset
        _y_offset   = y_offset

        def topo_fn(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
            """Return terrain elevation [km, positive up] at grid-frame (x, y)."""
            x_arr = np.asarray(x_array, dtype=np.float64).ravel()
            y_arr = np.asarray(y_array, dtype=np.float64).ravel()
            lat_q, lon_q = xy_to_latlon(
                x_arr, y_arr, _origin_lat, _origin_lon, _x_offset, _y_offset
            )
            return _query_dem(lat_q, lon_q) / 1000.0

    else:
        # ── No topography: flat surface at sea level ──────────────────────────
        h_max = 0.0
        rec_elev_km = np.zeros(n_ch, dtype=np.float64)
        elev_grid_m = np.zeros(nx * ny, dtype=np.float64)

        def topo_fn(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
            """Flat surface at sea level — returns 0 everywhere."""
            return np.zeros(np.asarray(x_array).size, dtype=np.float64)

    # ── terrain and depth axis ────────────────────────────────────────────────
    z_max  = h_max + depth_max          # deepest model point below model top [km]
    nz     = int(np.ceil(z_max / dz)) + 1
    z_grid = np.arange(nz, dtype=np.float64) * dz

    # ── z coordinates in model frame (depth below model top) ─────────────────
    rec_z = h_max - rec_elev_km   # receiver depth [km];  0 ≤ rec_z ≤ h_max
    ev_z  = h_max + ev_depth      # source depth   [km];  ≥ h_max

    # ── nearest grid-node indices ─────────────────────────────────────────────
    src_ix = snap_to_grid(ev_x,  x_grid, dx, nx)
    src_iy = snap_to_grid(ev_y,  y_grid, dy, ny)
    src_iz = snap_to_grid(ev_z,  z_grid, dz, nz)
    rec_ix = snap_to_grid(rx,    x_grid, dx, nx)
    rec_iy = snap_to_grid(ry,    y_grid, dy, ny)
    rec_iz = snap_to_grid(rec_z, z_grid, dz, nz)

    # ── surface z-index: surface_iz[ix, iy] ──────────────────────────────────
    # elev_grid_m is already available: shape (nx*ny,), same row-major order as
    # the (nx, ny) meshgrid built with indexing="ij" above.
    surf_depth_flat = h_max - elev_grid_m / 1000.0           # [km], (nx*ny,)
    surface_iz = snap_to_grid(surf_depth_flat, z_grid, dz, nz).reshape(nx, ny)

    log_or_print(logger,
        f"[build_model_grid]  origin ({origin_lat:.5f}°N, {origin_lon:.5f}°E)\n"
        f"  x (north) [0.00 … {x_grid[-1]:.2f}] km  nx={nx}  dx={dx} km\n"
        f"  y (east)  [0.00 … {y_grid[-1]:.2f}] km  ny={ny}  dy={dy} km\n"
        f"  z (depth) [0.00 … {z_grid[-1]:.2f}] km  nz={nz}  dz={dz} km\n"
        f"  h_max={h_max:.3f} km  (z=0 ↔ {h_max:.3f} km a.s.l., "
        f"sea level at z={h_max:.3f} km)\n"
        f"  n_events={len(ev_x)}  n_receivers={n_ch}"
    )

    return {
        # projection reference
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "x_offset":   x_offset,    # raw-frame northing of SW corner [km]
        "y_offset":   y_offset,    # raw-frame easting  of SW corner [km]
        "h_max":      h_max,       # peak terrain elevation in domain [km a.s.l.]
        # grid axes
        "x_grid":     x_grid,
        "y_grid":     y_grid,
        "z_grid":     z_grid,
        # grid metadata
        "dx": dx,  "dy": dy,  "dz": dz,
        "nx": nx,  "ny": ny,  "nz": nz,
        # surface z-index map
        "surface_iz": surface_iz,    # (nx, ny) int64 — z_grid index of terrain surface
        # source positions
        "source_x":    ev_x,
        "source_y":    ev_y,
        "source_z":    ev_z,
        "source_ix":   src_ix,
        "source_iy":   src_iy,
        "source_iz":   src_iz,
        # receiver positions
        "receiver_x":  rx,
        "receiver_y":  ry,
        "receiver_z":  rec_z,
        "receiver_ix": rec_ix,
        "receiver_iy": rec_iy,
        "receiver_iz": rec_iz,
    }


# ===========================================================================
# Private helpers
# ===========================================================================

def snap_to_grid(vals: np.ndarray, grid: np.ndarray, step: float, n: int) -> np.ndarray:
    """Return 0-based nearest-node indices, clamped to [0, n-1]."""
    idx = np.round((vals - grid[0]) / step).astype(np.int64)
    return np.clip(idx, 0, n - 1)


# ===========================================================================
# Source–receiver geometry helpers
# ===========================================================================

def compute_azimuth(source_x, source_y, receiver_x, receiver_y):
    """Azimuth (rad) from each source to each receiver. Shape (n_ev, n_rec)."""
    return np.arctan2(
        receiver_y[None, :] - source_y[:, None],
        receiver_x[None, :] - source_x[:, None],
    ).astype(np.float32)


def compute_fiber_orientation(receiver_x, receiver_y):
    """Fiber cable orientation angle (degrees) per channel. Shape (n_ch,).

    Uses forward difference; last channel copies from second-to-last.
    """
    dx = np.diff(receiver_x, append=receiver_x[-1])
    dy = np.diff(receiver_y, append=receiver_y[-1])
    # Last channel has dx=dy=0; copy from second-to-last
    angles = np.degrees(np.arctan2(dy, dx)).astype(np.float32)
    if len(angles) > 1:
        angles[-1] = angles[-2]
    return angles
