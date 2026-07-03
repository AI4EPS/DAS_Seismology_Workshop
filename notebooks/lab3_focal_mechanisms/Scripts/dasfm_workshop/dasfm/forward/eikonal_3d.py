"""3D Eikonal solver for single-receiver traveltime, takeoff, and ray distance.

Wraps pykonal to compute a complete source-parameter lookup for one receiver,
analogous to ``compute_ray_lookup()`` in the 2D FSM path.

Supports loading 3D tomographic velocity models (e.g. Biondi et al. 2023 npz)
and interpolating them onto per-receiver local Cartesian grids.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator


# ── 3D tomographic velocity model loading and interpolation ──────────────────

def validate_velocity_3d(vel_3d_path):
    """Pre-flight check: 3-D tomographic velocity .npz file exists and has required keys.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required arrays (vp, boundbox, ds, oz) are missing.
    """
    vel_3d_path = Path(vel_3d_path)
    if not vel_3d_path.exists():
        raise FileNotFoundError(f"3D velocity model not found: {vel_3d_path}")
    with np.load(str(vel_3d_path), allow_pickle=True) as d:
        keys = set(d.files)
    required = {"vp", "boundbox", "ds", "oz"}
    missing = required - keys
    if missing:
        raise KeyError(
            f"3D velocity model {vel_3d_path} missing required arrays: {sorted(missing)}"
        )


def load_velocity_3d(vel_3d_path):
    """Load a 3D tomographic velocity model from .npz file.

    Expected npz keys:
        vp       : (nlat, nlon, nz) float — P-wave velocity [km/s]
        vs       : (nlat, nlon, nz) float — S-wave velocity [km/s] (optional)
        boundbox : (4,) float — [minLon, maxLon, minLat, maxLat]
        ds       : (3,) float — [dy, dx, dz] grid spacing [km]
        oz       : scalar float — z origin [km], negative = above sea level

    Returns
    -------
    dict with keys:
        rgi_vp   : RegularGridInterpolator for Vp in (lat, lon, z_tomo) space
        lat_axis : (nlat,) float64 — latitude axis [degrees]
        lon_axis : (nlon,) float64 — longitude axis [degrees]
        z_axis   : (nz,) float64 — depth axis [km], z=0 = sea level
        vp_min, vp_max : float — velocity range for logging
    """
    vel_3d_path = Path(vel_3d_path)
    if not vel_3d_path.exists():
        raise FileNotFoundError(f"3D velocity model not found: {vel_3d_path}")

    d = np.load(str(vel_3d_path), allow_pickle=True)
    vp = d['vp']
    bb = d['boundbox']   # [minLon, maxLon, minLat, maxLat]
    ds = d['ds']         # [dy, dx, dz] km
    oz = float(d['oz'])  # top of model (negative = above sea level)

    nlat, nlon, nz = vp.shape
    lat_axis = np.linspace(bb[2], bb[3], nlat)
    lon_axis = np.linspace(bb[0], bb[1], nlon)
    z_axis = oz + np.arange(nz) * ds[2]  # z=0 is sea level

    rgi_vp = RegularGridInterpolator(
        (lat_axis, lon_axis, z_axis), vp,
        method="linear", bounds_error=False, fill_value=None,
    )

    return {
        'rgi_vp': rgi_vp,
        'lat_axis': lat_axis,
        'lon_axis': lon_axis,
        'z_axis': z_axis,
        'vp_min': float(vp.min()),
        'vp_max': float(vp.max()),
    }


def estimate_eikonal_memory(nx, ny, nz, num_workers=1):
    """Estimate per-worker and total RAM for 3D Eikonal computation.

    Returns (per_worker_gb, total_gb, available_gb).
    """
    import psutil
    n_nodes = nx * ny * nz
    # Per-worker: vp_3d + meshgrid(3) + pts(3) + pykonal(~4) + gradients(3) + ray_dist(2) + output(4)
    # ≈ 12 * n_nodes * 8 bytes (float64) + 4 * n_nodes * 4 bytes (float32 output)
    per_worker_bytes = (12 * n_nodes * 8 + 4 * n_nodes * 4) * 1.2  # 1.2x safety margin
    per_worker_gb = per_worker_bytes / 1e9
    total_gb = per_worker_gb * num_workers
    available_gb = psutil.virtual_memory().available / 1e9
    return per_worker_gb, total_gb, available_gb


def check_eikonal_memory(geo, dr, num_workers, logger):
    """Check if 3D Eikonal computation fits in available RAM.

    Raises RuntimeError if estimated total exceeds 80% of available RAM.
    """
    nx = int(np.ceil(geo['x_grid'][-1] / dr)) + 1
    ny = int(np.ceil(geo['y_grid'][-1] / dr)) + 1
    nz = int(np.ceil(geo['z_grid'][-1] / dr)) + 1

    per_worker_gb, total_gb, available_gb = estimate_eikonal_memory(nx, ny, nz, num_workers)
    logger.info(f"  Memory estimate: grid ({nx}x{ny}x{nz}), "
                f"{per_worker_gb:.1f} GB/worker x {num_workers} = {total_gb:.1f} GB total, "
                f"available: {available_gb:.1f} GB")

    if total_gb > 0.8 * available_gb:
        raise RuntimeError(
            f"3D Eikonal would use ~{total_gb:.1f} GB RAM "
            f"({per_worker_gb:.1f} GB/worker x {num_workers} workers), "
            f"but only {available_gb:.1f} GB available. "
            f"Reduce num_cpu_workers or increase grid_spacing_km (dr={dr}).")


def build_receiver_grid(geo, receiver_x, receiver_y, receiver_z, dr):
    """Build a local Cartesian grid with receiver exactly on a node.

    Shifts the global grid origin so the receiver position falls precisely
    on a grid node, eliminating snap_to_grid precision loss (±dr/2).

    Parameters
    ----------
    geo : dict — output of build_model_grid()
    receiver_x, receiver_y, receiver_z : float — exact receiver position [km]
    dr : float — grid spacing [km]

    Returns
    -------
    x_grid, y_grid, z_grid : 1D arrays — shifted grid axes
    rec_ix, rec_iy, rec_iz : int — receiver indices (exact)
    nx, ny, nz : int — grid dimensions
    """
    x_max = geo['x_grid'][-1]
    y_max = geo['y_grid'][-1]
    z_max = geo['z_grid'][-1]

    # Shift: grid starts at receiver_coord % dr, so receiver is on a node
    x_start = receiver_x % dr
    y_start = receiver_y % dr
    z_start = receiver_z % dr

    nx = int(np.ceil((x_max - x_start) / dr)) + 1
    ny = int(np.ceil((y_max - y_start) / dr)) + 1
    nz = int(np.ceil((z_max - z_start) / dr)) + 1

    x_grid = x_start + np.arange(nx, dtype=np.float64) * dr
    y_grid = y_start + np.arange(ny, dtype=np.float64) * dr
    z_grid = z_start + np.arange(nz, dtype=np.float64) * dr

    rec_ix = int(round((receiver_x - x_start) / dr))
    rec_iy = int(round((receiver_y - y_start) / dr))
    rec_iz = int(round((receiver_z - z_start) / dr))

    return x_grid, y_grid, z_grid, rec_ix, rec_iy, rec_iz, nx, ny, nz


def interpolate_tomo_to_grid(tomo, geo, x_grid, y_grid, z_grid):
    """Interpolate 3D tomo Vp onto a local Cartesian grid.

    Parameters
    ----------
    tomo : dict — output of load_velocity_3d(), must contain 'rgi_vp'
    geo : dict — output of build_model_grid(), must contain
        origin_lat, origin_lon, x_offset, y_offset, h_max
    x_grid, y_grid, z_grid : 1D arrays — local grid axes [km]

    Returns
    -------
    vp_3d : (nx, ny, nz) float64 — velocity on the local grid
    """
    from dasfm.forward.geometry import xy_to_latlon

    nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)

    # Build 3D meshgrid of query points
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    # shape: (nx, ny, nz) each

    # Convert (x, y) → (lat, lon)
    lat_q, lon_q = xy_to_latlon(
        xx.ravel(), yy.ravel(),
        geo['origin_lat'], geo['origin_lon'],
        geo['x_offset'], geo['y_offset'],
    )

    # Convert z (depth below model top) → z_tomo (depth relative to sea level)
    # dasfm: z=0 is h_max (highest terrain), z positive downward
    # tomo:  z=0 is sea level, z positive downward
    z_tomo = zz.ravel() - geo['h_max']

    # Query tomo interpolator
    pts = np.column_stack([lat_q, lon_q, z_tomo])
    vp_flat = tomo['rgi_vp'](pts)
    vp_3d = vp_flat.reshape(nx, ny, nz)

    return vp_3d


# ── Numba-compiled core for ray distance ──────────────────────────────────────

@njit(cache=True)
def ray_distance_3d_core(order, r_flat,
                           ix_up, iy_up, iz_up,
                           delta_x, delta_y, delta_z,
                           _nx, _ny, _nz,
                           src_ix, src_iy, src_iz, near_r2):
    nyz = _ny * _nz
    for i in range(order.shape[0]):
        fk = order[i]
        ix = fk // nyz
        iy = (fk % nyz) // _nz
        iz = fk % _nz
        dix = ix - src_ix
        diy = iy - src_iy
        diz = iz - src_iz
        if dix * dix + diy * diy + diz * diz <= near_r2:
            continue
        r_x = r_flat[ix_up[fk] * nyz + iy * _nz + iz] + delta_x[fk]
        r_y = r_flat[ix * nyz + iy_up[fk] * _nz + iz] + delta_y[fk]
        r_z = r_flat[ix * nyz + iy * _nz + iz_up[fk]] + delta_z[fk]
        r_new = min(r_x, min(r_y, r_z))
        if r_new < r_flat[fk]:
            r_flat[fk] = r_new


def ray_distance_3d(TT, vp, src_ix, src_iy, src_iz, dh, near_cells=5):
    """Compute ray distance field from Eikonal traveltime using gradient tracing."""
    nx, ny, nz = TT.shape
    dT_dx = np.gradient(TT, dh, axis=0)
    dT_dy = np.gradient(TT, dh, axis=1)
    dT_dz = np.gradient(TT, dh, axis=2)

    ix_arr = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz)).ravel()
    iy_arr = np.broadcast_to(np.arange(ny)[None, :, None], (nx, ny, nz)).ravel()
    iz_arr = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz)).ravel()

    ix_up = np.where(dT_dx.ravel() >= 0,
                     np.clip(ix_arr - 1, 0, nx - 1),
                     np.clip(ix_arr + 1, 0, nx - 1)).astype(np.int64)
    iy_up = np.where(dT_dy.ravel() >= 0,
                     np.clip(iy_arr - 1, 0, ny - 1),
                     np.clip(iy_arr + 1, 0, ny - 1)).astype(np.int64)
    iz_up = np.where(dT_dz.ravel() >= 0,
                     np.clip(iz_arr - 1, 0, nz - 1),
                     np.clip(iz_arr + 1, 0, nz - 1)).astype(np.int64)

    delta_x = (vp * np.abs(dT_dx) * dh).ravel()
    delta_y = (vp * np.abs(dT_dy) * dh).ravel()
    delta_z = (vp * np.abs(dT_dz) * dh).ravel()

    r_flat = np.full(nx * ny * nz, np.inf, dtype=np.float64)
    nc = near_cells
    for dix in range(-nc, nc + 1):
        for diy in range(-nc, nc + 1):
            for diz in range(-nc, nc + 1):
                if dix * dix + diy * diy + diz * diz > nc * nc:
                    continue
                jx = src_ix + dix
                jy = src_iy + diy
                jz = src_iz + diz
                if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                    r_flat[jx * ny * nz + jy * nz + jz] = (
                        np.sqrt(dix**2 + diy**2 + diz**2) * dh
                    )

    order = np.argsort(TT.ravel()).astype(np.int64)
    ray_distance_3d_core(order, r_flat, ix_up, iy_up, iz_up,
                           delta_x, delta_y, delta_z,
                           np.int64(nx), np.int64(ny), np.int64(nz),
                           np.int64(src_ix), np.int64(src_iy), np.int64(src_iz),
                           np.int64(nc * nc))
    return r_flat.reshape(nx, ny, nz).astype(np.float32)


# ── Public entry point ────────────────────────────────────────────────────────

def compute_eikonal_lookup(vp_3d, nx, ny, nz, dr,
                             receiver_ix, receiver_iy, receiver_iz,
                             x_grid, y_grid, z_grid, geo):
    """Solve 3D Eikonal for a single receiver and return a RayParamDB.

    Parameters
    ----------
    vp_3d : (nx, ny, nz) float64 — 3D velocity model
    nx, ny, nz : int — grid dimensions
    dr : float — grid spacing (km)
    receiver_ix, receiver_iy, receiver_iz : int — receiver grid indices
    x_grid, y_grid, z_grid : 1D arrays — grid axes
    geo : dict — geometry dict (for origin_lat/lon, x/y_offset, h_max)

    Returns
    -------
    db : RayParamDB (geometry="cart_3d")
        Per-receiver traveltime / takeoff / raypath-length / azimuth fields
        on the local Cartesian grid. Ready to persist via ``db.to_hdf5``.
    timing_str : str
        Timing breakdown (init, eikonal, takeoff, ray-dist).
    """
    import pykonal

    six = int(receiver_ix)
    siy = int(receiver_iy)
    siz = int(receiver_iz)

    t0 = time.perf_counter()
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0.0, 0.0, 0.0
    solver.velocity.node_intervals = dr, dr, dr
    solver.velocity.npts = nx, ny, nz
    solver.velocity.values = vp_3d.copy()
    src_idx = (six, siy, siz)
    solver.traveltime.values[src_idx] = 0.0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver.solve()
    TT = solver.traveltime.values.copy()
    t_eikonal = time.perf_counter() - t0

    # Takeoff angle and azimuth from traveltime gradient
    t0 = time.perf_counter()
    dT_dx = np.gradient(TT, dr, axis=0)
    dT_dy = np.gradient(TT, dr, axis=1)
    dT_dz = np.gradient(TT, dr, axis=2)
    grad_mag = np.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2) + 1e-12
    sx_f = dT_dx / grad_mag
    sy_f = dT_dy / grad_mag
    sz_f = dT_dz / grad_mag
    ito_3d = np.pi - np.arccos(np.clip(sz_f, -1, 1))
    az_3d = (np.arctan2(sy_f, sx_f) + np.pi).astype(np.float32)
    az_3d = ((az_3d + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    t_takeoff = time.perf_counter() - t0

    # Ray distance
    t0 = time.perf_counter()
    r_3d = ray_distance_3d(TT, vp_3d, six, siy, siz, dr, near_cells=5)
    t_raydist = time.perf_counter() - t0

    # Pack into RayParamDB (transpose to z-first for RegularGridInterpolator)
    from dasfm.io.data import RayParamDB
    db = RayParamDB(
        traveltime=TT.transpose(2, 0, 1).astype(np.float32),
        takeoff=ito_3d.transpose(2, 0, 1).astype(np.float32),
        raypath_length=r_3d.transpose(2, 0, 1).astype(np.float32),
        azimuth=az_3d.transpose(2, 0, 1).astype(np.float32),
        grid_z=z_grid.astype(np.float32),
        grid_x=x_grid.astype(np.float32),
        grid_y=y_grid.astype(np.float32),
        geometry="cart_3d",
        receiver_x=float(x_grid[six]),
        receiver_y=float(y_grid[siy]),
        receiver_z=float(z_grid[siz]),
        dz=float(dr),
        dr=float(dr),
        origin_lat=float(geo["origin_lat"]),
        origin_lon=float(geo["origin_lon"]),
        x_offset=float(geo["x_offset"]),
        y_offset=float(geo["y_offset"]),
        h_max=float(geo["h_max"]),
    )

    timing_str = (f"init={t_init:.2f}s  eikonal={t_eikonal:.2f}s  "
                  f"takeoff={t_takeoff:.2f}s  ray_dist={t_raydist:.2f}s  "
                  f"total={t_init+t_eikonal+t_takeoff+t_raydist:.2f}s")

    return db, timing_str
