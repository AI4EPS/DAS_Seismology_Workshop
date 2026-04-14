"""ray_lookup_2d — 2D FSM ray-parameter lookup table construction.

Builds per-receiver cylindrical lookup tables via azimuthal cross-sections
using the Fast Sweeping Method (FSM).

Public functions:
* build_velocity_2d
* compute_lookup_table
* compute_ray_lookup — single-receiver lookup table
* interpolate_skipped
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from numba import njit as _njit

from .traveltime import fsm_2d


# ===========================================================================
# 2-D velocity model builder
# ===========================================================================

def build_velocity_2d(
    depth_1d: np.ndarray,
    vp_1d: np.ndarray,
    z_grid: np.ndarray,
    r_max: float,
    dr: float,
    surface_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate a 1-D velocity profile into a 2-D (depth × radial) model.

    ``z_grid`` is referenced to the **model top** (z = 0).  For each radial
    column *r*, the ground surface sits at ``z = surface_depth[r]`` (≥ 0).
    The 1-D profile is applied starting from that local surface depth::

        depth_below_surface(iz, ir) = z_grid[iz] − surface_depth[ir]

    * If < 0 : node is above the local surface → assigned ``vp_1d[0]``
    * If ≥ 0 : velocity = interp(depth_below_surface, depth_1d, vp_1d)

    When *surface_depth* is ``None`` the surface is at z = 0 everywhere
    (no topographic correction; standard flat-Earth assumption).

    Parameters
    ----------
    depth_1d : np.ndarray, shape (n,)
        Depth nodes of the 1-D model [km], starting at 0 (ground surface).
    vp_1d : np.ndarray, shape (n,)
        P-wave velocity at each depth node [km/s].
    z_grid : np.ndarray, shape (nz,)
        Model depth grid [km], z = 0 is the model top.
    r_max : float
        Maximum radial distance [km].  Used to compute *nr* when
        *surface_depth* is ``None``.
    dr : float
        Radial grid spacing [km].
    surface_depth : np.ndarray, shape (n_cols,), optional
        Depth of the ground surface below the model top at each column
        [km, ≥ 0].  When provided, its length determines the number of
        columns in the returned *vp_2d* (overriding the *r_max / dr* count).
        Compute as ``h_max − h(r)`` where ``h_max`` is the maximum terrain
        elevation and ``h(r)`` is the local elevation.

    Returns
    -------
    vp_2d : np.ndarray, shape (nz, n_cols), float64
        2-D velocity model [km/s].
    r_grid : np.ndarray, shape (nr,), float64
        Radial grid [km] from 0 to *r_max*.  Always derived from *r_max* and
        *dr*; use the length of *surface_depth* when building a full
        cross-section model.
    """
    nr = int(np.ceil(r_max / dr)) + 1
    r_grid = np.arange(nr, dtype=np.float64) * dr
    nz = len(z_grid)

    if surface_depth is None:
        # No topographic correction — surface coincides with model top (z = 0)
        vp_col = np.interp(z_grid, depth_1d, vp_1d)       # (nz,)
        vp_2d  = np.tile(vp_col[:, np.newaxis], (1, nr))   # (nz, nr)
    else:
        # depth_below_surface[iz, ic] = z_grid[iz] − surface_depth[ic]
        sd = np.asarray(surface_depth, dtype=np.float64)       # (n_cols,)
        z_below_2d = z_grid[:, np.newaxis] - sd[np.newaxis, :]  # (nz, n_cols)
        # clamp: nodes above local surface → depth 0 (surface velocity)
        z_below_2d = np.maximum(z_below_2d, 0.0)
        n_cols = sd.size
        vp_2d = np.interp(
            z_below_2d.ravel(), depth_1d, vp_1d
        ).reshape(nz, n_cols)

    return vp_2d.astype(np.float64), r_grid


# ===========================================================================
# Takeoff angle from traveltime gradient
# ===========================================================================

def takeoff_from_gradient(
    TT: np.ndarray,
    dz: float,
    dx: float,
) -> np.ndarray:
    """Compute takeoff angle [rad] at every grid node from the traveltime gradient.

    Standard seismological convention: **0° = straight down, 90° = horizontal,
    180° = straight up**.  For a surface DAS receiver and subsurface earthquake,
    the upgoing ray has ito ∈ (90°, 180°].

    Derivation: with a virtual source at the DAS receiver, *TT* increases away
    from the receiver, so the actual earthquake ray travels in the ``-∇T``
    direction.  The angle this direction makes with the downward vertical (+z)::

        cos(ito) = -dT/dz / sqrt((dT/dz)^2 + (dT/dx)^2)

    Parameters
    ----------
    TT : np.ndarray, shape (nz, n_cols)
        First-arrival traveltime with virtual source at ``(src_iz, src_ix)``.
    dz : float
        Vertical grid spacing [km].
    dx : float
        Horizontal grid spacing [km].

    Returns
    -------
    ito : np.ndarray, shape (nz, n_cols), float32
        Takeoff angle [rad].  0 = straight down, π/2 = horizontal, π = straight up.
    """
    dT_dz = np.gradient(TT, dz, axis=0)
    dT_dx = np.gradient(TT, dx, axis=1)

    grad_mag = np.sqrt(dT_dz**2 + dT_dx**2)

    # Negative sign: ray direction is -∇T; angle measured from downward vertical.
    # Default to π (straight up) at ill-defined nodes (source node, grad≈0).
    # Use np.divide with where= to avoid division at near-zero gradient nodes.
    cos_ito = np.full_like(grad_mag, -1.0)
    _mask = grad_mag > 1e-12
    np.divide(-dT_dz, grad_mag, out=cos_ito, where=_mask)
    cos_ito = np.clip(cos_ito, -1.0, 1.0)

    return np.arccos(cos_ito).astype(np.float32)


# ===========================================================================
# Single-receiver lookup table (one cross-section)
# ===========================================================================

def compute_lookup_table(
    vp_2d: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
    src_iz: int,
    src_ix: int = 0,
    max_iter: int = 50,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the traveltime and takeoff-angle fields for one FSM run.

    A virtual source is placed at ``(src_iz, src_ix)`` with initial traveltime
    *t0* and the FSM fills the full ``(nz, n_cols)`` grid.  The horizontal
    spacing is derived from ``|r_grid[1] - r_grid[0]|``, so *r_grid* may be
    either a half-plane array ``[0, dr, …, r_max]`` or a full cross-section
    array ``[−r_max, …, 0, …, r_max]`` — both have uniform spacing *dr*.

    Parameters
    ----------
    vp_2d : np.ndarray, shape (nz, n_cols)
        2-D velocity model [km/s].
    z_grid : np.ndarray, shape (nz,)
        Depth grid [km].
    r_grid : np.ndarray, shape (n_cols,)
        Grid along the horizontal axis [km].  Only the spacing
        ``|r_grid[1] - r_grid[0]|`` is used internally.
    src_iz : int
        0-based depth index of the virtual source (receiver depth node).
    src_ix : int
        0-based horizontal index of the virtual source.  Default 0 (left
        edge, i.e. standard half-plane FSM).  For a full cross-section
        use ``src_ix = nr - 1`` (centre column).
    max_iter : int
        Maximum FSM sweep-group iterations.  Default 50.
    t0 : float
        Initial traveltime [s] at the virtual source node.  Default 0.0.

    Returns
    -------
    TT : np.ndarray, shape (nz, n_cols), float64
        First-arrival traveltime [s].
    ito : np.ndarray, shape (nz, n_cols), float32
        Takeoff angle [rad].
    """
    dz = float(z_grid[1] - z_grid[0]) if len(z_grid) > 1 else 1.0
    dx = float(abs(r_grid[1] - r_grid[0])) if len(r_grid) > 1 else 1.0

    TT = fsm_2d(
        vp_2d,
        source_z=src_iz,
        source_x=src_ix,
        dz=dz,
        dx=dx,
        t0=float(t0),
        max_iter=max_iter,
    )
    ito = takeoff_from_gradient(TT, dz, dx)
    return TT, ito


# ===========================================================================
# Ray-path distance field
# ===========================================================================

@_njit(cache=True)
def ray_distance_core(order, r_flat, iz_up_flat, ix_up_flat,
                       delta_z_flat, delta_x_flat, nc,
                       src_iz, src_ix, near_r2):
    """Inner loop for ray distance, accelerated by numba.

    Nodes within near_r2 (squared grid-cell distance) of the source are
    pre-initialized with Euclidean distance and skipped here, because
    the traveltime gradient is unreliable near the Eikonal source.
    """
    for i in range(order.shape[0]):
        flat_k = order[i]
        iz = flat_k // nc
        ix = flat_k % nc

        # Skip near-source nodes (already set to Euclidean distance)
        diz = iz - src_iz
        dix = ix - src_ix
        if diz * diz + dix * dix <= near_r2:
            continue

        r_z = r_flat[iz_up_flat[flat_k] * nc + ix] + delta_z_flat[flat_k]
        r_x = r_flat[iz * nc + ix_up_flat[flat_k]] + delta_x_flat[flat_k]
        r_new = r_z if r_z < r_x else r_x
        if r_new < r_flat[flat_k]:
            r_flat[flat_k] = r_new


def ray_distance_2d(
    TT: np.ndarray,
    vp_2d: np.ndarray,
    src_iz: int,
    src_ix: int,
    dz: float,
    dx: float,
    near_cells: int = 5,
) -> np.ndarray:
    """Compute the ray-path distance field [km] on a 2-D FSM grid.

    Uses the relation **∇r = v · ∇T**, which holds exactly along seismic
    rays.  Starting from r = 0 at the virtual source (receiver position),
    nodes are visited in ascending-traveltime order so that each node's
    upwind (smaller-T) neighbour is already updated.

    Near the Eikonal source (within *near_cells* grid cells), the gradient
    is unreliable, so Euclidean straight-line distance is used instead.

    The inner loop is JIT-compiled by numba when available.

    Parameters
    ----------
    TT : np.ndarray, shape (nz, nc)
        First-arrival traveltime [s] from the FSM virtual source.
    vp_2d : np.ndarray, shape (nz, nc)
        P-wave velocity [km/s].
    src_iz : int
        Depth index of the virtual source (receiver).
    src_ix : int
        Horizontal index of the virtual source (receiver).
    dz : float
        Vertical grid spacing [km].
    dx : float
        Horizontal grid spacing [km].
    near_cells : int
        Grid cells within this radius of the source use Euclidean distance.

    Returns
    -------
    r : np.ndarray, shape (nz, nc), float32
        Ray-path distance [km]; unreachable nodes retain ``inf``.
    """
    nz, nc = TT.shape
    dT_dz = np.gradient(TT, dz, axis=0)
    dT_dx = np.gradient(TT, dx, axis=1)

    # Build flattened upwind-index arrays
    iz_arr = np.arange(nz, dtype=np.int64)[:, None] * np.ones(nc, dtype=np.int64)
    ix_arr = np.ones(nz, dtype=np.int64)[:, None] * np.arange(nc, dtype=np.int64)

    iz_up_flat = np.where(
        dT_dz.ravel() >= 0,
        np.clip(iz_arr.ravel() - 1, 0, nz - 1),
        np.clip(iz_arr.ravel() + 1, 0, nz - 1),
    ).astype(np.int64)
    ix_up_flat = np.where(
        dT_dx.ravel() >= 0,
        np.clip(ix_arr.ravel() - 1, 0, nc - 1),
        np.clip(ix_arr.ravel() + 1, 0, nc - 1),
    ).astype(np.int64)

    delta_z_flat = (vp_2d * np.abs(dT_dz) * dz).ravel()
    delta_x_flat = (vp_2d * np.abs(dT_dx) * dx).ravel()

    # Initialize: near-source nodes → Euclidean distance; rest → inf
    r_flat = np.full(nz * nc, np.inf, dtype=np.float64)
    nc_r = near_cells
    for diz in range(-nc_r, nc_r + 1):
        for dix in range(-nc_r, nc_r + 1):
            if diz * diz + dix * dix > nc_r * nc_r:
                continue
            jz = src_iz + diz
            jx = src_ix + dix
            if 0 <= jz < nz and 0 <= jx < nc:
                r_flat[jz * nc + jx] = np.sqrt(
                    (diz * dz) ** 2 + (dix * dx) ** 2
                )

    order = np.argsort(TT.ravel()).astype(np.int64)
    ray_distance_core(order, r_flat, iz_up_flat, ix_up_flat,
                       delta_z_flat, delta_x_flat, np.int64(nc),
                       np.int64(src_iz), np.int64(src_ix),
                       np.int64(nc_r * nc_r))

    return r_flat.reshape(nz, nc).astype(np.float32)


# ===========================================================================
# Subsampled-receiver interpolation
# ===========================================================================

def interpolate_skipped(
    arr: np.ndarray,
    computed_idx: np.ndarray,
    n_ch: int,
    axis: int = 0,
) -> np.ndarray:
    """Linearly interpolate from computed channels to all channels.

    Channels outside the range of *computed_idx* receive the nearest edge
    value (clamp).

    Parameters
    ----------
    arr : np.ndarray
        Array with *n_computed* entries along *axis*.
    computed_idx : np.ndarray, shape (n_computed,), int
        Sorted 0-based receiver indices that were actually computed.
    n_ch : int
        Total number of receivers.
    axis : int
        Axis along which to interpolate (default 0).

    Returns
    -------
    out : np.ndarray, same dtype as *arr*
        Array with *n_ch* entries along *axis*.
    """
    # Move interpolation axis to front
    arr_moved = np.moveaxis(arr, axis, 0)
    n_c = arr_moved.shape[0]
    trail_shape = arr_moved.shape[1:]

    flat_c = arr_moved.reshape(n_c, -1).astype(np.float64)
    x_c    = computed_idx.astype(np.float64)
    x_all  = np.arange(n_ch, dtype=np.float64)

    f = interp1d(
        x_c,
        flat_c,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(flat_c[0], flat_c[-1]),
    )
    out_flat = f(x_all).reshape((n_ch,) + trail_shape).astype(arr.dtype)

    return np.moveaxis(out_flat, 0, axis)


# ===========================================================================
# Main precomputation routine
# ===========================================================================

def compute_ray_lookup(
    geo: dict,
    receiver_ix: int,
    receiver_iy: int,
    receiver_iz: int,
    vel_1d_depth: np.ndarray,
    vel_1d_vp: np.ndarray,
    dr: float = 0.1,
    daz: float = 5.0,
    r_max: float | None = None,
    max_iter: int = 300,
    verbose: bool = True,
    receiver_x_exact: float | None = None,
    receiver_y_exact: float | None = None,
) -> dict:
    """Compute 3-D traveltime and takeoff-angle lookup tables for ONE receiver.

    Uses seismic reciprocity: a virtual source is placed at the DAS channel
    and the FSM sweeps radial 2-D cross-sections in *n_az* azimuthal directions.
    The per-azimuth results are scatter-interpolated onto the regular 3-D output
    grid ``(z, x, y)`` from *geo*.

    This function handles **a single receiver**.  Looping over all channels,
    stacking results, and assembling the full ``(n_ch, nz, nx, ny)`` lookup is
    the caller's responsibility.

    The receiver position is specified as grid indices so that no coordinate
    conversion or DEM query is needed here; pass ``geo["receiver_ix"][i_ch]``
    etc. directly from the output of
    :func:`~dasfm.forward.geometry.build_model_grid`.

    Coordinate convention (x = northing, y = easting, z = depth positive down):

    * ``z = 0``  — highest terrain elevation within the domain (``h_max``)
    * ``receiver_z = z_grid[receiver_iz]``
    * ``source_z   = h_max + depth_km``

    Parameters
    ----------
    geo : dict
        Output of :func:`~dasfm.forward.geometry.build_model_grid`.  Required
        keys: ``x_grid``, ``y_grid``, ``z_grid``, ``h_max``, ``dz``,
        ``surface_iz``.
    receiver_ix : int
        0-based index into ``geo["x_grid"]`` for the receiver's northing.
    receiver_iy : int
        0-based index into ``geo["y_grid"]`` for the receiver's easting.
    receiver_iz : int
        0-based index into ``geo["z_grid"]`` for the receiver's depth.
        Pass ``geo["receiver_iz"][i_ch]`` (computed from DEM by
        :func:`~dasfm.forward.geometry.build_model_grid`).
    vel_1d_depth : np.ndarray, shape (n,)
        Depth nodes of the 1-D velocity model [km], starting at 0 (ground
        surface, depth below local terrain).
    vel_1d_vp : np.ndarray, shape (n,)
        P-wave velocity at each depth node [km/s].
    dr : float
        Radial FSM grid spacing [km].  Default 0.1.
    daz : float
        Azimuthal step [degrees].  ``180 / daz`` two-sided cross-sections are
        computed.  Default 5.0 (36 planes → 360° coverage).
    r_max : float, optional
        Maximum radial distance for the FSM cross-sections [km].
        Default: distance from the receiver to the farthest corner of the
        output grid + 2 km (guarantees full grid coverage).
    max_iter : int
        Maximum FSM iterations per cross-section.  Default 300.
    verbose : bool
        Print a one-line summary.  Default True.

    Returns
    -------
    dict with keys:

    **Lookup tables** (cylindrical grid)
        ``T_cyl``   (nz, nr, n_az_full+1) float32 — first-arrival traveltime [s]
        ``ito_cyl`` (nz, nr, n_az_full+1) float32 — takeoff angle [rad]
        ``r_cyl``   (nz, nr, n_az_full+1) float32 — ray distance [km]
        (0 = straight down, π/2 = horizontal, π = straight up)

    **Grid axes**
        ``z_grid``  (nz,) float32 — depth axis [km]
        ``r_grid``  (nr,) float32 — radial distance from receiver [km]
        ``az_grid`` (n_az_full+1,) float32 — azimuth [rad], 0 to 2π (periodic)

    **Receiver position** (grid-frame)
        ``receiver_x``, ``receiver_y``, ``receiver_z`` — float [km]
        ``receiver_ix``, ``receiver_iy``, ``receiver_iz`` — int (echoed back)

    **Spacings**
        ``dz``, ``dr`` — float [km]
    """
    # ── Grid axes from geo ────────────────────────────────────────────────────
    z_grid = np.asarray(geo["z_grid"], dtype=np.float64)
    x_geo  = np.asarray(geo["x_grid"], dtype=np.float64)
    y_geo  = np.asarray(geo["y_grid"], dtype=np.float64)
    dz     = float(geo["dz"])
    nz     = len(z_grid)
    nx_geo = len(x_geo)
    ny_geo = len(y_geo)
    h_max  = float(geo["h_max"])

    # ── Receiver position ──────────────────────────────────────────────────────
    receiver_ix = int(receiver_ix)
    receiver_iy = int(receiver_iy)
    receiver_iz = int(receiver_iz)
    # Use exact continuous coordinates if provided, otherwise snap to grid
    rx = float(receiver_x_exact) if receiver_x_exact is not None else float(x_geo[receiver_ix])
    ry = float(receiver_y_exact) if receiver_y_exact is not None else float(y_geo[receiver_iy])
    rz = float(z_grid[receiver_iz])

    # ── Radial extent: cover the farthest grid corner from this receiver ──────
    if r_max is None:
        corners_x = np.array([x_geo[0],  x_geo[0],  x_geo[-1], x_geo[-1]])
        corners_y = np.array([y_geo[0],  y_geo[-1], y_geo[0],  y_geo[-1]])
        r_max = float(np.hypot(corners_x - rx, corners_y - ry).max()) + 2.0

    # ── Full cross-section x-axis: [-r_max, …, 0, …, r_max] ─────────────────
    nr        = int(np.ceil(r_max / dr)) + 1
    r_grid_half = np.arange(nr, dtype=np.float64) * dr          # [0, dr, …, r_max]
    x_full    = np.concatenate([-r_grid_half[1:][::-1], r_grid_half])  # (2*nr-1,)
    n_full    = len(x_full)
    ix_center = nr - 1     # index of x = 0 (receiver position)

    # ── Azimuthal sampling ────────────────────────────────────────────────────
    az_deg    = np.arange(0.0, 180.0, float(daz))
    n_az      = len(az_deg)
    n_az_full = 2 * n_az   # full 360° coverage

    # ── Surface depth interpolator from geo["surface_iz"] ────────────────────
    surface_iz_2d  = np.asarray(geo["surface_iz"], dtype=np.int64)   # (nx, ny)
    surface_depth_2d = z_grid[surface_iz_2d]                          # (nx, ny) [km]
    _surf_rgi = RegularGridInterpolator(
        (x_geo, y_geo), surface_depth_2d,
        method="linear", bounds_error=False, fill_value=None,
    )

    def _surface_depth_at(cx_arr: np.ndarray, cy_arr: np.ndarray) -> np.ndarray:
        """Return surface depth [km] for profile points, clamped to grid."""
        cx_c = np.clip(cx_arr, x_geo[0], x_geo[-1])
        cy_c = np.clip(cy_arr, y_geo[0], y_geo[-1])
        return _surf_rgi(np.column_stack([cx_c, cy_c]))

    # ── Azimuthal cross-sections → cylindrical grid ─────────────────────────
    src_iz  = int(np.clip(receiver_iz, 0, nz - 1))
    dx_fsm  = float(dr)   # horizontal FSM grid spacing = dr

    T_cyl   = np.empty((nz, nr, n_az_full), dtype=np.float32)
    ito_cyl = np.empty((nz, nr, n_az_full), dtype=np.float32)
    r_cyl   = np.empty((nz, nr, n_az_full), dtype=np.float32)

    for k, az in enumerate(az_deg):
        az_rad = np.radians(az)
        cx = rx + x_full * np.cos(az_rad)
        cy = ry + x_full * np.sin(az_rad)

        surface_depth = _surface_depth_at(cx, cy)
        z_below_2d = np.maximum(z_grid[:, None] - surface_depth[None, :], 0.0)
        vp_2d = np.interp(
            z_below_2d.ravel(), vel_1d_depth, vel_1d_vp
        ).reshape(nz, n_full)

        TT, ito = compute_lookup_table(
            vp_2d, z_grid, x_full,
            src_iz=src_iz, src_ix=ix_center,
            max_iter=max_iter, t0=0.0,
        )

        r_2d = ray_distance_2d(TT, vp_2d, src_iz, ix_center, dz, dx_fsm)

        # Positive half (azimuth = az): x >= 0, columns ix_center..end
        T_cyl[:, :, k]   = TT[:, ix_center:].astype(np.float32)
        ito_cyl[:, :, k] = ito[:, ix_center:]
        r_cyl[:, :, k]   = r_2d[:, ix_center:].astype(np.float32)

        # Negative half reversed (azimuth = az + 180°): columns ix_center..0
        T_cyl[:, :, k + n_az]   = TT[:, ix_center::-1].astype(np.float32)
        ito_cyl[:, :, k + n_az] = ito[:, ix_center::-1]
        r_cyl[:, :, k + n_az]   = r_2d[:, ix_center::-1].astype(np.float32)

    # ── Periodic padding: append az=2π (copy of az=0) for interpolation ──────
    az_grid = np.deg2rad(np.arange(0.0, 360.0, float(daz)))    # (n_az_full,)
    az_grid = np.append(az_grid, 2.0 * np.pi)                   # (n_az_full+1,)
    T_cyl   = np.concatenate([T_cyl,   T_cyl[:, :, 0:1]], axis=2)
    ito_cyl = np.concatenate([ito_cyl, ito_cyl[:, :, 0:1]], axis=2)
    r_cyl   = np.concatenate([r_cyl,   r_cyl[:, :, 0:1]], axis=2)

    from dasfm.io.data import RayParamDB
    return RayParamDB(
        traveltime=T_cyl,              # (nz, nr, n_az_full+1) float32 s
        takeoff=ito_cyl,                # (nz, nr, n_az_full+1) float32 rad
        raypath_length=r_cyl,           # (nz, nr, n_az_full+1) float32 km
        grid_z=z_grid.astype(np.float32),
        grid_r=r_grid_half.astype(np.float32),
        grid_az=az_grid.astype(np.float32),
        geometry="cyl_2d",
        receiver_x=float(rx),
        receiver_y=float(ry),
        receiver_z=float(rz),
        receiver_ix=int(receiver_ix),
        receiver_iy=int(receiver_iy),
        receiver_iz=int(receiver_iz),
        dz=dz,
        dr=float(dr),
        origin_lat=float(geo["origin_lat"]),
        origin_lon=float(geo["origin_lon"]),
        x_offset=float(geo["x_offset"]),
        y_offset=float(geo["y_offset"]),
        h_max=float(geo["h_max"]),
    )
