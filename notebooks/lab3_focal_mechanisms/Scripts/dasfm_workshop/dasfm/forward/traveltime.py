"""2D first-arrival traveltime forward modeling via the Fast Sweeping Method.

Solves the eikonal equation on a 2D Cartesian grid::

    |∇T(z, x)|² = s(z, x)²    where  s = 1 / v

using the Fast Sweeping Method (FSM) with an upwind finite-difference stencil.
The grid axes are:

* ``z`` — depth (rows 0 … nz-1)
* ``x`` — horizontal / along-fiber (columns 0 … nx-1)

Multiple simultaneous sources are supported via arrays of source positions and
initial traveltimes.

If ``numba`` is installed, the inner sweep loops are JIT-compiled and run at
near-C speed.  If ``numba`` is absent the identical pure-Python loops are used
(correct but slow for large grids; install numba for production use).

Algorithm
---------
Four alternating diagonal sweeps per iteration::

    (z↑, x↑) → (z↑, x↓) → (z↓, x↑) → (z↓, x↓)

Each node is updated with the upwind Godunov stencil:

1. Find the minimum neighbour traveltime in z (``Tz``) and x (``Tx``).
2. Sort so that ``Tz ≤ Tx``.
3. Candidate from the smaller direction only: ``T1D = Tz + dz·s``.
4. If ``T1D ≤ Tx`` use it; otherwise solve the 2D quadratic::

       (T−Tz)²/dz² + (T−Tx)²/dx² = s²

Iterations stop when ``max(|ΔT|) < 1e-9`` or ``max_iter`` is reached.

Reference MATLAB implementation
---------------------------------
``old_code/YTF_FSM_new.m``  +  ``old_code/eikslv2d.m``

Public functions
----------------
* :func:`fsm_2d`
"""

from __future__ import annotations

import math

import numpy as np

from numba import njit as _njit

TRAVELTIME_INF = 9999.0   # sentinel traveltime for unreachable / uninitialised nodes


# ===========================================================================
# Inner kernel — upwind eikonal solver at a single node
# ===========================================================================

@_njit(cache=True)
def eikonal_solve_node(TT, v, iz, ix, dz, dx):
    """Upwind eikonal update at grid node (iz, ix).

    Parameters
    ----------
    TT : (nz, nx) float64
        Current traveltime field.
    v  : (nz, nx) float64
        Velocity model [distance / time].
    iz, ix : int
        0-based row and column indices of the node to update.
    dz, dx : float
        Grid spacings in z and x directions.

    Returns
    -------
    float
        Updated traveltime candidate.  Returns ``TRAVELTIME_INF`` if no valid update.
    """
    nz = TT.shape[0]
    nx = TT.shape[1]
    s  = 1.0 / v[iz, ix]      # slowness

    # --- minimum neighbour in z ------------------------------------------------
    if iz == 0:
        Tz = TT[iz + 1, ix]
    elif iz == nz - 1:
        Tz = TT[iz - 1, ix]
    else:
        Tz = min(TT[iz - 1, ix], TT[iz + 1, ix])
    hz = dz

    # --- minimum neighbour in x ------------------------------------------------
    if ix == 0:
        Tx = TT[iz, ix + 1]
    elif ix == nx - 1:
        Tx = TT[iz, ix - 1]
    else:
        Tx = min(TT[iz, ix - 1], TT[iz, ix + 1])
    hx = dx

    # --- sort so that Tz ≤ Tx (Godunov ordering) ------------------------------
    if Tz > Tx:
        Tz, Tx = Tx, Tz
        hz, hx = hx, hz

    # --- 1D update from the smaller-traveltime direction ----------------------
    t1d = Tz + hz * s
    if t1d <= Tx:
        return t1d

    # --- 2D quadratic update: (T-Tz)²/hz² + (T-Tx)²/hx² = s² ----------------
    ta  =  1.0 / hz**2 + 1.0 / hx**2
    tb  = -2.0 * (Tz / hz**2 + Tx / hx**2)
    tc  =  Tz**2 / hz**2 + Tx**2 / hx**2 - s**2
    dis = tb**2 - 4.0 * ta * tc

    if dis >= 0.0:
        return (-tb + math.sqrt(dis)) / (2.0 * ta)
    return TRAVELTIME_INF


# ===========================================================================
# Fast Sweeping Method — four alternating sweep directions
# ===========================================================================


@_njit(cache=True)
def fsm_one_group(TT, v, src_z, src_x, dz, dx):
    """Execute one group of 4 sweeps and return max |ΔT|."""
    nz = TT.shape[0]
    nx = TT.shape[1]
    ns = src_z.shape[0]

    # snapshot before sweeps
    T_prev = TT.copy()

    # --- sweep 1: z↑, x↑ ------------------------------------------------------
    for iz in range(nz):
        for ix in range(nx):
            is_src = False
            for ir in range(ns):
                if iz == src_z[ir] and ix == src_x[ir]:
                    is_src = True
                    break
            if is_src:
                continue
            t = eikonal_solve_node(TT, v, iz, ix, dz, dx)
            if t < TT[iz, ix]:
                TT[iz, ix] = t

    # --- sweep 2: z↑, x↓ ------------------------------------------------------
    for iz in range(nz):
        for ix in range(nx - 1, -1, -1):
            is_src = False
            for ir in range(ns):
                if iz == src_z[ir] and ix == src_x[ir]:
                    is_src = True
                    break
            if is_src:
                continue
            t = eikonal_solve_node(TT, v, iz, ix, dz, dx)
            if t < TT[iz, ix]:
                TT[iz, ix] = t

    # --- sweep 3: z↓, x↑ ------------------------------------------------------
    for iz in range(nz - 1, -1, -1):
        for ix in range(nx):
            is_src = False
            for ir in range(ns):
                if iz == src_z[ir] and ix == src_x[ir]:
                    is_src = True
                    break
            if is_src:
                continue
            t = eikonal_solve_node(TT, v, iz, ix, dz, dx)
            if t < TT[iz, ix]:
                TT[iz, ix] = t

    # --- sweep 4: z↓, x↓ ------------------------------------------------------
    for iz in range(nz - 1, -1, -1):
        for ix in range(nx - 1, -1, -1):
            is_src = False
            for ir in range(ns):
                if iz == src_z[ir] and ix == src_x[ir]:
                    is_src = True
                    break
            if is_src:
                continue
            t = eikonal_solve_node(TT, v, iz, ix, dz, dx)
            if t < TT[iz, ix]:
                TT[iz, ix] = t

    # --- max change across one group of 4 sweeps ------------------------------
    max_change = 0.0
    for iz in range(nz):
        for ix in range(nx):
            c = abs(TT[iz, ix] - T_prev[iz, ix])
            if c > max_change:
                max_change = c
    return max_change


# ===========================================================================
# Public API
# ===========================================================================

def fsm_2d(
    velocity: np.ndarray,
    source_z: int | np.ndarray,
    source_x: int | np.ndarray,
    dz: float,
    dx: float,
    t0: float | np.ndarray = 0.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Compute a 2D first-arrival traveltime field via the Fast Sweeping Method.

    Solves the eikonal equation ``|∇T|² = 1/v²`` on a 2D Cartesian grid.
    Grid axes: rows = depth (z), columns = horizontal (x).

    Parameters
    ----------
    velocity : np.ndarray, shape (nz, nx)
        2D velocity model.  Units must be consistent with *dz* / *dx*
        (e.g. km/s with km grid spacings, or m/s with m spacings).
        Rows are the depth axis, columns the horizontal axis.
    source_z : int or array-like of int
        Source depth grid index / indices (0-based row index into *velocity*).
    source_x : int or array-like of int
        Source horizontal grid index / indices (0-based column index).
    dz : float
        Grid spacing in the depth direction [same length unit as velocity×time].
    dx : float
        Grid spacing in the horizontal direction.
    t0 : float or array-like of float
        Initial traveltime(s) at the source node(s).  Default ``0.0``.
        Use non-zero values for delayed or offset sources.
    max_iter : int
        Maximum number of sweep-group iterations (each iteration = 4 sweeps).
        Default 50.  Convergence is checked after every group; the loop exits
        early when ``max(|ΔT|) < 1e-9``.

    Returns
    -------
    np.ndarray, shape (nz, nx), dtype float64
        First-arrival traveltime field in the same time units as
        ``dz / velocity``.  Nodes that cannot be reached retain the sentinel
        value ``9999``.

    Notes
    -----
    * If ``numba`` is installed the inner loops are JIT-compiled; otherwise
      identical pure-Python loops are used (slower for large grids).
    * Source nodes are never overwritten by the sweep update, so their
      traveltime stays fixed at *t0*.

    Examples
    --------
    Homogeneous halfspace, single source at grid centre::

        import numpy as np
        from dasfm.forward import fsm_2d

        nz, nx = 101, 201
        v = np.full((nz, nx), 3.0)           # 3 km/s everywhere
        TT = fsm_2d(v, source_z=50, source_x=100, dz=0.05, dx=0.05)
        # TT[0, 100] ≈ 2.5 s  (50 nodes × 0.05 km / 3 km/s)
    """
    velocity = np.ascontiguousarray(velocity, dtype=np.float64)
    nz, nx   = velocity.shape

    src_z = np.atleast_1d(np.asarray(source_z, dtype=np.int64)).ravel()
    src_x = np.atleast_1d(np.asarray(source_x, dtype=np.int64)).ravel()
    if src_z.shape != src_x.shape:
        raise ValueError("source_z and source_x must have the same length")

    t0_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(t0, dtype=np.float64)), src_z.shape
    ).copy()

    if np.any((src_z < 0) | (src_z >= nz)):
        raise ValueError(f"source_z has indices outside [0, {nz - 1}]")
    if np.any((src_x < 0) | (src_x >= nx)):
        raise ValueError(f"source_x has indices outside [0, {nx - 1}]")

    # --- initialise traveltime field ------------------------------------------
    TT = np.full((nz, nx), TRAVELTIME_INF, dtype=np.float64)
    for iz, ix, t in zip(src_z.tolist(), src_x.tolist(), t0_arr.tolist()):
        TT[iz, ix] = t

    # --- iterate sweep groups until convergence --------------------------------
    for _it in range(max_iter):
        max_change = fsm_one_group(TT, velocity, src_z, src_x,
                                    float(dz), float(dx))
        if max_change < 1e-9:
            break

    return TT
