"""eval_ray_lookup.py — Batch evaluation of source parameters for all events at all DAS channels

Workflow
--------
1. For each receiver, load (or accept pre-built) :class:`RayParamDB` volume.
2. Trilinearly interpolate ``(traveltime, takeoff, raypath_length, azimuth)``
   at every source point → ``(n_ev, n_rx)`` arrays.
3. Optionally linearly interpolate along the receiver axis to fill missing
   DAS channels.
4. Wrap everything with step-level metadata in a :class:`RayParamTable`.

Public functions
----------------
* :func:`interp_lookup_channels` — per-channel interp + table assembly
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

from .ray_lookup_2d import interpolate_skipped
from dasfm.io.data import RayParamDB, RayParamTable


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Interpolate lookuped channels → channel-source table
# ──────────────────────────────────────────────────────────────────────────────

def _to_rpdb(item: Union[Path, str, RayParamDB]) -> RayParamDB:
    """Coerce a file path or a pre-loaded :class:`RayParamDB` into a
    :class:`RayParamDB`. Path inputs are loaded via
    :meth:`RayParamDB.from_hdf5`.
    """
    if isinstance(item, RayParamDB):
        return item
    return RayParamDB.from_hdf5(item)


def interp_lookup_channels(
    files: list[Union[Path, str, RayParamDB]],
    channel_ids: np.ndarray,
    pts: np.ndarray,
    *,
    forward_method: str,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    n_ch_total: int | None = None,
    verbose: bool = True,
    network: np.ndarray | None = None,
    station: np.ndarray | None = None,
    location: np.ndarray | None = None,
    rec_azi: np.ndarray | None = None,
    dasname: str | None = None,
    perturb_vert_uncert_km: float | None = None,
    perturb_horz_uncert_km: float | None = None,
) -> RayParamTable:
    """Interpolate per-receiver lookups into a complete :class:`RayParamTable`.

    Loads each entry of *files* (either a saved lookup path or a pre-built
    :class:`~dasfm.io.data.RayParamDB`), trilinearly interpolates the
    traveltime / takeoff / raypath-length / azimuth volumes at the source
    points in *pts*, optionally fills missing DAS channels, and packages
    the result with the step-level metadata into a ready-to-save
    :class:`~dasfm.io.data.RayParamTable`.

    The result is always a **nominal** table (no trial axis). Use
    :meth:`RayParamTable.stack_mc_trials` to combine the outputs of
    several calls into a single Monte-Carlo-shaped table.

    Parameters
    ----------
    files : list of Path, str, or RayParamDB
        Per-receiver lookup volumes. Mixed lists are allowed.
    channel_ids : (n_computed,) int64
        Integer receiver indices (one per entry of *files*). Used only
        when ``n_ch_total`` is set, to drive linear fill along the
        channel axis.
    pts : (n_ev, 3) float64
        Query points ``[source_z, source_x, source_y]`` in grid frame.
    forward_method : str
        Method label stored on the output table, e.g. ``"sta_1d"``,
        ``"das_2d"``, ``"sta_3d"``. Required.
    receiver_x, receiver_y : np.ndarray
        Full receiver coordinates, shape ``(n_out,)``. For STA workflows
        pass the coordinates of the receivers actually computed; for DAS
        workflows pass the coordinates of **all** ``n_ch_total`` channels.
        Both must match the final channel dimension.
    n_ch_total : int or None
        If provided, linearly interpolate along the receiver axis to
        produce ``n_out == n_ch_total`` (DAS workflow). If ``None``,
        ``n_out == n_computed`` (STA workflow).
    verbose : bool
        Show a tqdm progress bar during the per-receiver interpolation loop.
    network, station, location : np.ndarray, optional
        STA metadata, shape ``(n_out,)`` string arrays. Attached to the
        output table unchanged.
    rec_azi : np.ndarray, optional
        DAS receiver orientation (radians), shape ``(n_out,)``. Attached
        to the output table unchanged.
    dasname : str, optional
        DAS deployment identifier stored as a scalar attr on the output.
    perturb_vert_uncert_km, perturb_horz_uncert_km : float, optional
        Gaussian perturbation sigmas stored as scalar attrs (set when
        building an MC trial).

    Returns
    -------
    RayParamTable
        Nominal-shape table ``(n_ev, n_out)`` ready for serialization
        via :meth:`RayParamTable.to_hdf5` or assembly via
        :meth:`RayParamTable.stack_mc_trials`.

    Notes
    -----
    Azimuth handling:

    * ``cyl_2d`` lookups do not store an azimuth field; the table's
      ``azimuth`` is computed geometrically from source and receiver
      positions (great-circle back-azimuth in the grid frame).
    * ``cart_3d`` lookups carry their own ``azimuth`` field, which is
      trilinearly interpolated alongside the other three volumes.
    """
    n_computed = len(channel_ids)
    n_ev = pts.shape[0]
    n_out = n_ch_total if n_ch_total is not None else n_computed

    if receiver_x.shape != (n_out,) or receiver_y.shape != (n_out,):
        raise ValueError(
            f"receiver_x/y must have shape ({n_out},); "
            f"got receiver_x.shape={receiver_x.shape}, "
            f"receiver_y.shape={receiver_y.shape}"
        )

    T_comp   = np.full((n_ev, n_computed), np.nan, dtype=np.float32)
    ito_comp = np.full((n_ev, n_computed), np.nan, dtype=np.float32)
    r_comp   = np.full((n_ev, n_computed), np.nan, dtype=np.float32)
    az_comp  = None  # allocated on first encounter of a cart_3d lookup

    loop_iter = list(zip(channel_ids, files))
    if verbose:
        loop_iter = tqdm(loop_iter, desc="Interpolating computed channels")

    for j, (_ch_id, item) in enumerate(loop_iter):
        db = _to_rpdb(item)
        tt_j, ito_j, r_j, az_j = db.query(pts)
        T_comp[:, j]   = tt_j
        ito_comp[:, j] = ito_j
        r_comp[:, j]   = r_j
        if az_j is not None:
            if az_comp is None:
                az_comp = np.full((n_ev, n_computed), np.nan, dtype=np.float32)
            az_comp[:, j] = az_j

    if n_ch_total is not None:
        T_comp   = interpolate_skipped(T_comp,   channel_ids, n_ch_total, axis=1)
        ito_comp = interpolate_skipped(ito_comp, channel_ids, n_ch_total, axis=1)
        r_comp   = interpolate_skipped(r_comp,   channel_ids, n_ch_total, axis=1)
        if az_comp is not None:
            az_comp = interpolate_skipped(az_comp, channel_ids, n_ch_total, axis=1)

    # Source coordinates — pulled from pts
    source_z = np.asarray(pts[:, 0], dtype=np.float64)
    source_x = np.asarray(pts[:, 1], dtype=np.float64)
    source_y = np.asarray(pts[:, 2], dtype=np.float64)

    # Azimuth: for cyl_2d (az_comp is None), compute geometrically using full
    # receiver coordinates. For cart_3d, use the interpolated field.
    if az_comp is None:
        az_out = np.arctan2(
            receiver_y[None, :] - source_y[:, None],
            receiver_x[None, :] - source_x[:, None],
        ).astype(np.float32)
    else:
        az_out = az_comp

    return RayParamTable(
        traveltime=T_comp,
        takeoff=ito_comp,
        azimuth=az_out,
        raypath_length=r_comp,
        source_x=source_x,
        source_y=source_y,
        source_z=source_z,
        receiver_x=np.asarray(receiver_x, dtype=np.float64),
        receiver_y=np.asarray(receiver_y, dtype=np.float64),
        forward_method=forward_method,
        network=network,
        station=station,
        location=location,
        rec_azi=rec_azi,
        dasname=dasname,
        perturb_vert_uncert_km=perturb_vert_uncert_km,
        perturb_horz_uncert_km=perturb_horz_uncert_km,
    )
