"""channel_selection.py — DAS channel subsampling utilities.

Provides :func:`subsample_das_channels` which thins a dense DAS channel
array while preserving all geometrically significant *bend points* (locations
where the fiber cable changes direction).

Algorithm
---------
1. **Smooth** — apply a boxcar moving-average of width *smooth_window* to
   *receiver_x* and *receiver_y* before bend detection.  This suppresses
   GPS noise so that only genuine cable bends are flagged.
2. **Uniform subsample** — keep every *dc*-th index: ``0, dc, 2·dc, …``.
3. **Bend detection** — walk along the smoothed path accumulating the
   absolute per-segment azimuth difference (wrapped to ±180°) since the
   last selected anchor.  When the running sum reaches *bend_threshold_deg*,
   the current channel is marked as a new bend anchor and the accumulator
   resets to zero.  Catches both sharp local kinks and slow continuous
   bends that any single local-angle test would miss.
4. **Union** — return the sorted union of both sets.

Inputs use the Cartesian grid frame produced by
:func:`~dasfm.forward.geometry.build_model_grid`
(x = northing [km], y = easting [km]).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from dasfm.utils.step_utils import log_or_print


def subsample_das_channels(
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    dc: int,
    bend_threshold_deg: float = 5.0,
    smooth_window: int = 1,
    verbose: bool = True,
    logger=None,
) -> np.ndarray:
    """Subsample DAS channels by stride *dc*, preserving fiber bend points.

    Parameters
    ----------
    receiver_x : array-like, shape (n_ch,)
        Channel northing coordinates [km], e.g. ``geo["receiver_x"]``.
    receiver_y : array-like, shape (n_ch,)
        Channel easting coordinates [km], e.g. ``geo["receiver_y"]``.
    dc : int
        Uniform subsampling stride.  Indices ``0, dc, 2·dc, …`` are always
        included, plus the last channel.
    bend_threshold_deg : float
        Cumulative absolute azimuth-change threshold [degrees] applied along
        the *smoothed* path.  Starting from the previous anchor, the absolute
        per-segment azimuth differences are summed; when the running sum
        reaches this value, the current channel is marked as a new bend
        anchor and the accumulator resets.  Default 5.0°.
    smooth_window : int
        Boxcar window width [channels] applied to *receiver_x* and
        *receiver_y* before bend detection.  Use ``1`` (default) for no
        smoothing.  Values of 10–50 are typical for noisy GPS tracks.
        Boundary samples are handled with mirror-reflection padding
        (``scipy.ndimage.uniform_filter1d`` mode ``"mirror"``).
    verbose : bool
        Print a one-line summary.  Default True.

    Returns
    -------
    idx : np.ndarray, shape (n_selected,), int64
        Sorted indices of the selected channels (union of uniform subsample
        and bend points).

    Examples
    --------
    ::

        geo = build_model_grid(catalog, receiver_lat=rec_lat,
                               receiver_lon=rec_lon, ...)

        idx = subsample_das_channels(
            geo["receiver_x"], geo["receiver_y"],
            dc=50, bend_threshold_deg=5.0, smooth_window=20,
        )

        # Re-run build_model_grid with subsampled receivers
        geo_sub = build_model_grid(catalog,
                                   receiver_lat=rec_lat[idx],
                                   receiver_lon=rec_lon[idx], ...)
    """
    receiver_x = np.asarray(receiver_x, dtype=np.float64)
    receiver_y = np.asarray(receiver_y, dtype=np.float64)
    n_ch = len(receiver_x)

    if n_ch == 0:
        return np.array([], dtype=np.int64)

    # ── 1. Smooth coordinates before bend detection ───────────────────────────
    w = int(smooth_window)
    if w > 1:
        # uniform_filter1d with mode="mirror" avoids edge discontinuities
        x_s = uniform_filter1d(receiver_x, size=w, mode="mirror")
        y_s = uniform_filter1d(receiver_y, size=w, mode="mirror")
    else:
        x_s = receiver_x
        y_s = receiver_y

    # ── 2. Uniform subsample ──────────────────────────────────────────────────
    uniform_idx = np.arange(0, n_ch, int(dc), dtype=np.int64)
    # Always include the last channel
    if uniform_idx[-1] != n_ch - 1:
        uniform_idx = np.append(uniform_idx, n_ch - 1)

    # ── 3. Bend point detection: cumulative azimuth on smoothed path ──────────
    bend_idx = np.array([], dtype=np.int64)

    if n_ch >= 3:
        dx = np.diff(x_s)                          # (n_ch-1,) segment dx [km]
        dy = np.diff(y_s)
        seg_len = np.hypot(dx, dy)                 # (n_ch-1,)

        # Per-segment azimuth [deg]; undefined for zero-length segments
        azi = np.degrees(np.arctan2(dy, dx))       # (n_ch-1,)

        # Per-interior-point azimuth change, wrapped to (-180, 180]
        # delta_azi[k] is the bend at channel index k+1
        raw_diff = np.diff(azi)                    # (n_ch-2,)
        delta_azi = (raw_diff + 180.0) % 360.0 - 180.0

        # Zero out contributions from zero-length segments on either side
        valid = (seg_len[:-1] > 0) & (seg_len[1:] > 0)
        delta_azi = np.where(valid, delta_azi, 0.0)

        abs_delta = np.abs(delta_azi)

        # Walk along the path accumulating |delta_azi| from the last anchor.
        # Reset to 0 when the running sum reaches the threshold.
        bend_list = []
        cum = 0.0
        thr = float(bend_threshold_deg)
        for k in range(abs_delta.shape[0]):
            cum += float(abs_delta[k])
            if cum >= thr:
                bend_list.append(k + 1)            # bend at channel k+1
                cum = 0.0
        bend_idx = np.asarray(bend_list, dtype=np.int64)

    # ── 4. Union ──────────────────────────────────────────────────────────────
    idx = np.union1d(uniform_idx, bend_idx).astype(np.int64)

    if verbose:
        log_or_print(logger,
            f"[subsample_das_channels]  n_ch={n_ch}  dc={dc}  "
            f"smooth_window={w}  cum_bend_threshold={bend_threshold_deg}°\n"
            f"  uniform: {len(uniform_idx)}  cum-bends: {len(bend_idx)}  "
            f"union: {len(idx)}  "
            f"(compression {n_ch / len(idx):.1f}×)"
        )

    return idx
