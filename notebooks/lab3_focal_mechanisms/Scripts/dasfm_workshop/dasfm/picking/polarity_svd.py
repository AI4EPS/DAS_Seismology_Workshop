"""SVD-based absolute polarity inversion from pairwise relative polarity matrices.

The SVD step resolves the relative sign between channels but leaves a global
±1 ambiguity **per event** (the leading singular vector can be globally
flipped).  When a few traditional-station first-motion polarities are
available, they can anchor this ambiguity via a per-event majority vote.

Public function
---------------
* :func:`solve_polarity_svd`
"""

from __future__ import annotations

import numpy as np
import torch

from dasfm.picking.mccc import Pkic_from_Ckij_Skij
from dasfm.utils.step_utils import log_or_print


def solve_polarity_svd(
    pol_com_mccc,
    pol_com_mccc_shift,
    station_polarities: np.ndarray | None = None,
    station_channel_indices: np.ndarray | None = None,
    station_event_indices: np.ndarray | None = None,
    logger=None,
    **kwargs,
) -> dict:
    """Resolve absolute polarity per channel and event via SVD.

    This is the second stage of polarity picking, applied to the pairwise
    relative polarity matrices (Ckij/Skij) produced by the MCCC pipeline
    (:mod:`dasfm.picking.mccc_kernel`).

    After SVD, a per-event global sign ambiguity remains (all DAS channels for
    one event can be collectively flipped without changing the relative
    polarity).  If station first-motion readings are supplied, each event's
    DAS polarity column is compared against those readings at the nearest DAS
    channels via majority vote, and the column is flipped if they disagree.

    Parameters
    ----------
    pol_com_mccc : np.ndarray, shape (n_ch, n_ev, n_ev)
        Symmetric relative polarity matrix at zero spatial shift,
        as returned by the ``pick_polarity_*`` functions.
    pol_com_mccc_shift : np.ndarray, shape (n_ch, n_ev, n_ev)
        Symmetric relative polarity matrix with one-channel spatial shift.
    station_polarities : array-like of int, shape (n_readings,) or None
        First-motion polarity for each station reading.
        Values: ``+1`` (up / compression) or ``-1`` (down / dilatation).
    station_channel_indices : array-like of int, shape (n_readings,) or None
        DAS channel index closest to the recording station for each reading.
    station_event_indices : array-like of int, shape (n_readings,) or None
        Index of the event (0-based, matching the event axis of
        *pol_com_mccc*) to which each reading belongs.

    All three station arrays must be supplied together and have the same
    length *n_readings*.  A "reading" is one (station, event) pair; not
    every station needs a reading for every event.

    Example
    -------
    Three stations, readings for a subset of events::

        station_polarities      = np.array([ +1, -1, +1, -1,  +1])
        station_channel_indices = np.array([120, 120, 840, 840, 3200])
        station_event_indices   = np.array([  0,   3,   1,   3,    2])

    Returns
    -------
    dict with keys:
        ``Pkic``         — (n_ch, n_ev) float32 absolute polarity
        ``svd_info``     — SVD quality metrics from
                           :func:`~dasfm.picking.mccc.Pkic_from_Ckij_Skij`
        ``flip_applied`` — (n_ev,) bool, ``True`` where the column was flipped
                           by the station-polarity vote (always all-False when
                           no station data are supplied)
    """
    # Smooth is applied per-pair in MCCC output (mccc_kernel)
    is_sparse = isinstance(pol_com_mccc, list)
    Pkic, svd_info = Pkic_from_Ckij_Skij(
        pol_com_mccc if is_sparse else torch.as_tensor(np.asarray(pol_com_mccc)),
        pol_com_mccc_shift if is_sparse else torch.as_tensor(np.asarray(pol_com_mccc_shift)),
    )
    Pkic_np = Pkic.cpu().numpy().astype(np.float32)   # (n_ch, n_ev)
    n_ev    = Pkic_np.shape[1]
    flip_applied = np.zeros(n_ev, dtype=bool)

    if (station_polarities is not None
            and station_channel_indices is not None
            and station_event_indices is not None):
        st_pol  = np.asarray(station_polarities,      dtype=np.float32)
        ch_idx  = np.asarray(station_channel_indices, dtype=int)
        ev_idx  = np.asarray(station_event_indices,   dtype=int)

        if not (len(st_pol) == len(ch_idx) == len(ev_idx)):
            raise ValueError(
                "station_polarities, station_channel_indices and "
                "station_event_indices must all have the same length"
            )

        for c in range(n_ev):
            mask = ev_idx == c
            if not np.any(mask):
                continue

            # DAS polarity at station channel locations for this event
            das_at_st = Pkic_np[ch_idx[mask], c]    # (n_readings_for_c,)

            # majority vote: positive total → agree, negative → flip
            vote = float(np.sum(np.sign(das_at_st) * st_pol[mask]))
            if vote < 0:
                Pkic_np[:, c] *= -1.0
                flip_applied[c] = True

        n_flipped = int(flip_applied.sum())
        log_or_print(logger, f"[solve_polarity_svd] station correction: "
                    f"flipped {n_flipped}/{n_ev} events")

    return {
        "Pkic":         Pkic_np,
        "svd_info":     svd_info,
        "flip_applied": flip_applied,
    }
