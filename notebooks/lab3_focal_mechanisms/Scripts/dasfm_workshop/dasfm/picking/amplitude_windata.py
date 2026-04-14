"""S/P amplitude ratio extraction from pre-extracted P/S windows.

Algorithm
---------
The amplitude of a window is its peak-to-peak value (``max − min``).
The S/P ratio is simply ``ptp(s_data) / ptp(p_data)`` per channel.

S-wave contamination check
--------------------------
For near-offset channels the S-wave may arrive before the end of the P
window.  A per-event warning is emitted when any channel satisfies::

    s_traveltime[k] − p_traveltime[k]  <  (n_win_p − p_shift_index[k]) × dt

i.e. the S arrival falls inside the P window.  Affected channel counts are
reported; no data modification is performed.

Public function
---------------
* :func:`pick_sp_ratio_windata`
"""

from __future__ import annotations

import warnings

import numpy as np
from tqdm import tqdm

from dasfm.io.data import DASData
from dasfm.utils.step_utils import log_or_print


def pick_sp_ratio_single(event: DASData) -> dict:
    """Per-channel S/P amplitude ratio for a single event.

    The amplitude of each window is its peak-to-peak value (``max − min``).
    A contamination check warns when the S arrival falls inside the P window
    for any channel, but does not modify the data.

    Parameters
    ----------
    event : DASData
        Pre-loaded event data.  Must provide ``p_data``, ``s_data``,
        ``p_shift_index``, ``p_traveltime``, ``s_traveltime``, and ``dt``.

    Returns
    -------
    dict with keys:
        ``p_amp``    — (n_ch,) float32, peak-to-peak P amplitude
        ``s_amp``    — (n_ch,) float32, peak-to-peak S amplitude
        ``sp_valid`` — (n_ch,) bool, True = valid measurement
    """
    p_data = np.asarray(event.p_data, dtype=np.float32)   # (n_ch, n_win_p)
    s_data = np.asarray(event.s_data, dtype=np.float32)   # (n_ch, n_win_s)
    n_ch = p_data.shape[0]
    n_win_p = p_data.shape[1]
    dt = event.dt

    # ── amplitude: peak-to-peak of the full window ────────────────────
    p_amp = p_data.max(axis=1) - p_data.min(axis=1)       # (n_ch,)
    s_amp = s_data.max(axis=1) - s_data.min(axis=1)       # (n_ch,)

    # ── S-contamination check ─────────────────────────────────────────
    p_shift  = np.asarray(event.p_shift_index, dtype=np.int64)
    p_tail_s = (n_win_p - p_shift) * dt                   # [s] (n_ch,)
    sp_gap   = (np.asarray(event.s_traveltime)
                - np.asarray(event.p_traveltime))          # [s] (n_ch,)
    contaminated = sp_gap < p_tail_s
    n_cont = int(contaminated.sum())
    if n_cont > 0:
        eid = event.event_id or "unknown"
        warnings.warn(
            f"[sp_ratio] {eid}: {n_cont}/{n_ch} channels may have "
            f"S-wave contamination in P window "
            f"(min S-P gap {sp_gap[contaminated].min():.3f}s < "
            f"P tail {p_tail_s[contaminated].max():.3f}s)",
            stacklevel=2,
        )

    # ── validity mask ─────────────────────────────────────────────────
    qc = p_amp > 0
    if event.valid_mask is not None:
        qc &= np.asarray(event.valid_mask, dtype=bool)

    return {
        "p_amp":    p_amp,
        "s_amp":    s_amp,
        "sp_valid": qc,
    }


def pick_sp_ratio_windata(
    events: list[DASData],
    logger=None,
) -> dict:
    """Per-channel S/P amplitude ratio from pre-extracted P and S windows.

    Calls :func:`pick_sp_ratio_single` per event.

    Parameters
    ----------
    events : list[DASData]
        Pre-loaded event data.

    Returns
    -------
    dict with keys:
        ``sp_ratio``  — (n_ch, n_ev) float32, S/P amplitude ratio; 0 where invalid
        ``sp_valid``  — (n_ch, n_ev) bool, True = valid measurement
        ``p_amp``     — (n_ch, n_ev) float32, peak-to-peak P amplitude
        ``s_amp``     — (n_ch, n_ev) float32, peak-to-peak S amplitude
        ``event_ids`` — list[str] event IDs in input order
    """
    if len(events) == 0:
        raise ValueError("events list is empty")

    n_ch = events[0].n_channels
    n_ev = len(events)

    log_or_print(logger, f"[sp_ratio] events={n_ev}  channels={n_ch}")

    sp_ratio  = np.zeros((n_ch, n_ev), dtype=np.float32)
    sp_valid  = np.zeros((n_ch, n_ev), dtype=bool)
    p_amp_out = np.zeros((n_ch, n_ev), dtype=np.float32)
    s_amp_out = np.zeros((n_ch, n_ev), dtype=np.float32)

    for ev_idx, event in enumerate(tqdm(events, desc="S/P ratio", unit="event")):
        res = pick_sp_ratio_single(event)
        safe_p = np.where(res["p_amp"] > 0, res["p_amp"], 1.0)
        sp_ratio[:, ev_idx]  = np.where(res["sp_valid"], res["s_amp"] / safe_p, 0.0)
        sp_valid[:, ev_idx]  = res["sp_valid"]
        p_amp_out[:, ev_idx] = res["p_amp"]
        s_amp_out[:, ev_idx] = res["s_amp"]

    return {
        "sp_ratio":  sp_ratio,
        "sp_valid":  sp_valid,
        "p_amp":     p_amp_out,
        "s_amp":     s_amp_out,
        "event_ids": [e.event_id for e in events],
    }
