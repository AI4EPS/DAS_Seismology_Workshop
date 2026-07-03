"""
Solution quality metrics: STDR, plane uncertainty (numpy).

Extracted from util_inv.py (old_code). Converted from torch to numpy.
"""
from __future__ import annotations

import math
import numpy as np

from . import moment_tensor as pmt
from .kagan import dcm2kagan
from dasfm.utils.step_utils import log_or_print


def _strain_rate_pattern_numpy(M0, tak, azi):
    """Simplified strain-rate pattern for 1 source (3,3) at N stations (numpy).

    Returns amp_P (N,), amp_S (N,).
    """
    rad_t = np.radians(tak)
    rad_a = np.radians(azi)

    g = np.array([
        np.sin(rad_t) * np.cos(rad_a),
        np.sin(rad_t) * np.sin(rad_a),
        np.cos(rad_t),
    ])  # (3, N)

    # P-wave: g^T M g
    Mg = M0 @ g  # (3, N)
    amp_P = np.sum(g * Mg, axis=0)  # (N,)

    # S-wave: |M g - (g^T M g) g|
    proj = amp_P[np.newaxis, :] * g  # (3, N)
    s_vec = Mg - proj  # (3, N)
    amp_S = np.sqrt(np.sum(s_vec ** 2, axis=0))  # (N,)

    return amp_P, amp_S


def cal_stdr(solution_list, pol_obs, tak_S_t, azi_S_t, device=None):
    """
    Compute STDR (Station Distribution Ratio) for each candidate solution.
    Modifies solution_list in-place, adding 'stdr' key.

    Uses Aki-Richards P-wave radiation pattern, consistent with the
    polarity forward model (compute_das_polarity_misfit_torch and
    compute_sta_pol_misfit_torch).

    All inputs are numpy arrays. device parameter is ignored (kept for compatibility).
    """
    pol_obs_np = np.asarray(pol_obs, dtype=np.float32)
    tak_np = np.asarray(tak_S_t, dtype=np.float32)
    azi_np = np.asarray(azi_S_t, dtype=np.float32)

    for i in range(len(solution_list)):
        inv_fm = solution_list[i]["fm_mean"]
        amp_P = pmt.radiation_pattern(inv_fm, tak_np, azi_np, type="P")
        wt = np.sqrt(np.abs(amp_P))
        stdr = float(np.sum(wt * np.abs(pol_obs_np)) / np.sum(np.abs(pol_obs_np)))
        solution_list[i]["stdr"] = stdr

    return solution_list


def compute_plane_uncertainty(stk_all, dip_all, rak_all, stk_mean, dip_mean, rak_mean):
    """
    RMS Kagan angle between a set of mechanisms and a reference mechanism.

    Returns:
        rms_uncertainty (float), all_kagan_angles (numpy array).
    """
    D_all = np.asarray(pmt.sdr2dcm(stk_all, dip_all, rak_all), dtype=np.float32)
    D_mean = np.asarray(pmt.sdr2dcm(stk_mean, dip_mean, rak_mean), dtype=np.float32)

    if D_mean.ndim == 2:
        D_mean = D_mean[np.newaxis]

    kagan_angles = dcm2kagan(D_all, D_mean, reduce=True)
    rms_uncertainty = float(np.sqrt(np.mean(kagan_angles ** 2)))
    return rms_uncertainty, kagan_angles
