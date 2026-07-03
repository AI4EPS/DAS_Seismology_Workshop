"""
Kagan angle and weighted average utilities (numpy).

Extracted from hashTorch.py (old_code). Converted from torch to numpy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform

from . import moment_tensor as pmt


# ---------------------------------------------------------------------------
#  Kagan angle (numpy, vectorised)
# ---------------------------------------------------------------------------
def dcm2kagan(dcm1: np.ndarray, dcm2: np.ndarray, reduce: bool = True):
    """
    Vectorised Kagan angle from DCM arrays.

    Args:
        dcm1, dcm2: (n, 3, 3) numpy arrays.
    Returns:
        kagan: (n,) if reduce else (n, 4).
    """
    if dcm1.ndim == 2:
        dcm1 = dcm1[np.newaxis]
        dcm2 = dcm2[np.newaxis]
    n = dcm1.shape[0]
    kagan = np.zeros((n, 4), dtype=dcm1.dtype)

    U = dcm1 @ dcm2.transpose(0, 2, 1)
    c = np.clip((U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2] - 1) / 2, -1.0, 1.0)
    kagan[:, 0] = np.arccos(c)

    for idx, d_vals in enumerate([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]], start=1):
        d = np.array(d_vals, dtype=dcm2.dtype)
        U = dcm1 @ (dcm2 * d).transpose(0, 2, 1)
        c = np.clip((U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2] - 1) / 2, -1.0, 1.0)
        kagan[:, idx] = np.arccos(c)

    kagan = np.degrees(kagan)
    return np.min(kagan, axis=-1) if reduce else kagan


# ---------------------------------------------------------------------------
#  Pairwise Kagan distance
# ---------------------------------------------------------------------------
def sdr2kagan_pdist(stk, dip, rak, square=False):
    """
    Pairwise Kagan angles for a set of focal mechanisms (degrees).
    Returns condensed distance vector (or square matrix if square=True).
    """
    nfm = len(stk)
    if nfm == 1:
        return np.array([0.0])
    dcm = np.asarray(pmt.sdr2dcm(stk, dip, rak), dtype=np.float32)
    index_i, index_j = np.triu_indices(nfm, k=1)
    kagan_p = dcm2kagan(dcm[index_i], dcm[index_j], reduce=True)
    if square:
        return squareform(kagan_p)
    return kagan_p


# ---------------------------------------------------------------------------
#  Weighted average of focal mechanisms
# ---------------------------------------------------------------------------
def get_weight_from_misfit(misfit):
    """Convert misfit values to Gaussian weights."""
    if np.min(misfit) < 1e-8:
        return np.exp(-(((misfit - np.min(misfit)) / 1) ** 2))
    return np.exp(-(((misfit - np.min(misfit)) / np.min(misfit)) ** 2))


def sdr_weighted_average(stk, dip, rak, weight=None):
    """
    SKHASH-style weighted average of focal mechanisms.

    Returns:
        N_avg, S_avg (unit vectors), kagan, angle_N_deg, angle_S_deg
    """
    nfm = len(stk)
    if weight is None:
        weight1 = np.ones(nfm)
    else:
        weight1 = np.array(weight)
        weight1 = weight1 / np.sum(weight1) * nfm

    Nmat = np.zeros((nfm, 3))
    Smat = np.zeros((nfm, 3))
    for i in range(nfm):
        Nmat[i, :], Smat[i, :] = pmt.sdr2ns(stk[i], dip[i], rak[i])

    # alignment to best-weight reference
    best_idx = np.argmax(weight1)
    N_ref, S_ref = Nmat[best_idx], Smat[best_idx]
    N_aligned = np.zeros_like(Nmat)
    S_aligned = np.zeros_like(Smat)

    for i in range(nfm):
        candidates = [
            ( Nmat[i],  Smat[i]),
            (-Nmat[i], -Smat[i]),
            ( Smat[i],  Nmat[i]),
            (-Smat[i], -Nmat[i]),
        ]
        best_score = -np.inf
        best_cand = candidates[0]
        for Nc, Sc in candidates:
            score = np.dot(Nc, N_ref) + np.dot(Sc, S_ref)
            if score > best_score:
                best_score = score
                best_cand = (Nc, Sc)
        N_aligned[i], S_aligned[i] = best_cand

    # weighted average
    N_avg = np.sum(N_aligned * weight1[:, np.newaxis], axis=0)
    S_avg = np.sum(S_aligned * weight1[:, np.newaxis], axis=0)
    norm_n, norm_s = np.linalg.norm(N_avg), np.linalg.norm(S_avg)

    if norm_n < 1e-9 or norm_s < 1e-9:
        return N_ref, S_ref, np.zeros(nfm), np.zeros(nfm), np.zeros(nfm)

    N_avg /= norm_n
    S_avg /= norm_s

    # deviation
    dot_n = np.clip(np.sum(N_aligned * N_avg, axis=1), -1.0, 1.0)
    dot_s = np.clip(np.sum(S_aligned * S_avg, axis=1), -1.0, 1.0)
    diff_n = np.arccos(dot_n)
    diff_s = np.arccos(dot_s)
    avang1 = np.sqrt(np.sum(weight1 * diff_n**2) / np.sum(weight1))
    avang2 = np.sqrt(np.sum(weight1 * diff_s**2) / np.sum(weight1))

    # SKHASH-style iterative orthogonalisation
    if (avang1 + avang2) >= 0.0001:
        maxmisf = 0.01
        fract1 = avang1 / (avang1 + avang2)
        for _ in range(100):
            dot1 = np.dot(N_avg, S_avg)
            misf = 90.0 - np.degrees(np.arccos(np.clip(dot1, -1.0, 1.0)))
            if abs(misf) <= maxmisf:
                break
            theta1 = np.radians(misf * fract1)
            theta2 = np.radians(misf * (1.0 - fract1))
            temp_N = N_avg.copy()
            N_avg = N_avg - S_avg * np.sin(theta1)
            S_avg = S_avg - temp_N * np.sin(theta2)
            N_avg /= np.linalg.norm(N_avg)
            S_avg /= np.linalg.norm(S_avg)

    kagan = np.zeros(nfm)
    return N_avg, S_avg, kagan, np.degrees(diff_n), np.degrees(diff_s)


def sdr_weighted_average_iter(stk, dip, rak, weight=None, threshold=30, iter_max=5):
    """Iteratively remove outliers and compute weighted average."""
    idx_keep = np.arange(len(stk))
    Nvec_unit = Svec_unit = kagan = angle_N = angle_S = None
    for _ in range(iter_max):
        Nvec_unit, Svec_unit, kagan, angle_N, angle_S = sdr_weighted_average(
            stk[idx_keep], dip[idx_keep], rak[idx_keep],
            weight=weight[idx_keep],
        )
        isel = kagan <= threshold
        idx_keep = idx_keep[isel]
        if np.count_nonzero(~isel) == 0:
            break
    return Nvec_unit, Svec_unit, kagan, angle_N, angle_S
