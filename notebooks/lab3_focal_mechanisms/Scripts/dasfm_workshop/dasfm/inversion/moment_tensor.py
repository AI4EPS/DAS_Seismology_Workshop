"""
Moment tensor utilities: SDR ↔ normal/slip vectors, DCM, T/B/P axes, Kagan angle (numpy).

Extracted from MomentTensor.py (old_code).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
#  SDR ↔ fault vectors
# ---------------------------------------------------------------------------
def sdr2ns(stk: float, dip: float, rak: float):
    """
    Strike/dip/rake (degrees) → fault normal & slip vectors.
    Reference: Aki & Richards (2002), p. 108-110.
    """
    d2r = np.pi / 180
    stk1, dip1, rak1 = stk * d2r, dip * d2r, rak * d2r
    n1 = -np.sin(dip1) * np.sin(stk1)
    n2 =  np.sin(dip1) * np.cos(stk1)
    n3 = -np.cos(dip1)
    v1 =  np.cos(rak1) * np.cos(stk1) + np.cos(dip1) * np.sin(rak1) * np.sin(stk1)
    v2 =  np.cos(rak1) * np.sin(stk1) - np.cos(dip1) * np.sin(rak1) * np.cos(stk1)
    v3 = -np.sin(rak1) * np.sin(dip1)
    return np.array([n1, n2, n3]), np.array([v1, v2, v3])


def ns2sdr(vec_normal, vec_slip):
    """Normal & slip vectors → strike/dip/rake (degrees)."""
    r2d = 180 / np.pi
    if 1 - np.abs(vec_normal[2]) < 1e-14:
        dip = 0
        stk = np.arctan2(-vec_slip[0], vec_slip[1])
        rak = np.arctan2(
            np.sin(stk) * vec_slip[0] - np.cos(stk) * vec_slip[1],
            np.cos(stk) * vec_slip[0] + np.sin(stk) * vec_slip[1],
        )
    else:
        dip = np.arctan2(
            np.sqrt(vec_normal[0] ** 2 + vec_normal[1] ** 2), -vec_normal[2]
        )
        stk = np.arctan2(-vec_normal[0], vec_normal[1])
        rak = np.arctan2(
            -vec_slip[2] / np.sin(dip),
            np.cos(stk) * vec_slip[0] + np.sin(stk) * vec_slip[1],
        )
    if dip >= np.pi * 0.5:
        dip = np.pi - dip
        stk += np.pi
        rak = -rak
    stk *= r2d
    dip *= r2d
    rak *= r2d
    if stk < 0:
        stk += 360
    if rak <= -180:
        rak += 360
    elif rak >= 180:
        rak -= 360
    return stk, dip, rak


def auxplane(stk1, dip1, rak1):
    """Auxiliary fault plane from primary SDR."""
    if np.isscalar(stk1):
        vec_normal, vec_slip = sdr2ns(stk1, dip1, rak1)
        return ns2sdr(vec_slip, vec_normal)
    else:
        stk2 = np.zeros_like(stk1, dtype=np.float32)
        dip2 = np.zeros_like(stk1, dtype=np.float32)
        rak2 = np.zeros_like(stk1, dtype=np.float32)
        for i in range(len(stk2)):
            stk2[i], dip2[i], rak2[i] = auxplane(stk1[i], dip1[i], rak1[i])
        return stk2, dip2, rak2


# ---------------------------------------------------------------------------
#  T / B / P axes
# ---------------------------------------------------------------------------
def ns2tbp(vec_normal, vec_slip):
    """Normal & slip vectors → T, B, P axes."""
    vecT = (vec_normal + vec_slip) / np.sqrt(2)
    vecP = (vec_normal - vec_slip) / np.sqrt(2)
    vecB = np.cross(vecT, vecP, axis=0)
    return vecT, vecB, vecP


def sdr2tbp(stk, dip, rak):
    """Strike/dip/rake → T, B, P axes."""
    vec_normal, vec_slip = sdr2ns(stk, dip, rak)
    return ns2tbp(vec_normal, vec_slip)


# ---------------------------------------------------------------------------
#  Direction Cosine Matrix (DCM)
# ---------------------------------------------------------------------------
def sdr2dcm(stk, dip, rak):
    """
    SDR → DCM matrix.
    Scalar input → (3,3);  array input → (n, 3, 3).
    Reference: Kagan (2007).
    """
    if np.isscalar(stk):
        vecT, vecB, vecP = sdr2tbp(stk, dip, rak)
        return np.vstack([vecT, vecP, vecB]).T
    else:
        n = len(stk)
        D = np.zeros([n, 3, 3])
        for i in range(n):
            D[i, :, :] = sdr2dcm(stk[i], dip[i], rak[i])
        return D


# ---------------------------------------------------------------------------
#  Kagan angle (numpy)
# ---------------------------------------------------------------------------
def tbp2kagan(T1, B1, P1, T2, B2, P2, reduce=True):
    """Kagan angle from T/B/P axes (scalar, numpy)."""
    D1 = np.vstack([T1, P1, B1]).T
    D2 = np.vstack([T2, P2, B2]).T
    kagan = np.zeros(4)
    c = min(max(((D1 @ D2.T).trace() - 1) / 2, -1.0), 1.0)
    kagan[0] = np.arccos(c)
    c = min(max(((D1 @ (D2 * np.array([-1, -1, 1])).T).trace() - 1) / 2, -1.0), 1.0)
    kagan[1] = np.arccos(c)
    c = min(max(((D1 @ (D2 * np.array([-1, 1, -1])).T).trace() - 1) / 2, -1.0), 1.0)
    kagan[2] = np.arccos(c)
    c = min(max(((D1 @ (D2 * np.array([1, -1, -1])).T).trace() - 1) / 2, -1.0), 1.0)
    kagan[3] = np.arccos(c)
    kagan *= 180 / np.pi
    return np.min(kagan) if reduce else kagan


def dcm2kagan_np(dcm1, dcm2, reduce=True):
    """Vectorised Kagan angle from DCM arrays (numpy)."""
    if len(dcm1.shape) == 2:
        dcm1 = np.expand_dims(dcm1, axis=0)
        dcm2 = np.expand_dims(dcm2, axis=0)
    n = dcm1.shape[0]
    kagan = np.zeros([n, 4])
    U = np.matmul(dcm1, dcm2.transpose(0, 2, 1))
    c = np.minimum(np.maximum((U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2] - 1) / 2, -1.0), 1.0)
    kagan[:, 0] = np.arccos(c)
    for idx, d in enumerate([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]], start=1):
        U = np.matmul(dcm1, (dcm2 * np.array(d)).transpose(0, 2, 1))
        c = np.minimum(np.maximum((U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2] - 1) / 2, -1.0), 1.0)
        kagan[:, idx] = np.arccos(c)
    kagan *= 180 / np.pi
    return np.min(kagan, axis=1) if reduce else kagan


def sdr2kagan(s1, d1, r1, s2, d2, r2, reduce=True):
    """Kagan angle from two SDR triples."""
    T1, B1, P1 = sdr2tbp(s1, d1, r1)
    T2, B2, P2 = sdr2tbp(s2, d2, r2)
    return tbp2kagan(T1, B1, P1, T2, B2, P2, reduce=reduce)


# ---------------------------------------------------------------------------
#  Radiation pattern (Aki & Richards)
# ---------------------------------------------------------------------------
def radiation_pattern(fm, takeoff, azimuth, type="P"):
    """
    Far-field radiation pattern.  fm = [strike, dip, rake] in degrees.
    takeoff/azimuth in degrees.
    """
    d2r = np.pi / 180
    inc = takeoff * d2r
    azi = azimuth * d2r
    strike, dip, rake = fm[0] * d2r, fm[1] * d2r, fm[2] * d2r
    si, ci = np.sin(inc), np.cos(inc)
    s2i, c2i = np.sin(2 * inc), np.cos(2 * inc)
    sd, cd = np.sin(dip), np.cos(dip)
    s2d, c2d = np.sin(2 * dip), np.cos(2 * dip)
    sr, cr = np.sin(rake), np.cos(rake)
    sas, cas = np.sin(azi - strike), np.cos(azi - strike)
    s2as = 2 * sas * cas
    c2as = cas**2 - sas**2
    if type == "P":
        return (
            -cas * cd * cr * s2i
            + cr * s2as * sd * si**2
            + c2d * s2i * sas * sr
            + s2d * (ci**2 + (-1) * sas**2 * si**2) * sr
        )
    elif type == "SV":
        return (
            -c2i * cas * cd * cr
            + 0.5 * cr * s2as * s2i * sd
            + c2d * c2i * sas * sr
            + (-0.5) * s2d * s2i * (1 + sas**2) * sr
        )
    elif type == "SH":
        return (
            cd * ci * cr * sas
            + c2as * cr * sd * si
            + c2d * cas * ci * sr
            + (-0.5) * s2as * s2d * si * sr
        )


def compute_sta_pol_misfit(stk_g, dip_g, rak_g, takeoffs, azimuths, polarities):
    """Compute station polarity misfit for all focal mechanisms (numpy).

    Parameters
    ----------
    stk_g, dip_g, rak_g : array (num_Mt,)
        Strike/dip/rake grids in degrees.
    takeoffs : array (n_sta,)
        Takeoff angles in degrees.
    azimuths : array (n_sta,)
        Azimuths in degrees.
    polarities : array (n_sta,)
        Observed polarities (+1/-1).

    Returns
    -------
    misfit : array (num_Mt,), float32
        Number of mismatched polarities (weighted by 1 per station).
    """
    sdr = [stk_g, dip_g, rak_g]
    n_sta = len(takeoffs)
    num_Mt = len(stk_g)
    misfit = np.zeros(num_Mt, dtype=np.float32)
    for i in range(n_sta):
        rad_p = radiation_pattern(sdr, takeoffs[i], azimuths[i], "P")
        weight = float(np.abs(polarities[i]))
        misfit += (np.sign(rad_p) != np.sign(polarities[i])).astype(np.float32) * weight
    return misfit


def compute_sta_sp_theo(stk_g, dip_g, rak_g, takeoffs, azimuths):
    """Compute theoretical log10(S/P) for station data (numpy).

    Parameters
    ----------
    stk_g, dip_g, rak_g : array (num_Mt,)
        Strike/dip/rake grids in degrees.
    takeoffs : array (n_sta,)
        Takeoff angles in degrees.
    azimuths : array (n_sta,)
        Azimuths in degrees.

    Returns
    -------
    sp_theo : array (num_Mt, n_sta), float32
        Theoretical log10(S/P) amplitude ratios.
    """
    sdr = [stk_g, dip_g, rak_g]
    n_sta = len(takeoffs)
    num_Mt = len(stk_g)
    sp_theo = np.zeros((num_Mt, n_sta), dtype=np.float32)
    for i in range(n_sta):
        P_amp  = radiation_pattern(sdr, takeoffs[i], azimuths[i], "P")
        SV_amp = radiation_pattern(sdr, takeoffs[i], azimuths[i], "SV")
        SH_amp = radiation_pattern(sdr, takeoffs[i], azimuths[i], "SH")
        S_amp  = np.sqrt(SV_amp**2 + SH_amp**2)
        sp_theo[:, i] = np.log10(np.abs(S_amp) / (np.abs(P_amp) + 1e-30)).astype(np.float32)
    return sp_theo


# ---------------------------------------------------------------------------
#  Torch versions of STA forward (GPU-accelerated)
# ---------------------------------------------------------------------------

def radiation_pattern_batch_torch(stk_t, dip_t, rak_t, takeoffs_t, azimuths_t):
    """Vectorized P/SV/SH radiation pattern for all mechanisms × all stations.

    Aki & Richards far-field radiation pattern, vectorized over both
    (num_Mt,) mechanisms and (n_sta,) stations using broadcasting.

    Parameters
    ----------
    stk_t, dip_t, rak_t : torch.Tensor (num_Mt,)
    takeoffs_t, azimuths_t : torch.Tensor (n_sta,)

    Returns
    -------
    P, SV, SH : torch.Tensor (num_Mt, n_sta)
    """
    import torch
    d2r = torch.pi / 180.0

    # (num_Mt, 1)
    strike = stk_t[:, None] * d2r
    dip    = dip_t[:, None] * d2r
    rake   = rak_t[:, None] * d2r

    # (1, n_sta)
    inc = takeoffs_t[None, :] * d2r
    azi = azimuths_t[None, :] * d2r

    si, ci = torch.sin(inc), torch.cos(inc)
    s2i, c2i = torch.sin(2 * inc), torch.cos(2 * inc)
    sd, cd = torch.sin(dip), torch.cos(dip)
    s2d, c2d = torch.sin(2 * dip), torch.cos(2 * dip)
    sr, cr = torch.sin(rake), torch.cos(rake)
    sas = torch.sin(azi - strike)
    cas = torch.cos(azi - strike)
    s2as = 2 * sas * cas
    c2as = cas**2 - sas**2

    P = (-cas * cd * cr * s2i
         + cr * s2as * sd * si**2
         + c2d * s2i * sas * sr
         + s2d * (ci**2 + (-1) * sas**2 * si**2) * sr)

    SV = (-c2i * cas * cd * cr
          + 0.5 * cr * s2as * s2i * sd
          + c2d * c2i * sas * sr
          + (-0.5) * s2d * s2i * (1 + sas**2) * sr)

    SH = (cd * ci * cr * sas
          + c2as * cr * sd * si
          + c2d * cas * ci * sr
          + (-0.5) * s2as * s2d * si * sr)

    return P, SV, SH


def compute_sta_pol_misfit_torch(stk_t, dip_t, rak_t, takeoffs_t, azimuths_t, polarities_t):
    """Compute station polarity misfit for all focal mechanisms (torch).

    Parameters
    ----------
    stk_t, dip_t, rak_t : torch.Tensor (num_Mt,) on device
    takeoffs_t, azimuths_t : torch.Tensor (n_sta,) on device
    polarities_t : torch.Tensor (n_sta,) on device

    Returns
    -------
    misfit : torch.Tensor (num_Mt,) on device
    """
    import torch
    P, _, _ = radiation_pattern_batch_torch(stk_t, dip_t, rak_t, takeoffs_t, azimuths_t)
    weights = torch.abs(polarities_t)  # (n_sta,)
    mismatch = (torch.sign(P) != torch.sign(polarities_t[None, :]))  # (num_Mt, n_sta)
    return (mismatch.float() * weights[None, :]).sum(dim=1)  # (num_Mt,)


def compute_sta_sp_theo_torch(stk_t, dip_t, rak_t, takeoffs_t, azimuths_t):
    """Compute theoretical log10(S/P) for station data (torch).

    Parameters
    ----------
    stk_t, dip_t, rak_t : torch.Tensor (num_Mt,) on device
    takeoffs_t, azimuths_t : torch.Tensor (n_sta,) on device

    Returns
    -------
    sp_theo : torch.Tensor (num_Mt, n_sta) on device
    """
    import torch
    P, SV, SH = radiation_pattern_batch_torch(stk_t, dip_t, rak_t, takeoffs_t, azimuths_t)
    S = torch.sqrt(SV**2 + SH**2)
    return torch.log10(torch.abs(S) / (torch.abs(P) + 1e-30))
