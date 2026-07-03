"""
Strain-rate radiation pattern for DAS focal mechanism inversion (PyTorch).

Extracted from two_funs.py (old_code).
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
#  Core radiation tensors
# ---------------------------------------------------------------------------
def strain_rate_pattern(M0, g, r, rho, VP, VS):
    """
    Compute 6 strain-rate radiation tensors (P1-P3, S1-S3) contracted with M0.

    Args:
        M0: (3, 3, N) moment tensors.
        g:  (3,) or (3, 1) unit direction vector.
        r, rho, VP, VS: scalars.
    Returns:
        A_P1..A_S3: each (3, 3, N).
    """
    device = M0.device
    g_i = g.reshape(3, 1, 1, 1)
    g_j = g.reshape(1, 3, 1, 1)
    g_k = g.reshape(1, 1, 3, 1)
    g_l = g.reshape(1, 1, 1, 3)

    delta = torch.eye(3, device=device)
    delta_ij = delta[:, :, None, None]
    delta_ik = delta[:, None, :, None]
    delta_il = delta[:, None, None, :]
    delta_jk = delta[None, :, :, None]
    delta_jl = delta[None, :, None, :]
    delta_kl = delta[None, None, :, :]

    T_P1 = -g_i * g_j * g_k * g_l

    T_P2 = (
        delta_ij * g_k * g_l - 10 * g_i * g_j * g_k * g_l
        + delta_kl * g_i * g_j + delta_jk * g_i * g_l
        + delta_ik * g_j * g_l + delta_jl * g_i * g_k
        + delta_il * g_j * g_k
    )

    T_P3 = (
        6 * delta_ij * g_k * g_l - 45 * g_i * g_j * g_k * g_l
        + 6 * delta_jk * g_i * g_l + 6 * delta_ik * g_j * g_l
        + 6 * delta_jl * g_i * g_k + 6 * delta_il * g_j * g_k
        - 2 * delta_ik * delta_jl + delta_ij * delta_kl
        + 6 * delta_kl * g_i * g_j
    )

    T_S1 = g_i * g_j * g_k * g_l - 0.5 * delta_jk * g_i * g_l - 0.5 * delta_ik * g_j * g_l

    T_S2 = (
        -delta_ij * g_k * g_l + 10 * g_i * g_j * g_k * g_l
        - 2.5 * delta_jk * g_i * g_l - 2.5 * delta_ik * g_j * g_l
        - delta_jl * g_i * g_k - delta_il * g_j * g_k
        + delta_ik * delta_jl - delta_kl * g_i * g_j
    )

    T_S3 = (
        -6 * delta_ij * g_k * g_l + 45 * g_i * g_j * g_k * g_l
        - 7.5 * delta_jk * g_i * g_l - 7.5 * delta_ik * g_j * g_l
        - 6 * delta_jl * g_i * g_k - 6 * delta_il * g_j * g_k
        + 3 * delta_ik * delta_jl + delta_ij * delta_kl
        - 6 * delta_kl * g_i * g_j
    )

    A_P1 = torch.einsum("ijkl,kln->ijn", T_P1, M0)
    A_P2 = torch.einsum("ijkl,kln->ijn", T_P2, M0)
    A_P3 = torch.einsum("ijkl,kln->ijn", T_P3, M0)
    A_S1 = torch.einsum("ijkl,kln->ijn", T_S1, M0)
    A_S2 = torch.einsum("ijkl,kln->ijn", T_S2, M0)
    A_S3 = torch.einsum("ijkl,kln->ijn", T_S3, M0)

    # Normalization: scale far-field P term (A_P1) so that unit moment tensor gives O(1).
    # Equivalent to multiplying the physical formula by 4π·ρ·r·VP⁴; does not affect polarity (sign) or inter-station amplitude ratios.
    _vp_r = VP / r                          # typical value ~4.5 (VP=4500, r=1000m)
    # A_P1 is not scaled (einsum output is already O(1))
    A_P2 *= _vp_r                          # VP/r
    A_P3 *= _vp_r ** 2                     # (VP/r)²
    A_S1 *= (VP / VS) ** 4                 # (VP/VS)⁴
    A_S2 *= _vp_r * (VP / VS) ** 3        # VP/r · (VP/VS)³
    A_S3 *= _vp_r ** 2 * (VP / VS) ** 2  # (VP/r)² · (VP/VS)²

    return A_P1, A_P2, A_P3, A_S1, A_S2, A_S3


# ---------------------------------------------------------------------------
#  DAS radiation pattern  (many M0, single station)
# ---------------------------------------------------------------------------
def strain_rate_pattern_das_torch(
    M0, tak, azi, r=1000, rho=2000, ff=None, VP=None, VS=None, vp_vs_ratio=1.7,
):
    """
    DAS strain-rate radiation pattern for N moment tensors at a single channel.

    Args:
        M0: (3, 3, N) moment tensors on GPU.
        tak, azi: scalars (torch tensors) — takeoff angle and azimuth [deg].
        r: source-receiver distance [m].
        rho: density [kg/m³].
        ff: complex frequency (torch tensor). If None, use r1-only mode.
        VP, VS: P/S wave velocities [m/s]. Required when ff is provided.
        vp_vs_ratio: VP/VS ratio (default 1.7). Used in r1-only mode.
    Returns:
        pol_P (N,), amp_tot_P (N,), amp_tot_S (N,).

    NOTE: ``pol_P`` is the **raw strain-rate sign** — it has the *opposite*
    sign convention from the particle-velocity P polarity used by traditional
    seismometers and dasfm's ``Pkic`` (which is sta4mccc_ref-calibrated to
    particle-velocity convention by step2b).  step3 polarity-misfit
    calculation should NOT consume this directly; use
    :func:`compute_das_polarity_misfit_torch` instead, which evaluates the
    Aki-Richards particle-velocity formula.

    This output is preserved unchanged so that any future application matching
    real DAS strain-rate first-arrivals (e.g. direct-wave strain measurements)
    can use it without an extra sign flip.
    """
    device = M0.device
    rad_t = tak * math.pi / 180.0
    rad_a = azi * math.pi / 180.0

    g = torch.stack([
        torch.sin(rad_t) * torch.cos(rad_a),
        torch.sin(rad_t) * torch.sin(rad_a),
        torch.cos(rad_t),
    ], dim=0).to(device)

    if ff is not None and VP is not None and VS is not None:
        # ── Full mode: all 3 terms (r1 + r2 + r3) ──
        A_P1, A_P2, A_P3, A_S1, A_S2, A_S3 = strain_rate_pattern(M0, g, r, rho, VP, VS)

        # polarity (raw strain-rate sign — see module docstring)
        ffr = ff.abs()
        pol_all = A_P1 * ffr * ffr + A_P2 * ffr + A_P3
        trace_nn = torch.einsum("iin->n", pol_all)
        pol_P = torch.sign(trace_nn)

        # total amplitude
        A_P1c = A_P1.to(torch.complex64)
        A_P2c = A_P2.to(torch.complex64)
        A_P3c = A_P3.to(torch.complex64)
        A_S1c = A_S1.to(torch.complex64)
        A_S2c = A_S2.to(torch.complex64)
        A_S3c = A_S3.to(torch.complex64)

        p_mat = torch.abs(A_P1c * ff * ff + A_P2c * ff + A_P3c)
        s_mat = torch.abs(A_S1c * ff * ff + A_S2c * ff + A_S3c)
        amp_tot_P = torch.linalg.norm(p_mat, dim=(0, 1))
        amp_tot_S = torch.linalg.norm(s_mat, dim=(0, 1))

    else:
        # ── r1-only mode: far-field term only ──
        g_i = g.reshape(3, 1, 1, 1)
        g_j = g.reshape(1, 3, 1, 1)
        g_k = g.reshape(1, 1, 3, 1)
        g_l = g.reshape(1, 1, 1, 3)

        delta = torch.eye(3, device=device)
        delta_ik = delta[:, None, :, None]
        delta_jk = delta[None, :, :, None]

        T_P1 = -g_i * g_j * g_k * g_l
        T_S1 = g_i * g_j * g_k * g_l - 0.5 * delta_jk * g_i * g_l - 0.5 * delta_ik * g_j * g_l

        A_P1 = torch.einsum("ijkl,kln->ijn", T_P1, M0)
        A_S1 = torch.einsum("ijkl,kln->ijn", T_S1, M0)

        # polarity from far-field P term (raw strain-rate sign — see docstring)
        trace_nn = torch.einsum("iin->n", A_P1)
        pol_P = torch.sign(trace_nn)

        # amplitude (S scaled by vp_vs_ratio^4)
        amp_tot_P = torch.linalg.norm(A_P1, dim=(0, 1))
        amp_tot_S = torch.linalg.norm(A_S1, dim=(0, 1)) * (vp_vs_ratio ** 4)

    return pol_P, amp_tot_P, amp_tot_S


# ---------------------------------------------------------------------------
#  All-station pattern  (single M0, many stations)
# ---------------------------------------------------------------------------
def strain_rate_pattern_allsta_torch(M0, tak, azi, vp_vs_ratio=1.7):
    """
    Simplified strain-rate pattern for 1 source (3,3) at N stations.
    Returns amp_P (N,), amp_S (N,).
    """
    device = M0.device
    N = tak.shape[0]

    rad_t = tak * math.pi / 180.0
    rad_a = azi * math.pi / 180.0

    g = torch.stack([
        torch.sin(rad_t) * torch.cos(rad_a),
        torch.sin(rad_t) * torch.sin(rad_a),
        torch.cos(rad_t),
    ], dim=0).to(device)

    M0 = M0.unsqueeze(-1).expand(3, 3, N)

    g_i = g.reshape(3, 1, 1, 1, N)
    g_j = g.reshape(1, 3, 1, 1, N)
    g_k = g.reshape(1, 1, 3, 1, N)
    g_l = g.reshape(1, 1, 1, 3, N)

    delta = torch.eye(3, device=device)
    delta_ik = delta[:, None, :, None].expand(3, 3, 3, 3).unsqueeze(-1).expand(3, 3, 3, 3, N)
    delta_jk = delta[None, :, :, None].expand(3, 3, 3, 3).unsqueeze(-1).expand(3, 3, 3, 3, N)

    T_P1 = -g_i * g_j * g_k * g_l
    T_S1 = (g_i * g_j * g_k * g_l - 0.5 * delta_jk * g_i * g_l - 0.5 * delta_ik * g_j * g_l) * vp_vs_ratio**4

    A_P1 = torch.einsum("ijklm,klm->ijm", T_P1, M0)
    A_S1 = torch.einsum("ijklm,klm->ijm", T_S1, M0)

    amp_P = torch.linalg.norm(A_P1, dim=(0, 1))
    amp_S = torch.linalg.norm(A_S1, dim=(0, 1))
    return amp_P, amp_S


# ---------------------------------------------------------------------------
#  S/P misfit norm
# ---------------------------------------------------------------------------

def sp_misfit_norm(diff, norm="L1", cauchy_c=None):
    """Compute S/P misfit with selectable norm (torch version).

    Parameters
    ----------
    diff : tensor, shape (N, n_ch) or (N, n_sta)
        Residual matrix (theo - obs).
    norm : "L1", "L2", or "cauchy"
    cauchy_c : float or None
        Fixed scale parameter for Cauchy norm.  If None, falls back to L1.
    """
    if norm == "L2":
        return torch.sum(diff ** 2, dim=1) ** 0.5
    elif norm == "cauchy" and cauchy_c is not None:
        return torch.sum(torch.log1p((diff / cauchy_c) ** 2), dim=1)
    else:  # L1
        return torch.sum(torch.abs(diff), dim=1)


def sp_misfit_norm_np(diff, norm="L1", cauchy_c=None):
    """Compute S/P misfit with selectable norm (numpy version).

    Numpy mirror of :func:`sp_misfit_norm` for code paths that operate on
    numpy arrays (e.g. STA S/P forward result after ``.cpu().numpy()``).

    Parameters
    ----------
    diff : np.ndarray, shape (N, n_ch) or (N, n_sta)
        Residual matrix (theo - obs).
    norm : "L1", "L2", or "cauchy"
    cauchy_c : float or None
        Fixed scale parameter for Cauchy norm.  If None, falls back to L1.
    """
    import numpy as np
    if norm == "L2":
        return np.sqrt(np.sum(diff ** 2, axis=1))
    if norm == "cauchy" and cauchy_c is not None:
        return np.sum(np.log1p((diff / cauchy_c) ** 2), axis=1)
    return np.sum(np.abs(diff), axis=1)


# ---------------------------------------------------------------------------
#  DAS forward (batch): polarity + S/P misfit for one event, one trial
# ---------------------------------------------------------------------------

def compute_das_forward(M_g_t, tak_deg_np, azi_deg_np,
                         sp_obs_das, do_mean_alignment, device,
                         sp_norm="L1", cauchy_c=None,
                         vp_vs_ratio=1.7):
    """Compute DAS S/P amplitude-ratio misfit for one event, one trial.

    Strain-rate forward (DAS physics) batched over moment tensors with one
    GPU dispatch per channel — this is the right strategy when ``num_Mt`` is
    large (grid stage, ~30k SDR samples).  For the recompute helpers where
    ``num_Mt = 5``, see :func:`dasfm.inversion.repr_misfit._das_sp_misfit_loop_M`
    which loops over the small candidate axis instead and is much faster.

    Polarity is computed separately by
    :func:`compute_das_polarity_misfit_torch` (Aki-Richards particle-velocity
    formula matching the ``Pkic`` convention).  This function used to also
    return a ``misfit_pol`` based on the strain-rate sign convention, but
    that path was deprecated and removed when DAS polarity physics was
    decoupled — strain-rate sign is opposite the particle-velocity sign that
    sta4mccc_ref-calibrated ``Pkic`` carries.

    Args:
        M_g_t: (3,3,N) moment tensors on device.
        tak_deg_np, azi_deg_np: (n_ch,) numpy arrays in degrees.
        sp_obs_das: (n_ch,) torch tensor.  Already includes any
            sp_decay_k * distance and sp_decay_c offsets applied by the caller.
        do_mean_alignment: bool.  When True, the per-event mean of theoretical
            log10(S/P) is shifted to match the observed mean (``corr_factor``
            is recorded).  When False, theo and obs are compared directly.
        device: str.

    Returns:
        misfit_sp: (N,) tensor or None when every ``sp_obs_das`` entry is NaN.
        corr_factor: (N,1) tensor or None — only set when ``do_mean_alignment``.
    """
    num_Mt = M_g_t.shape[2]
    n_ch = len(tak_deg_np)

    amp_tot_P = torch.zeros((num_Mt, n_ch), device=device)
    amp_tot_S = torch.zeros((num_Mt, n_ch), device=device)

    for ichan in range(n_ch):
        tak_i = torch.tensor(tak_deg_np[ichan], dtype=torch.float32, device=device)
        azi_i = torch.tensor(azi_deg_np[ichan], dtype=torch.float32, device=device)
        _, amp_tot_P[:, ichan], amp_tot_S[:, ichan] = strain_rate_pattern_das_torch(
            M_g_t, tak_i, azi_i, vp_vs_ratio=vp_vs_ratio)

    misfit_sp = corr_factor = None
    if sp_obs_das is not None:
        sp_mask = ~torch.isnan(sp_obs_das)
        if sp_mask.sum() > 0:
            midd = torch.log10(amp_tot_S / (amp_tot_P + 1e-30))
            midd_valid = midd[:, sp_mask]
            sp_valid_obs = sp_obs_das[sp_mask]
            if do_mean_alignment:
                theo_mean = midd_valid.mean(dim=1, keepdim=True)
                obs_mean = sp_valid_obs.mean()
                midd_valid = midd_valid - theo_mean + obs_mean
                corr_factor = theo_mean - obs_mean
            misfit_sp = sp_misfit_norm(midd_valid - sp_valid_obs, sp_norm, cauchy_c=cauchy_c)

    return misfit_sp, corr_factor


# ---------------------------------------------------------------------------
#  DAS polarity misfit (Aki-Richards particle-velocity, batch over channels)
# ---------------------------------------------------------------------------

def compute_das_polarity_misfit_torch(
    stk_t, dip_t, rak_t, tak_deg_np, azi_deg_np, pol_obs,
):
    """DAS polarity misfit using Aki-Richards particle-velocity P-radiation.

    Computes the polarity (sign) misfit between observed Pkic — which has
    particle-velocity sign convention after sta4mccc_ref calibration in
    step2b — and the theoretical particle-velocity P polarity from the
    Aki-Richards far-field radiation pattern.

    This function replaces the historical strain-rate-based DAS polarity
    forward (which required a manual ``-1`` flip to align with Pkic).
    The physics is now identical to STA polarity forward
    (:func:`compute_sta_pol_misfit_torch`) — DAS channels and traditional
    stations differ only in their (takeoff, azimuth) coverage, not in the
    polarity sign convention.

    Vectorised: one batched GPU op over ``(num_Mt, n_ch)``, no Python loop.

    Parameters
    ----------
    stk_t, dip_t, rak_t : torch.Tensor (num_Mt,) on device
        Strike/dip/rake of candidate mechanisms (degrees).
    tak_deg_np, azi_deg_np : np.ndarray (n_ch,)
        Per-channel takeoff and azimuth (degrees).
    pol_obs : torch.Tensor (n_ch,) on device
        Observed Pkic polarities.  Channels with ``pol_obs == 0`` (e.g.
        invalid picks masked by ``pol_valid``) have zero weight automatically.

    Returns
    -------
    misfit_pol : torch.Tensor (num_Mt,) on device
        Σ ``|pol_obs|`` over channels where ``sign(P_AR) != sign(pol_obs)``,
        i.e. weighted count of polarity disagreements.  Same convention as
        :func:`dasfm.inversion.moment_tensor.compute_sta_pol_misfit_torch`.
    """
    from dasfm.inversion.moment_tensor import radiation_pattern_batch_torch

    device = stk_t.device
    tak_t = torch.as_tensor(tak_deg_np, dtype=torch.float32, device=device)
    azi_t = torch.as_tensor(azi_deg_np, dtype=torch.float32, device=device)

    # P_AR shape: (num_Mt, n_ch) — Aki-Richards particle-velocity P amplitude
    P_AR, _, _ = radiation_pattern_batch_torch(stk_t, dip_t, rak_t, tak_t, azi_t)

    weights = torch.abs(pol_obs)                                       # (n_ch,)
    mismatch = (torch.sign(P_AR) != torch.sign(pol_obs[None, :]))      # (num_Mt, n_ch)
    return (mismatch.float() * weights[None, :]).sum(dim=1)            # (num_Mt,)
