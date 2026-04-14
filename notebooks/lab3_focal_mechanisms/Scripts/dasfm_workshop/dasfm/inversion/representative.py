"""representative.py — Stage 3: representative-mechanism misfit + top-N selection.

For each cluster candidate's ``fm_mean``:
    1. Run forward model on **trial-0 (unperturbed) ray** to compute true misfit
       (SKHASH-style, matches legacy mech_misfit semantics)
    2. Fill in misfit fields (misfit_pol0, misfit_amp0)
    3. Sort by (-accept_ratio, misfit_pol0) and **truncate to top-N (5)**
    4. Compute STDR for the top-N

This file imports torch — must be **lazy-imported** by pipeline.process_event
(not at pipeline.py top level).
"""
from __future__ import annotations
import numpy as np
import torch

from dasfm.inversion.modes import ModeSpec
from dasfm.inversion.grid import sdr_to_mt
from dasfm.inversion.forward_model import (
    compute_das_polarity_misfit_torch,
    sp_misfit_norm,
    sp_misfit_norm_np,
    strain_rate_pattern_allsta_torch,
)
from dasfm.inversion.moment_tensor import (
    compute_sta_pol_misfit_torch,
    compute_sta_sp_theo_torch,
)
from dasfm.inversion.quality import cal_stdr
from dasfm.inversion.filter import assemble_pol


# Top-N cap on candidate list returned to the caller.  cluster.py returns ALL
# surviving clusters; representative computes the true fm_mean misfit for each,
# then sorts and truncates here so downstream save / plot only sees the top-N.
TOP_N_CANDIDATES = 5


# ═══════════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_cand_tensors(cand_list, device):
    """Convert ``[c['fm_mean'] for c in cand_list]`` to torch tensors.

    Returns (M_t, stk_t, dip_t, rak_t):
        M_t   : (3, 3, N_cand) moment tensors
        stk_t : (N_cand,)
        dip_t : (N_cand,)
        rak_t : (N_cand,)
    All on `device`.
    """
    fm = np.array([c["fm_mean"] for c in cand_list], dtype=np.float64)
    M_np = sdr_to_mt(fm[:, 0], fm[:, 1], fm[:, 2]).astype(np.float32)
    M_t = torch.from_numpy(M_np).to(device)
    stk_t = torch.from_numpy(fm[:, 0].astype(np.float32)).to(device)
    dip_t = torch.from_numpy(fm[:, 1].astype(np.float32)).to(device)
    rak_t = torch.from_numpy(fm[:, 2].astype(np.float32)).to(device)
    return M_t, stk_t, dip_t, rak_t


def _percentile_in_grid(value, grid_array):
    """Fraction of grid samples ≤ value.  Returns NaN if grid is missing/empty."""
    if grid_array is None or len(grid_array) == 0:
        return float("nan")
    return float(np.sum(grid_array <= value) / len(grid_array))


def _overwrite_cand(cand_list, pol_misfit_cand, pol_weighted,
                     sp_misfit_cand=None, sp_grid=None):
    """Fill in misfit fields, sort by (-accept_ratio, misfit_pol0), truncate to top-N.

    Mutates ``cand_list`` in place.  When ``sp_misfit_cand is None`` (pol-only
    modes), ``misfit_amp0`` / ``misfit_amp_rate0`` are written as NaN.
    """
    has_sp = sp_misfit_cand is not None
    nan = float("nan")
    for i, c in enumerate(cand_list):
        c["misfit_pol0"] = float(pol_misfit_cand[i])
        c["misfit_pol_ratio0"] = float(pol_misfit_cand[i] / pol_weighted)
        if has_sp:
            c["misfit_amp0"] = float(sp_misfit_cand[i])
            c["misfit_amp_rate0"] = _percentile_in_grid(sp_misfit_cand[i], sp_grid)
        else:
            c["misfit_amp0"] = nan
            c["misfit_amp_rate0"] = nan
    cand_list.sort(
        key=lambda s: (-s["accept_ratio"], s["misfit_pol0"])
    )
    del cand_list[TOP_N_CANDIDATES:]


# ═══════════════════════════════════════════════════════════════════════════
#  Per-stage forward helpers (operating on N_cand mechanisms × trial-0 ray)
# ═══════════════════════════════════════════════════════════════════════════

def _sta_pol_cand_forward(stk_t, dip_t, rak_t, obs, device):
    """Compute STA polarity misfit on N_cand × stk/dip/rak tensors using trial-0 ray."""
    sta_takeoff_t = torch.from_numpy(np.asarray(obs.sta_takeoff_t0, dtype=np.float32)).to(device)
    sta_az_t = torch.from_numpy(np.asarray(obs.sta_az_t0, dtype=np.float32)).to(device)
    sta_pol_t = torch.from_numpy(obs.sta_pol_i.astype(np.float32)).to(device)
    return compute_sta_pol_misfit_torch(
        stk_t, dip_t, rak_t, sta_takeoff_t, sta_az_t, sta_pol_t).cpu().numpy()


def _das_sp_misfit_loop_M(M_t, tak_deg_np, azi_deg_np, sp_obs_das,
                            do_mean_alignment, device, sp_norm, cauchy_c,
                            vp_vs_ratio=1.7):
    """DAS S/P misfit on N_cand mechanisms (loop over M, batch over n_ch).

    Loops over the M (candidate) axis and calls strain_rate_pattern_allsta_torch
    (which batches over n_ch in a single GPU op).  For ``N_cand ≈ 5`` this is
    much faster than calling compute_das_forward in a Python loop over channels.

    Mirrors the SP block of compute_das_forward exactly, including
    ``do_mean_alignment`` and the NaN-mask handling.

    Returns sp_misfit (N_cand,) numpy or None when every sp_obs_das entry is NaN.
    """
    n_cand = M_t.shape[2]
    n_ch = len(tak_deg_np)
    tak_t = torch.as_tensor(tak_deg_np, dtype=torch.float32, device=device)
    azi_t = torch.as_tensor(azi_deg_np, dtype=torch.float32, device=device)
    amp_P = torch.empty((n_cand, n_ch), dtype=torch.float32, device=device)
    amp_S = torch.empty((n_cand, n_ch), dtype=torch.float32, device=device)
    for i in range(n_cand):
        amp_P[i], amp_S[i] = strain_rate_pattern_allsta_torch(
            M_t[:, :, i], tak_t, azi_t, vp_vs_ratio=vp_vs_ratio)
    sp_mask = ~torch.isnan(sp_obs_das)
    if sp_mask.sum() == 0:
        return None
    midd = torch.log10(amp_S / (amp_P + 1e-30))
    midd_valid = midd[:, sp_mask]
    sp_valid_obs = sp_obs_das[sp_mask]
    if do_mean_alignment:
        theo_mean = midd_valid.mean(dim=1, keepdim=True)
        obs_mean = sp_valid_obs.mean()
        midd_valid = midd_valid - theo_mean + obs_mean
    return sp_misfit_norm(midd_valid - sp_valid_obs, sp_norm, cauchy_c=cauchy_c).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 dispatch — uses ModeSpec, calls assemble_pol from filter.py
# ═══════════════════════════════════════════════════════════════════════════

def _compute_pol_misfit_cand(spec, stk_t, dip_t, rak_t, obs, ctx):
    """Compute fm_mean pol misfit on trial-0 ray.

    Returns (pol_misfit_cand, pol_weighted) — same shape as filter._build_solve_inputs
    but operating on 1D N_cand arrays instead of 2D (NMC, num_Mt).
    Uses the SAME assemble_pol helper from filter.py — no duplication.
    """
    n_sta = obs.n_sta
    device = ctx["device"]
    pol_obs = obs.pol_obs

    das_pol_cand = (compute_das_polarity_misfit_torch(
        stk_t, dip_t, rak_t, obs.das_tak_t0, obs.das_az_t0, pol_obs).cpu().numpy()
        if spec.has_das_pol and pol_obs is not None else None)
    sta_pol_cand = (_sta_pol_cand_forward(stk_t, dip_t, rak_t, obs, device)
        if spec.has_sta_pol and n_sta > 0 else None)

    pol_obs_sum = (float(np.abs(obs.pol_obs_cpu).sum())
                   if obs.pol_obs_cpu is not None else None)
    return assemble_pol(
        spec, das_pol_cand, sta_pol_cand,
        n_sta=n_sta, pol_obs_sum=pol_obs_sum,
        das_pol_weight=ctx["das_pol_weight"],
    )


def _compute_sp_misfit_cand(spec, M_t, stk_t, dip_t, rak_t, obs, ctx):
    """Compute fm_mean sp misfit on trial-0 ray.

    Returns (N_cand,) numpy array, or None if mode unfeasible / data missing.
    """
    if spec.sp_source == "das":
        if obs.sp_obs_das is None:
            return None
        return _das_sp_misfit_loop_M(
            M_t,
            obs.das_tak_t0, obs.das_az_t0,
            obs.sp_obs_das,
            do_mean_alignment=ctx["do_mean_alignment"],
            device=ctx["device"],
            sp_norm=ctx["sp_norm"],
            cauchy_c=ctx.get("cauchy_c"),
            vp_vs_ratio=ctx.get("vp_vs_ratio", 1.7),
        )
    if spec.sp_source == "sta":
        if obs.sta_sp_obs_i is None or obs.n_sta_sp == 0:
            return None
        device = ctx["device"]
        takeoff_t = torch.from_numpy(np.asarray(obs.sta_sp_takeoff_t0, dtype=np.float32)).to(device)
        az_t = torch.from_numpy(np.asarray(obs.sta_sp_az_t0, dtype=np.float32)).to(device)
        sp_theo = compute_sta_sp_theo_torch(stk_t, dip_t, rak_t, takeoff_t, az_t).cpu().numpy()
        return sp_misfit_norm_np(
            sp_theo - obs.sta_sp_obs_i[None, :], ctx["sp_norm"],
            cauchy_c=ctx.get("cauchy_c"))
    return None


def _stdr_args(spec, obs):
    """Return (das_args, sta_args) for STDR computation.

    Each is (pol_obs, tak, az) or None if the mode doesn't use that source.
    Joint modes return both; single-source modes return one.
    """
    das_args = None
    sta_args = None
    if spec.has_das_pol or spec.sp_source == "das":
        if obs.pol_obs_cpu is not None and obs.das_tak_t0 is not None:
            das_args = (obs.pol_obs_cpu, obs.das_tak_t0, obs.das_az_t0)
    if spec.has_sta_pol:
        if obs.sta_pol_i is not None and obs.sta_takeoff_t0 is not None:
            sta_args = (obs.sta_pol_i.astype(np.float32), obs.sta_takeoff_t0, obs.sta_az_t0)
    return das_args, sta_args


def _select_sp_grid(spec, misfits):
    """Pick the per-grid sp misfit array (trial 0) for misfit_amp_rate0 percentile."""
    if spec.sp_source == "das" and misfits.das_sp is not None:
        return misfits.das_sp[0]
    if spec.sp_source == "sta" and misfits.sta_sp is not None:
        return misfits.sta_sp[0]
    return None


def compute_representatives(spec: ModeSpec, cand_list: list,
                             misfits, obs, ctx: dict) -> None:
    """Stage 3: fm_mean misfit on trial-0 ray + top-N selection + STDR.

    Mutates ``cand_list`` in place.  After this call:
        - Each candidate has misfit_pol0/misfit_pol_ratio0/misfit_amp0/
          misfit_amp_rate0/stdr fields populated
        - List is sorted by (-accept_ratio, misfit_pol0)
        - List is truncated to top-N (5)
    """
    if not cand_list:
        return

    M_t, stk_t, dip_t, rak_t = _build_cand_tensors(cand_list, ctx["device"])

    pol_misfit_cand, pol_weighted = _compute_pol_misfit_cand(
        spec, stk_t, dip_t, rak_t, obs, ctx)
    if pol_misfit_cand is None:
        # Mode no longer feasible at representative stage (shouldn't happen
        # if filter_acceptable already passed, but defensive guard).
        cand_list.clear()
        return

    sp_misfit_cand = (_compute_sp_misfit_cand(spec, M_t, stk_t, dip_t, rak_t, obs, ctx)
                      if spec.is_joint else None)
    if spec.is_joint and sp_misfit_cand is None:
        cand_list.clear()
        return

    sp_grid = _select_sp_grid(spec, misfits)
    _overwrite_cand(cand_list, pol_misfit_cand, pol_weighted,
                    sp_misfit_cand=sp_misfit_cand, sp_grid=sp_grid)

    # STDR on top-N (post-truncation)
    # Joint modes: compute DAS and STA stdr separately, then blend
    # using das_pol_weight (same weight as the polarity misfit combination).
    das_args, sta_args = _stdr_args(spec, obs)
    if das_args is not None and sta_args is not None:
        # Joint mode: weighted combination
        cal_stdr(cand_list, *das_args)
        das_stdr = [c["stdr"] for c in cand_list]
        cal_stdr(cand_list, *sta_args)
        sta_stdr = [c["stdr"] for c in cand_list]
        w = ctx["das_pol_weight"]
        for i, c in enumerate(cand_list):
            c["stdr"] = w * das_stdr[i] + (1 - w) * sta_stdr[i]
    elif das_args is not None:
        cal_stdr(cand_list, *das_args)
    elif sta_args is not None:
        cal_stdr(cand_list, *sta_args)
    else:
        for c in cand_list:
            c["stdr"] = float("nan")
