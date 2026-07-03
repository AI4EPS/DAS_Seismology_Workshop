"""grid_forward — Per-event observation prep + grid-stage forward.

Provides:
    EventObs            dataclass: per-event observed data + trial-0 ray snapshot
    ForwardMisfits      dataclass: per-event grid-stage misfit stacks
    prepare_observations(iev, ctx) -> EventObs
    run_grid_forward(obs, ctx)    -> ForwardMisfits

The 4 internal forward helpers:
    _forward_das_pol_all_trials(obs, ctx) -> (NMC_DAS, num_Mt) stack
    _forward_das_sp_all_trials(obs, ctx)  -> (sp_stack, corr_stack)
    _forward_sta_pol_all_trials(obs, ctx) -> (NMC_STA, num_Mt) stack
    _forward_sta_sp_all_trials(obs, ctx)  -> (NMC_STA, num_Mt) stack

trial-0 snapshot is done **once** in prepare_observations — replaces the old
``if imc == 0:`` hack scattered through the legacy event_loop.

This file imports torch at the top — must be **lazy-imported** by callers in the
fork-safety chain (step3_invert_serial.py, pipeline.py).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch

from dasfm.inversion.forward_model import (
    compute_das_polarity_misfit_torch,
    compute_das_forward,
    sp_misfit_norm_np,
)
from dasfm.inversion.moment_tensor import (
    compute_sta_pol_misfit_torch,
    compute_sta_sp_theo_torch,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EventObs:
    """Per-event observed data + trial-0 unperturbed ray params.

    All fields are populated by ``prepare_observations(iev, ctx)``.

    Trial-0 ray fields (``das_tak_t0``, ``sta_takeoff_t0``, etc.) are the
    unperturbed reference ray (snapshotted once here, used by both the
    forward step and the representative-misfit recompute).
    """
    eid: str
    iev: int
    # DAS observations
    pol_obs:        torch.Tensor | None = None     # (n_ch,) on device
    sp_obs_das:     torch.Tensor | None = None     # (n_ch,) on device, decay-corrected
    pol_obs_cpu:    np.ndarray | None = None       # numpy copy for filter (CPU side)
    # STA observations
    sta_pol_i:      np.ndarray | None = None       # (n_sta,) ±1
    sta_sp_obs_i:   np.ndarray | None = None       # (n_sta_sp,)
    n_sta: int = 0
    n_sta_sp: int = 0
    # Trial-0 unperturbed ray params (SKHASH-style for representative misfit + STDR)
    das_tak_t0:     np.ndarray | None = None       # (n_ch,) deg
    das_az_t0:      np.ndarray | None = None       # (n_ch,) deg
    sta_takeoff_t0:     np.ndarray | None = None       # (n_sta,) deg
    sta_az_t0:      np.ndarray | None = None       # (n_sta,) deg
    sta_sp_takeoff_t0:  np.ndarray | None = None       # (n_sta_sp,) deg
    sta_sp_az_t0:   np.ndarray | None = None       # (n_sta_sp,) deg
    # STA column indices into the full sta_mc_takeoff/azimuth array (None if no MC)
    sta_pol_col_idx: np.ndarray | None = None      # (n_sta,) int64
    sta_sp_col_idx:  np.ndarray | None = None      # (n_sta_sp,) int64


@dataclass
class ForwardMisfits:
    """Per-event grid-stage forward misfits, stacked 2D arrays.

    Each misfit array has shape ``(NMC_X, num_Mt)``.  NMC=1 case is ``(1, num_Mt)``.
    Row 0 is always the unperturbed reference trial.
    """
    das_pol:  np.ndarray | None = None    # (NMC_DAS, num_Mt) or None
    das_sp:   np.ndarray | None = None    # (NMC_DAS, num_Mt) or None
    das_corr: np.ndarray | None = None    # (NMC_DAS, num_Mt) — DAS sp mean-alignment factor
    sta_pol:  np.ndarray | None = None    # (NMC_STA, num_Mt) or None
    sta_sp:   np.ndarray | None = None    # (NMC_STA, num_Mt) or None


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 0: prepare_observations
# ═══════════════════════════════════════════════════════════════════════════

def prepare_observations(iev: int, ctx: dict) -> EventObs:
    """Build EventObs for one event.  Trial-0 ray snapshot done here.

    Reads from ctx:
        Pkic, pol_valid, sp_ratios                 (DAS obs)
        weighted_polarity, sp_decay_k, sp_decay_c  (parameters)
        das_mc_takeoff_deg, das_mc_azimuth_deg, das_mc_distance  (DAS ray params)
        sta_pol_data, sta_sp_data                  (STA obs dicts)
        sta_mc_takeoff, sta_mc_azimuth, sta_name_to_col  (STA ray params)
        HAS_DAS_POL, HAS_DAS_SP, HAS_DAS_SRC, HAS_STA_MC
        n_ch, das_event_ids, device
    """
    eid = ctx["das_event_ids"][iev]
    obs = EventObs(eid=eid, iev=iev)
    device = ctx["device"]
    n_ch = ctx["n_ch"]

    # ── DAS polarity obs ──
    HAS_DAS_POL = ctx.get("HAS_DAS_POL", False)
    if HAS_DAS_POL and n_ch > 0:
        pol_obs = torch.tensor(ctx["Pkic"][:, iev], dtype=torch.float32, device=device)
        if not ctx.get("weighted_polarity", False):
            pol_obs = torch.sign(pol_obs)
        pol_valid = ctx.get("pol_valid")
        if pol_valid is not None:
            pol_obs[~torch.tensor(pol_valid[:, iev], dtype=torch.bool, device=device)] = 0.0
        obs.pol_obs = pol_obs
        obs.pol_obs_cpu = pol_obs.cpu().numpy()

    # ── DAS S/P obs (with decay corrections) ──
    HAS_DAS_SP = ctx.get("HAS_DAS_SP", False)
    if HAS_DAS_SP and n_ch > 0:
        sp_obs_das = torch.tensor(ctx["sp_ratios"][:, iev], dtype=torch.float32, device=device)
        if ctx.get("sp_decay_k") is not None and ctx.get("HAS_DAS_SRC", False):
            dist_km = torch.tensor(ctx["das_mc_distance"][0][iev],
                                    dtype=torch.float32, device=device)
            sp_obs_das = sp_obs_das + ctx["sp_decay_k"] * dist_km
        if ctx.get("sp_decay_c") is not None:
            sp_obs_das = sp_obs_das + ctx["sp_decay_c"]
        obs.sp_obs_das = sp_obs_das

    # ── DAS trial-0 ray snapshot ──
    if ctx.get("HAS_DAS_SRC", False) and n_ch > 0:
        obs.das_tak_t0 = np.asarray(ctx["das_mc_takeoff_deg"][0][iev],
                                     dtype=np.float32).copy()
        obs.das_az_t0 = np.asarray(ctx["das_mc_azimuth_deg"][0][iev],
                                    dtype=np.float32).copy()

    # ── STA polarity obs + trial-0 ray ──
    sta_pol_data = ctx.get("sta_pol_data")
    if sta_pol_data is not None and eid in sta_pol_data["eid_to_idx"]:
        sidx = sta_pol_data["eid_to_idx"][eid]
        sta_pol_i = sta_pol_data["polarities"][sidx]
        obs.n_sta = len(sta_pol_i)

        if ctx.get("HAS_STA_MC", False):
            # MC mode: look up column indices in the full sta_mc_takeoff array
            sta_name_to_col = ctx["sta_name_to_col"]
            stations = sta_pol_data["stations"][sidx]
            col_idx = np.array(
                [sta_name_to_col[n] for n in stations if n in sta_name_to_col],
                dtype=np.int64,
            )
            if len(col_idx) != obs.n_sta:
                # Some stations not in sta_name_to_col → truncate
                obs.n_sta = len(col_idx)
                sta_pol_i = sta_pol_i[:obs.n_sta]
            obs.sta_pol_col_idx = col_idx
            obs.sta_takeoff_t0 = np.degrees(
                ctx["sta_mc_takeoff"][0][iev, col_idx]).astype(np.float32)
            obs.sta_az_t0 = np.degrees(
                ctx["sta_mc_azimuth"][0][iev, col_idx]).astype(np.float32)
        else:
            # No MC: use the per-event takeoffs/azimuths stored in sta_pol_data
            obs.sta_takeoff_t0 = np.asarray(sta_pol_data["takeoffs"][sidx], dtype=np.float32)
            obs.sta_az_t0 = np.asarray(sta_pol_data["azimuths"][sidx], dtype=np.float32)
        obs.sta_pol_i = np.asarray(sta_pol_i)

    # ── STA S/P obs + trial-0 ray ──
    sta_sp_data = ctx.get("sta_sp_data")
    if sta_sp_data is not None and eid in sta_sp_data["eid_to_idx"]:
        sp_sidx = sta_sp_data["eid_to_idx"][eid]
        sta_sp_obs_i = sta_sp_data["ratios"][sp_sidx]
        obs.n_sta_sp = len(sta_sp_obs_i)

        if ctx.get("HAS_STA_MC", False):
            sta_name_to_col = ctx["sta_name_to_col"]
            stations = sta_sp_data["stations"][sp_sidx]
            col_idx = np.array(
                [sta_name_to_col[n] for n in stations if n in sta_name_to_col],
                dtype=np.int64,
            )
            if len(col_idx) != obs.n_sta_sp:
                obs.n_sta_sp = len(col_idx)
                sta_sp_obs_i = sta_sp_obs_i[:obs.n_sta_sp]
            obs.sta_sp_col_idx = col_idx
            obs.sta_sp_takeoff_t0 = np.degrees(
                ctx["sta_mc_takeoff"][0][iev, col_idx]).astype(np.float32)
            obs.sta_sp_az_t0 = np.degrees(
                ctx["sta_mc_azimuth"][0][iev, col_idx]).astype(np.float32)
        else:
            obs.sta_sp_takeoff_t0 = np.asarray(sta_sp_data["takeoffs"][sp_sidx], dtype=np.float32)
            obs.sta_sp_az_t0 = np.asarray(sta_sp_data["azimuths"][sp_sidx], dtype=np.float32)
        obs.sta_sp_obs_i = np.asarray(sta_sp_obs_i, dtype=np.float32)

    return obs


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1: run_grid_forward
# ═══════════════════════════════════════════════════════════════════════════

def run_grid_forward(obs: EventObs, ctx: dict) -> ForwardMisfits:
    """Run forward for all 4 measurement types × all MC trials.

    DAS sp returns BOTH the misfit stack AND the corr_factor stack
    (corr_factor is the per-trial mean alignment offset).
    """
    misfits = ForwardMisfits()

    if ctx.get("HAS_DAS_SRC", False) and ctx["n_ch"] > 0:
        if obs.pol_obs is not None:
            misfits.das_pol = _forward_das_pol_all_trials(obs, ctx)
        if obs.sp_obs_das is not None:
            misfits.das_sp, misfits.das_corr = _forward_das_sp_all_trials(obs, ctx)

    if obs.n_sta > 0:
        misfits.sta_pol = _forward_sta_pol_all_trials(obs, ctx)
    if obs.n_sta_sp > 0:
        misfits.sta_sp = _forward_sta_sp_all_trials(obs, ctx)

    return misfits


# ═══════════════════════════════════════════════════════════════════════════
#  Internal: per-measurement forward loops (NMC trials → stacked 2D array)
# ═══════════════════════════════════════════════════════════════════════════

def _forward_das_pol_all_trials(obs: EventObs, ctx: dict) -> np.ndarray:
    """DAS polarity forward, all NMC_DAS trials. Returns (NMC_DAS, num_Mt) stack.

    Uses Aki-Richards particle-velocity P-radiation (matches Pkic convention).
    """
    NMC_DAS = ctx["NMC_DAS"]
    num_Mt = ctx["num_Mt"]
    iev = obs.iev
    rows = np.empty((NMC_DAS, num_Mt), dtype=np.float32)
    for imc in range(NMC_DAS):
        m = compute_das_polarity_misfit_torch(
            ctx["stk_t"], ctx["dip_t"], ctx["rak_t"],
            ctx["das_mc_takeoff_deg"][imc][iev],
            ctx["das_mc_azimuth_deg"][imc][iev],
            obs.pol_obs,
        )
        rows[imc] = m.cpu().numpy()
    return rows


def _forward_das_sp_all_trials(obs: EventObs, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """DAS S/P forward, all NMC_DAS trials.

    Returns
    -------
    sp_stack : (NMC_DAS, num_Mt)  — sp misfit per trial × per mech
    corr_stack : (NMC_DAS, num_Mt) — mean alignment factor per trial × per mech
        (squeezed from compute_das_forward's (num_Mt, 1) output).
        Row 0 is the only one consumed downstream (corr_factor saved to result hdf5).
    """
    NMC_DAS = ctx["NMC_DAS"]
    num_Mt = ctx["num_Mt"]
    iev = obs.iev
    sp_rows = np.empty((NMC_DAS, num_Mt), dtype=np.float32)
    corr_rows = np.full((NMC_DAS, num_Mt), np.nan, dtype=np.float32)
    has_corr = False
    for imc in range(NMC_DAS):
        m_sp, corr = compute_das_forward(
            ctx["M_g_t"],
            ctx["das_mc_takeoff_deg"][imc][iev],
            ctx["das_mc_azimuth_deg"][imc][iev],
            sp_obs_das=obs.sp_obs_das,
            do_mean_alignment=ctx["do_mean_alignment"],
            device=ctx["device"],
            sp_norm=ctx["sp_norm"],
            cauchy_c=ctx.get("cauchy_c"),
            vp_vs_ratio=ctx.get("vp_vs_ratio", 1.7),
        )
        sp_rows[imc] = m_sp.cpu().numpy()
        if corr is not None:
            corr_rows[imc] = corr.squeeze(-1).cpu().numpy()    # (num_Mt, 1) → (num_Mt,)
            has_corr = True
    return sp_rows, (corr_rows if has_corr else None)


def _forward_sta_pol_all_trials(obs: EventObs, ctx: dict) -> np.ndarray:
    """STA polarity forward, all NMC_STA trials. Returns (NMC_STA, num_Mt) stack.

    No ``imc == 0`` snapshot — trial-0 ray was already captured in
    ``prepare_observations``.

    Internally branches on HAS_STA_MC (no-MC vs MC have different ray data
    layouts) — Level 1 of the MC unification.  External callers see a uniform
    (NMC_STA, num_Mt) output.
    """
    NMC_STA = ctx["NMC_STA"]
    num_Mt = ctx["num_Mt"]
    iev = obs.iev
    device = ctx["device"]
    pol_i_t = torch.tensor(obs.sta_pol_i, dtype=torch.float32, device=device)
    rows = np.empty((NMC_STA, num_Mt), dtype=np.float32)

    HAS_STA_MC = ctx.get("HAS_STA_MC", False)
    if HAS_STA_MC:
        col_idx = obs.sta_pol_col_idx
        for imc in range(NMC_STA):
            takeoff_deg = np.degrees(ctx["sta_mc_takeoff"][imc][iev, col_idx]).astype(np.float32)
            az_deg = np.degrees(ctx["sta_mc_azimuth"][imc][iev, col_idx]).astype(np.float32)
            takeoff_t = torch.tensor(takeoff_deg, dtype=torch.float32, device=device)
            az_t = torch.tensor(az_deg, dtype=torch.float32, device=device)
            misfit = compute_sta_pol_misfit_torch(
                ctx["stk_t"], ctx["dip_t"], ctx["rak_t"], takeoff_t, az_t, pol_i_t)
            rows[imc] = misfit.cpu().numpy()
    else:
        # NMC_STA == 1 case:  use the trial-0 snapshot directly.
        takeoff_t = torch.tensor(obs.sta_takeoff_t0, dtype=torch.float32, device=device)
        az_t = torch.tensor(obs.sta_az_t0, dtype=torch.float32, device=device)
        misfit = compute_sta_pol_misfit_torch(
            ctx["stk_t"], ctx["dip_t"], ctx["rak_t"], takeoff_t, az_t, pol_i_t)
        rows[0] = misfit.cpu().numpy()
    return rows


def _forward_sta_sp_all_trials(obs: EventObs, ctx: dict) -> np.ndarray:
    """STA S/P forward, all NMC_STA trials. Returns (NMC_STA, num_Mt) stack.

    STA SP misfit aggregates with sp_misfit_norm_np (L1/L2/cauchy).
    No mean alignment for STA SP (unlike DAS SP).
    """
    NMC_STA = ctx["NMC_STA"]
    num_Mt = ctx["num_Mt"]
    iev = obs.iev
    device = ctx["device"]
    sp_obs_np = obs.sta_sp_obs_i
    rows = np.empty((NMC_STA, num_Mt), dtype=np.float32)

    HAS_STA_MC = ctx.get("HAS_STA_MC", False)
    if HAS_STA_MC:
        col_idx = obs.sta_sp_col_idx
        for imc in range(NMC_STA):
            takeoff_deg = np.degrees(ctx["sta_mc_takeoff"][imc][iev, col_idx]).astype(np.float32)
            az_deg = np.degrees(ctx["sta_mc_azimuth"][imc][iev, col_idx]).astype(np.float32)
            takeoff_t = torch.tensor(takeoff_deg, dtype=torch.float32, device=device)
            az_t = torch.tensor(az_deg, dtype=torch.float32, device=device)
            sp_theo = compute_sta_sp_theo_torch(
                ctx["stk_t"], ctx["dip_t"], ctx["rak_t"], takeoff_t, az_t).cpu().numpy()
            rows[imc] = sp_misfit_norm_np(
                sp_theo - sp_obs_np, ctx["sp_norm"], cauchy_c=ctx.get("cauchy_c"))
    else:
        takeoff_t = torch.tensor(obs.sta_sp_takeoff_t0, dtype=torch.float32, device=device)
        az_t = torch.tensor(obs.sta_sp_az_t0, dtype=torch.float32, device=device)
        sp_theo = compute_sta_sp_theo_torch(
            ctx["stk_t"], ctx["dip_t"], ctx["rak_t"], takeoff_t, az_t).cpu().numpy()
        rows[0] = sp_misfit_norm_np(
            sp_theo - sp_obs_np, ctx["sp_norm"], cauchy_c=ctx.get("cauchy_c"))
    return rows
