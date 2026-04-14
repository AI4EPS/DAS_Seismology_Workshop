"""filter.py — Filter primitives + per-mode filter dispatch.

Replaces legacy search.py:
    - Old `search_and_cluster*` (NMC=1 path, tight threshold) — DELETED
    - Old `find_acceptable_*` (NMC>1 path, loose threshold with +0.5 buffer) — DELETED
    - Old `build_solution_and_cluster` — DELETED
    - Old `solve_mode` — DELETED

The unified replacement:
    - filter_pol_only / filter_joint:    one filter primitive,  per-trial tight
                                          threshold (same formula as old non-MC),
                                          union over trials.  NMC=1 is just a
                                          degenerate case of NMC>1.
    - build_solution:                    construct solution dict from union mask.
                                          misfit_pol/misfit_amp = trial-0 values
                                          (NOT min over trials — see docstring).
    - filter_acceptable:                 per-mode dispatch (uses ModeSpec) that
                                          assembles inputs and calls the primitives.
    - assemble_pol:                     shape-agnostic pol combination — used by
                                          BOTH filter_acceptable (2D) and
                                          representative._compute_pol_misfit_cand (1D).

This file is **torch-free** — pure numpy.  Safe in fork-safety chain.

NMC>1 numerical change:  the old `find_acceptable_*` had a `+ 0.5 * pol_weighted *
init_pol_error` buffer that the old `search_and_cluster*` did not.  The new code
uses the **tight** formula (no buffer) for both NMC=1 and NMC>1 — the user
accepted this change.
"""
from __future__ import annotations
import numpy as np

from dasfm.inversion.modes import ModeSpec


# Polarity misfit rate ceiling (50% = random chance level).  When the relaxation
# loop reaches this, polarity filter is essentially "off" — no further widening.
MAX_POL_THRESHOLD = 0.5

# Step size when widening the acceptance window during the relaxation loop.
# Each iteration adds RELAX_STEP_FRAC * init_*_error to the threshold.
RELAX_STEP_FRAC = 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 primitives — pure numerical helpers
# ═══════════════════════════════════════════════════════════════════════════

def filter_pol_only(pol_misfits: np.ndarray, pol_weighted: float,
                     init_pol_error: float) -> np.ndarray:
    """Polarity-only filter.  Per-trial threshold relaxation, union over trials.

    Each trial:
        thresh = min(pol_data) + pol_weighted * init_pol_error
    If < 20 mechanisms pass, widen by RELAX_STEP_FRAC * init_pol_error *
    pol_weighted per step, capped at MAX_POL_THRESHOLD.
    Final mask = OR of per-trial masks.

    Parameters
    ----------
    pol_misfits : np.ndarray (NMC, num_Mt)
        Polarity misfits per trial × per mechanism.  NMC=1 is shape (1, num_Mt).
    pol_weighted : float
        Total polarity weight (e.g. n_sta or |pol_obs|.sum()).
    init_pol_error : float
        Initial buffer as fraction of pol_weighted added to min misfit.

    Returns
    -------
    accept_count : np.ndarray (num_Mt,) int32
        Per-mechanism count of how many MC trials accepted it.
    """
    num_mt = pol_misfits.shape[1]
    accept_count = np.zeros(num_mt, dtype=np.int32)
    for pol_data in pol_misfits:                       # iterate trial axis
        thresh = np.min(pol_data) + pol_weighted * init_pol_error + 1e-5
        mask = pol_data <= thresh
        while mask.sum() < 20 and thresh / pol_weighted < MAX_POL_THRESHOLD:
            thresh += pol_weighted * init_pol_error * RELAX_STEP_FRAC
            mask = pol_data <= thresh
        accept_count += mask
    return accept_count


def filter_joint(pol_misfits: np.ndarray, amp_misfits: np.ndarray,
                  pol_weighted: float, init_pol_error: float,
                  init_amp_error_perc: float,
                  joint_pol_size: int = 60) -> np.ndarray:
    """Joint polarity + S/P filter.  Two-stage per-trial, union over trials.

    Stage 1: relax polarity threshold until >= joint_pol_size mechanisms pass.
    Stage 2: within the polarity pool, relax S/P threshold until >= 20 pass.

    Parameters
    ----------
    pol_misfits : np.ndarray (NMC, num_Mt)
    amp_misfits : np.ndarray (NMC, num_Mt)
    pol_weighted : float
    init_pol_error : float
    init_amp_error_perc : float
    joint_pol_size : int
        Minimum number of polarity-acceptable mechanisms before applying
        the amplitude filter (default 60).
    """
    num_mt = pol_misfits.shape[1]
    n_trials = min(pol_misfits.shape[0], amp_misfits.shape[0])
    accept_count = np.zeros(num_mt, dtype=np.int32)

    for imc in range(n_trials):
        pol_data = pol_misfits[imc]
        amp_data = amp_misfits[imc]

        # ── Stage 1: build polarity pool ──
        pol_thresh = np.min(pol_data) + pol_weighted * init_pol_error + 1e-5
        while np.sum(pol_data <= pol_thresh) < joint_pol_size:
            pol_thresh += pol_weighted * init_pol_error * RELAX_STEP_FRAC
        pol_mask = pol_data <= pol_thresh

        # ── Stage 2: filter by amplitude within polarity pool ──
        amp_in_pool = np.sort(amp_data[pol_mask])
        n_pool = len(amp_in_pool)
        amp_k = max(1, min(int(round(n_pool * init_amp_error_perc)), n_pool - 1))
        amp_thresh = amp_in_pool[amp_k]
        mask = pol_mask & (amp_data <= amp_thresh)
        while mask.sum() < 20:
            amp_k = min(amp_k + max(1, int(n_pool * init_amp_error_perc * RELAX_STEP_FRAC)), n_pool - 1)
            amp_thresh = amp_in_pool[amp_k]
            mask = pol_mask & (amp_data <= amp_thresh)
            if amp_k >= n_pool - 1:
                break

        accept_count += mask
    return accept_count


def build_solution(accept_count: np.ndarray, pol_misfits: np.ndarray,
                    amp_misfits: np.ndarray | None, pol_weighted: float,
                    stk: np.ndarray, dip: np.ndarray, rak: np.ndarray,
                    n_measure: int) -> dict | None:
    """Build solution dict from accept_count + **trial-0 misfit values**.

    The acceptance set (accept_count > 0) is computed from per-trial filters.
    accept_count records how many MC trials accepted each mechanism — used
    downstream by cluster.py for weighted accept_ratio.

    Per-mechanism misfit fields come from **trial 0 only** (unperturbed
    reference ray, same trial used for representative misfit recompute).

    Parameters
    ----------
    accept_count : np.ndarray (num_Mt,) int32
        Per-mechanism count of how many MC trials accepted it.
    pol_misfits : np.ndarray (NMC, num_Mt)
    amp_misfits : np.ndarray (NMC, num_Mt) or None
    pol_weighted : float
    stk, dip, rak : np.ndarray (num_Mt,) — full SDR grid
    n_measure : int — number of observations (n_ch or n_sta)

    Returns
    -------
    dict or None (when no mechanism was accepted by any trial).
    """
    igood = np.where(accept_count > 0)[0]
    if len(igood) == 0:
        return None

    # Trial 0 = unperturbed reference ray (guaranteed by step1 loader,
    # same trial used for representative misfit recompute).
    pol0 = pol_misfits[0]
    amp0 = (amp_misfits[0] if amp_misfits is not None
            else np.full_like(pol0, np.nan))

    return {
        "stk": stk.ravel()[igood].copy(),
        "dip": dip.ravel()[igood].copy(),
        "rak": rak.ravel()[igood].copy(),
        "mask": igood.copy(),
        "accept_count": accept_count[igood].copy(),
        "misfit_pol": pol0[igood].copy(),
        "misfit_pol_all": pol0.copy(),
        "misfit_pol_ratio": (pol0[igood] / pol_weighted).copy(),
        "misfit_amp": amp0[igood].copy(),
        "misfit_amp_all": amp0.copy(),
        "nsol": int(len(igood)),
        "nmeasure": int(n_measure),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Shape-agnostic pol assembly (used by BOTH filter and representative)
# ═══════════════════════════════════════════════════════════════════════════

def assemble_pol(spec: ModeSpec, das_pol, sta_pol, n_sta: int,
                   pol_obs_sum, das_pol_weight: float):
    """Assemble combined pol misfit + pol_weighted from das/sta sources.

    **Shape-agnostic**: das_pol/sta_pol can be 1D ``(N_cand,)`` (representative
    stage) or 2D ``(NMC, num_Mt)`` (filter stage).  The numpy broadcasting in
    ``das + w * sta`` works correctly for both shapes — same code, no shape
    branching.

    **Strict no-degrade policy**: if a mode requires both DAS and STA pol and
    STA is unavailable (n_sta == 0 or sta_pol is None), the mode is SKIPPED
    (return None).  No silent degradation to "DAS pol only".  This unifies the
    "missing data → skip" semantics across all 7 modes.

    Parameters
    ----------
    spec : ModeSpec
    das_pol : np.ndarray or None    # 1D or 2D
    sta_pol : np.ndarray or None    # same shape as das_pol
    n_sta : int                     # 0 means STA unavailable
    pol_obs_sum : float or None     # |pol_obs|.sum() — None if no DAS pol obs
    das_pol_weight : float

    Returns
    -------
    (combined, pol_weighted) or (None, None) if mode unfeasible.
    """
    # Joint mode: STA_pol_DAS_pol or STA_pol_DAS_pol_sp — needs BOTH sources.
    # Strict skip if either is missing.
    if spec.has_das_pol and spec.has_sta_pol:
        if pol_obs_sum is None or das_pol is None:
            return None, None
        if n_sta == 0 or sta_pol is None:
            return None, None     # skip, NOT degrade to DAS-only
        w = pol_obs_sum * (1 - das_pol_weight) / (n_sta * das_pol_weight)
        return das_pol + w * sta_pol, pol_obs_sum + n_sta * w

    # DAS only: DAS_pol or DAS_pol_sp
    if spec.has_das_pol:
        if pol_obs_sum is None or das_pol is None:
            return None, None
        return das_pol, pol_obs_sum

    # STA only: STA_pol, STA_pol_sp, STA_pol_DAS_sp
    if spec.has_sta_pol:
        if n_sta == 0 or sta_pol is None:
            return None, None
        return sta_pol, float(n_sta)

    return None, None


def _select_amp_stack(spec: ModeSpec, misfits, n_sta_sp: int):
    """Pick the right sp misfit stack from misfits according to spec.sp_source.

    Returns the (NMC, num_Mt) stack or None when the mode has no S/P source
    or the source is unavailable for this event.
    """
    if spec.sp_source == "das":
        return misfits.das_sp     # may be None
    if spec.sp_source == "sta":
        return misfits.sta_sp if n_sta_sp > 0 else None
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 dispatch — filter_acceptable (uses ModeSpec)
# ═══════════════════════════════════════════════════════════════════════════
# NOTE: filter_acceptable lives in this file because it's the per-mode entry
# point to the filter primitives above.  It's the only public function in this
# module besides the primitives.  All the dispatch logic is shape-agnostic
# via assemble_pol — no per-mode if-blocks.

def filter_acceptable(spec, misfits, obs, ctx):
    """Stage 1: filter — returns sol dict or None (mode skipped).

    Builds (pol_stack, amp_stack, pol_weighted) from misfits according to spec,
    then calls filter_pol_only / filter_joint, then build_solution.

    Parameters
    ----------
    spec : ModeSpec
    misfits : ForwardMisfits   (from forward.py)
    obs : EventObs             (from forward.py)
    ctx : dict                 — must contain init_pol_error, init_amp_error_perc,
                                  joint_pol_size, das_pol_weight,
                                  stk_g/dip_g/rak_g (full SDR grid as numpy)

    Returns
    -------
    sol : dict or None
    """
    pol_obs_sum = (float(np.abs(obs.pol_obs_cpu).sum())
                   if obs.pol_obs_cpu is not None else None)

    # Truncate joint mode to common n_trials (NMC_DAS may differ from NMC_STA).
    das_pol_input = misfits.das_pol
    sta_pol_input = misfits.sta_pol
    if (spec.has_das_pol and spec.has_sta_pol
            and das_pol_input is not None and sta_pol_input is not None):
        n_trials = min(das_pol_input.shape[0], sta_pol_input.shape[0])
        das_pol_input = das_pol_input[:n_trials]
        sta_pol_input = sta_pol_input[:n_trials]

    pol_stack, pol_weighted = assemble_pol(
        spec, das_pol_input, sta_pol_input,
        n_sta=obs.n_sta, pol_obs_sum=pol_obs_sum,
        das_pol_weight=ctx["das_pol_weight"],
    )
    if pol_stack is None:
        return None

    amp_stack = _select_amp_stack(spec, misfits, obs.n_sta_sp)
    if spec.is_joint and amp_stack is None:
        return None     # mode requires sp but data missing

    if amp_stack is not None:
        union_mask = filter_joint(
            pol_stack, amp_stack, pol_weighted,
            ctx["init_pol_error"], ctx["init_amp_error_perc"],
            ctx["joint_pol_size"],
        )
    else:
        union_mask = filter_pol_only(pol_stack, pol_weighted, ctx["init_pol_error"])

    n_measure = ctx["n_ch"] if spec.has_das_pol else obs.n_sta
    return build_solution(
        union_mask, pol_stack, amp_stack, pol_weighted,
        ctx["stk_g"], ctx["dip_g"], ctx["rak_g"], n_measure,
    )
