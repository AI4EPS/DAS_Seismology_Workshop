"""pipeline.py — Per-event inversion orchestration.

Stitches together the 4 stages for one event:

    obs = prepare_observations(iev, ctx)              ← stage 0 (forward.py)
    misfits = run_grid_forward(obs, ctx)              ← stage 1 (forward.py)
    for mode in inv_set:
        sol = filter_acceptable(spec, misfits, ...)   ← stage 2 (filter.py)
        cands = cluster_solution_nooverlap(sol)       ← stage 3 (cluster.py)
        compute_representatives(spec, cands, ...)     ← stage 4 (representative.py)
    save_event_results(...)

This file is **only orchestration** — no numerical logic.  Any numerical
helper goes in forward.py / filter.py / representative.py / cluster.py.

Import discipline (fork-safety):
    - Top-level imports: modes, filter, cluster (all torch-free)
    - Lazy imports inside process_event: forward, representative (both pull torch)
    - Lazy import inside _save_event_results: result_io (pulls torch until
      Phase 6.5 fix)

This guarantees pipeline.py module load doesn't trigger torch — main process
can call step3_invert_serial._load_data() without paying for torch import.
"""
from __future__ import annotations
from pathlib import Path
import time as _time

from tqdm import tqdm

from dasfm.inversion.modes import MODE_REGISTRY
from dasfm.inversion.filter import filter_acceptable
from dasfm.inversion.cluster import cluster_solution_nooverlap
from dasfm.utils.step_utils import build_obs


# ═══════════════════════════════════════════════════════════════════════════
#  Per-event orchestration
# ═══════════════════════════════════════════════════════════════════════════

def process_event(iev: int, ctx: dict) -> None:
    """Process one event end-to-end: forward → filter → cluster → representative → save.

    All numerical work happens in forward.py / filter.py / cluster.py /
    representative.py — pipeline.py just glues them together.
    """
    # Lazy imports — these pull torch but only at first call (post-fork in
    # multi-cpu mode, or in main process for serial/multi-gpu).
    from dasfm.inversion.grid_forward import prepare_observations, run_grid_forward
    from dasfm.inversion.representative import compute_representatives

    eid = ctx["das_event_ids"][iev]
    n_ev = ctx["n_ev"]
    logger = ctx.get("logger")

    _t_ev = _time.perf_counter()

    obs = prepare_observations(iev, ctx)

    _t0 = _time.perf_counter()
    misfits = run_grid_forward(obs, ctx)
    _t_fwd = _time.perf_counter() - _t0

    results: dict = {}
    _t0 = _time.perf_counter()
    for mode in ctx["inv_set"]:
        spec = MODE_REGISTRY[mode]
        sol = filter_acceptable(spec, misfits, obs, ctx)
        if sol is None:
            results[mode] = _empty_result(mode, spec, obs, ctx)
            continue
        cand_list = cluster_solution_nooverlap(sol, logger=logger)
        compute_representatives(spec, cand_list, misfits, obs, ctx)
        results[mode] = _build_result(mode, spec, sol, cand_list, misfits, obs, ctx)
    _t_solve = _time.perf_counter() - _t0

    _save_event_results(iev, results, ctx)

    _t_total = _time.perf_counter() - _t_ev
    if logger is not None:
        logger.log(f"  [{iev+1}/{n_ev}] {eid}  total={_t_total:.1f}s  "
                   f"fwd={_t_fwd:.1f}s  solve={_t_solve:.2f}s")


def process_events(ctx: dict, event_indices=None) -> None:
    """Top-level event loop.  Replaces legacy event_loop().

    Parameters
    ----------
    ctx : dict
        Per-runner context from _build_torch_state(data, device).
    event_indices : list[int] or None
        Subset of events to process.  None = all events.
    """
    if event_indices is None:
        event_indices = list(range(ctx["n_ev"]))

    pbar = tqdm(total=len(event_indices), desc="events", unit="ev",
                leave=True, disable=ctx.get("_quiet", False))
    for iev in event_indices:
        process_event(iev, ctx)
        pbar.update(1)
    pbar.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Result assembly + save
# ═══════════════════════════════════════════════════════════════════════════

def _build_result(mode: str, spec, sol, cand_list, misfits, obs, ctx) -> dict:
    """Assemble the per-mode result dict for save_inversion_result.

    Returns dict with keys: sol, cand, corr_factor, obs, meta.
    """
    n_ch = ctx["n_ch"]

    # corr_factor: only meaningful for DAS sp modes.  Take row 0 of misfits.das_corr
    # subsetted to the acceptable mechanisms (sol['mask']).
    corr_factor = None
    if spec.sp_source == "das" and misfits.das_corr is not None and sol is not None:
        igood = sol["mask"]
        if len(igood) > 0:
            corr_factor = misfits.das_corr[0][igood]

    # obs dict for plotting (per-mode subset of observations)
    obs_dict = build_obs(
        mode, obs.iev, n_ch,
        ctx.get("Pkic"),
        ctx.get("das_mc_takeoff_deg"),
        ctx.get("das_mc_azimuth_deg"),
        ctx.get("sp_ratios"),
        obs.sta_takeoff_t0, obs.sta_az_t0, obs.sta_pol_i,
        obs.sta_sp_takeoff_t0, obs.sta_sp_az_t0, obs.sta_sp_obs_i,
        ctx.get("HAS_DAS_SRC", False),
    )

    return {
        "sol": sol,
        "cand": cand_list,
        "corr_factor": corr_factor,
        "obs": obs_dict,
        "meta": _build_meta(spec, obs, n_ch, ctx["NMC"]),
    }


def _empty_result(mode: str, spec, obs, ctx) -> dict:
    """Result for a mode that was skipped (sol returned None from filter)."""
    n_ch = ctx["n_ch"]
    obs_dict = build_obs(
        mode, obs.iev, n_ch,
        ctx.get("Pkic"),
        ctx.get("das_mc_takeoff_deg"),
        ctx.get("das_mc_azimuth_deg"),
        ctx.get("sp_ratios"),
        obs.sta_takeoff_t0, obs.sta_az_t0, obs.sta_pol_i,
        obs.sta_sp_takeoff_t0, obs.sta_sp_az_t0, obs.sta_sp_obs_i,
        ctx.get("HAS_DAS_SRC", False),
    )
    return {
        "sol": [],
        "cand": [],
        "corr_factor": None,
        "obs": obs_dict,
        "meta": dict(num_das_pol=0, num_sta_pol=0, num_das_sp=0, num_sta_sp=0,
                      nmc=ctx["NMC"], skipped=True),
    }


def _build_meta(spec, obs, n_ch: int, NMC: int) -> dict:
    """Per-mode counts written as hdf5 attrs."""
    return dict(
        num_das_pol = n_ch if spec.has_das_pol else 0,
        num_sta_pol = obs.n_sta if spec.has_sta_pol else 0,
        num_das_sp  = n_ch if spec.sp_source == "das" else 0,
        num_sta_sp  = obs.n_sta_sp if spec.sp_source == "sta" else 0,
        nmc         = NMC,
    )


def _save_event_results(iev: int, results: dict, ctx: dict) -> None:
    """Write one .h5 per mode for this event."""
    # Lazy import — result_io imports torch (until Phase 6.5 fix).
    from dasfm.io.result_io import save_inversion_result

    eid = ctx["das_event_ids"][iev]
    ev_sol_dir = Path(ctx["RESULT_ROOT"]) / "inv_sol" / eid
    ev_sol_dir.mkdir(parents=True, exist_ok=True)

    for mode, r in results.items():
        save_inversion_result(
            ev_sol_dir / f"{mode}.h5",
            r["sol"], r["cand"],
            corr_factor=r.get("corr_factor"),
            obs=r.get("obs"),
            **r["meta"],
        )
