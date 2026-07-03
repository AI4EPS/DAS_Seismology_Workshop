"""mccc_context — Shared setup, context dataclass, and MCCC driver.

Called by ``step2b_polarity_serial`` / ``_cpus`` / ``_gpus``.  Houses:

* :class:`Step2bContext`      — frozen dataclass holding all shared state
* :func:`setup_context`       — read fft cache metadata, sparse rankings, memory budget
* :func:`alloc_dense_ckij`    — allocate Ckij/Skij with diagonal = 1
* :func:`run_with_iteration`  — sparse-aware MCCC driver (callback strategy)

The dense ``Ckij/Skij`` matrices are allocated **once** by ``run_with_iteration``
and stream-written by the backend's ``run_mccc_fn`` callback — no Python list
intermediate.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil

from dasfm.utils.step_utils import Logger, resolve_path, resolve_device
from dasfm.io.das_fft import load_das_fft_meta
from dasfm.picking.sparse_pairs import (
    precompute_pair_rankings, select_from_rankings, expand_pairs,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Step2bContext
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Step2bContext:
    """All shared state for the step2b pipeline.  Built once by setup_context()."""

    # ── paths ─────────────────────────────────────────────────────────────
    project_dir:  Path
    fft_dir:      Path
    pol_out_path: Path
    mccc_cache_path: Optional[Path]   # save Ckij/Skij cache here (None = skip)
    das_geo_path: Path
    sta4mccc_ref: Optional[Path]
    sta_polarity: Optional[Path]
    sta_geo:      Optional[Path]
    log_dir:      Path
    fig_dir:      Path
    cal_csv_path: Path     # auto-computed cal CSV target (may not be written)

    # ── discovered metadata ───────────────────────────────────────────────
    event_ids:    list
    n_ev:         int
    n_ch:         int
    nfast:        int
    dt:           float
    has_lr:       bool
    das_geo_df:   object   # pandas DataFrame
    channel_ids:  np.ndarray   # int32

    # ── mccc params ───────────────────────────────────────────────────────
    mccc_max_lag_sec:        float
    mccc_maxwin:             int
    mccc_damp:               float
    mccc_max_shift:          int
    polarity_smooth_window:  int
    polarity_method:         str
    cal_min_picks:           int

    # ── sparse params ─────────────────────────────────────────────────────
    sparse:              bool
    k_neighbors:         int
    top_xcorr_frac:      float
    n_remote:            int
    xcorr_subsample:     int
    stability_threshold: float
    rankings:            Optional[dict]   # precompute_pair_rankings result
    pairs_A:             Optional[set]
    pairs_B:             Optional[set]    # what MCCC computes first

    # ── device + parallelism ──────────────────────────────────────────────
    device:           Union[str, list]   # canonical: "cpu" | "cuda:N" | ["cuda:0", ...]
    use_gpu:          bool
    num_gpu:          int
    gpu_devices:      list
    num_cpu_workers:  int

    # ── derived flag ──────────────────────────────────────────────────────
    use_lr:      bool   # = (polarity_method == "hilbert" and has_lr)

    # ── misc ──────────────────────────────────────────────────────────────
    show_plots: bool
    t0:         float
    logger:     object
    mode_label: str


# ═══════════════════════════════════════════════════════════════════════════
#  setup_context — build the Step2bContext from user kwargs
# ═══════════════════════════════════════════════════════════════════════════

def setup_context(*, mode_label: str, **kwargs) -> Step2bContext:
    """Build the immutable :class:`Step2bContext` from the dispatcher's kwargs.

    Consolidates path resolution, fft cache metadata reading, sparse pair
    ranking, and memory budget checking into a single frozen dataclass.

    Parameters
    ----------
    mode_label : str
        Backend label for log messages, e.g. ``"serial"`` / ``"4 CPU workers"`` /
        ``"2 GPUs"``.
    **kwargs : dict
        The same kwargs the dispatcher passes through (see
        :func:`dasfm.steps.step2b_polarity.run`).
    """
    project_dir = kwargs["project_dir"]
    event_catalog = kwargs["event_catalog"]
    das_fft = kwargs["das_fft"]
    das_geo = kwargs["das_geo"]
    polarity_method = kwargs.get("polarity_method", "hilbert")
    sparse = kwargs.get("sparse", False)

    root = Path(project_dir).resolve()
    log_dir = root / "logs"
    fig_dir = root / "cache/figs/stage2b"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger("step2b_polarity", log_dir=str(log_dir))

    # ── Header ────────────────────────────────────────────────────────────
    logger.info()
    logger.info("=" * 60)
    logger.info()
    logger.info(f"  step2b_polarity — MCCC + SVD ({mode_label})")
    logger.info()
    logger.info("=" * 60)

    # ── Resolve paths + device ────────────────────────────────────────────
    fft_dir = resolve_path(das_fft, root)
    das_geo_path = resolve_path(das_geo, root)
    cat_file = resolve_path(event_catalog, root)
    pol_out = resolve_path(kwargs["pol_out_path"], root)
    sta4mccc_ref = (resolve_path(kwargs["sta4mccc_ref"], root)
                    if kwargs.get("sta4mccc_ref") else None)
    sta_pol_path = (resolve_path(kwargs["sta_polarity"], root)
                    if kwargs.get("sta_polarity") else None)
    sta_geo_path = (resolve_path(kwargs["sta_geo"], root)
                    if kwargs.get("sta_geo") else None)
    cal_csv_path = root / "cache/sta4mccc_ref/das_cal_polarity.csv"

    device, num_gpu, gpu_devices = resolve_device(kwargs.get("device", "cpu"))
    use_gpu = num_gpu > 0
    num_cpu_workers = int(kwargs.get("num_cpu_workers", 1))

    # ── Catalog ───────────────────────────────────────────────────────────
    catalog = pd.read_csv(cat_file)
    event_ids = [str(eid) for eid in catalog["event_id"].values]
    n_ev = len(event_ids)

    # ── DAS geo ───────────────────────────────────────────────────────────
    das_geo_df = pd.read_csv(das_geo_path)
    channel_ids = das_geo_df["index"].values.astype(np.int32)
    n_ch = len(das_geo_df)

    # ── Read fft cache metadata once (nfast / dt / has_lr) ────────────────
    logger.info("[1/4] Load precomputed FFT")
    nfast = dt_val = has_lr = None
    for eid in event_ids:
        meta = load_das_fft_meta(fft_dir, eid)
        if meta is not None:
            nfast = meta["nfast"]
            dt_val = meta["dt"]
            has_lr = meta["has_lr"]
            break
    if nfast is None:
        raise FileNotFoundError(
            f"No FFT cache found in {fft_dir}. Run step2a_window first."
        )

    if polarity_method == "hilbert" and not has_lr:
        raise FileNotFoundError(
            f"polarity_method='hilbert' requires Hilbert L/R FFTs in das_fft, "
            f"but {fft_dir} was written with hilbert=False. "
            f"Re-run step2a_window with hilbert=True."
        )

    use_lr = (polarity_method == "hilbert") and bool(has_lr)
    n_total_pairs = n_ev * (n_ev - 1) // 2

    logger.info(f"  {'Events':<12}: {n_ev}")
    logger.info(f"  {'Channels':<12}: {n_ch}")
    logger.info(f"  {'Total pairs':<12}: {n_total_pairs}")
    logger.info(f"  {'Hybrid LR':<12}: {has_lr}")

    # ── Sparse pair selection (precompute rankings + Group A / B) ─────────
    rankings = pairs_A = pairs_B = None
    if sparse:
        cat_df = pd.read_csv(cat_file)
        rank_dev = "cpu" if num_gpu == 0 else gpu_devices[0]

        import random as _random
        remote_seed = _random.randint(0, 2**31)

        logger.info("  Precomputing pair rankings...")
        rankings = precompute_pair_rankings(
            catalog_df=cat_df, fft_dir=fft_dir, event_ids=event_ids,
            dt=dt_val, maxlag=kwargs.get("mccc_max_lag_sec", 0.5),
            xcorr_subsample=kwargs.get("xcorr_subsample", 10),
            device=rank_dev,
            remote_seed=remote_seed,
        )

        k_n  = kwargs.get("k_neighbors", 10)
        frac = kwargs.get("top_xcorr_frac", 0.1)
        n_r  = kwargs.get("n_remote", 3)
        pairs_A = set(select_from_rankings(
            rankings, k_neighbors=k_n,
            top_xcorr_frac=frac, n_remote=n_r,
        ))
        n_total_full = rankings["n_total"]
        logger.info(f"  Pairs A (user): {len(pairs_A)} "
                    f"({len(pairs_A)/n_total_full*100:.1f}% of {n_total_full})")

        pairs_B, _, _, _ = expand_pairs(rankings, pairs_A, k_n, frac, n_r, logger)
        pairs_B = set(pairs_B)
        logger.info(f"  Pairs B (check): {len(pairs_B)} "
                    f"({len(pairs_B)/n_total_full*100:.1f}% of {n_total_full})")

    # ── Dense memory budget check ─────────────────────────────────────────
    _check_dense_memory_budget(n_ch, n_ev, logger)

    # ── Construct frozen ctx ──────────────────────────────────────────────
    return Step2bContext(
        project_dir=root,
        fft_dir=fft_dir,
        pol_out_path=pol_out,
        mccc_cache_path=(resolve_path(kwargs["mccc_cache_path"], root)
                         if kwargs.get("mccc_cache_path") else None),
        das_geo_path=das_geo_path,
        sta4mccc_ref=sta4mccc_ref,
        sta_polarity=sta_pol_path,
        sta_geo=sta_geo_path,
        log_dir=log_dir,
        fig_dir=fig_dir,
        cal_csv_path=cal_csv_path,

        event_ids=event_ids,
        n_ev=n_ev,
        n_ch=n_ch,
        nfast=nfast,
        dt=float(dt_val),
        has_lr=bool(has_lr),
        das_geo_df=das_geo_df,
        channel_ids=channel_ids,

        mccc_max_lag_sec=kwargs.get("mccc_max_lag_sec", 0.5),
        mccc_maxwin=kwargs.get("mccc_maxwin", 10),
        mccc_damp=kwargs.get("mccc_damp", 1.0),
        mccc_max_shift=kwargs.get("mccc_max_shift", 2),
        polarity_smooth_window=kwargs.get("polarity_smooth_window", 10),
        polarity_method=polarity_method,
        cal_min_picks=kwargs.get("cal_min_picks", 5),

        sparse=sparse,
        k_neighbors=kwargs.get("k_neighbors", 10),
        top_xcorr_frac=kwargs.get("top_xcorr_frac", 0.1),
        n_remote=kwargs.get("n_remote", 3),
        xcorr_subsample=kwargs.get("xcorr_subsample", 10),
        stability_threshold=kwargs.get("stability_threshold", 0.99),
        rankings=rankings,
        pairs_A=pairs_A,
        pairs_B=pairs_B,

        device=device,
        use_gpu=use_gpu,
        num_gpu=num_gpu,
        gpu_devices=list(gpu_devices),
        num_cpu_workers=num_cpu_workers,

        use_lr=use_lr,

        show_plots=bool(kwargs.get("show_plots", False)),
        t0=_time.time(),
        logger=logger,
        mode_label=mode_label,
    )


def _check_dense_memory_budget(n_ch: int, n_ev: int, logger):
    """Verify that dense ``(n_ch, n_ev, n_ev)`` Ckij + Skij fit in the budget."""
    bytes_needed = 2 * n_ch * n_ev * n_ev * 4   # 2 matrices, float32
    avail = psutil.virtual_memory().available * 0.5
    fraction = bytes_needed / avail
    logger.info(
        f"  Dense Ckij/Skij: {bytes_needed/1e9:.2f} GB "
        f"({fraction*100:.1f}% of {avail/1e9:.1f} GB available)"
    )
    if fraction > 0.30:
        max_n_ev = int((0.30 * avail / (2 * n_ch * 4)) ** 0.5)
        raise MemoryError(
            f"Dense Ckij/Skij would need {bytes_needed/1e9:.1f} GB "
            f"({fraction*100:.0f}% of available RAM budget, threshold 30%). "
            f"Reduce events to ≤{max_n_ev}, or wait for sparse-storage backend."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Dense Ckij/Skij allocation
# ═══════════════════════════════════════════════════════════════════════════

def alloc_dense_ckij(ctx: Step2bContext):
    """Allocate dense ``Ckij + Skij`` with the diagonal initialised to 1."""
    n_ch, n_ev = ctx.n_ch, ctx.n_ev
    Ckij = np.zeros((n_ch, n_ev, n_ev), dtype=np.float32)
    Skij = np.zeros((n_ch, n_ev, n_ev), dtype=np.float32)
    diag = np.eye(n_ev, dtype=np.float32)[None, :, :]
    Ckij[:] = diag
    Skij[:] = diag
    return Ckij, Skij


# ═══════════════════════════════════════════════════════════════════════════
#  run_with_iteration — sparse-aware MCCC driver (callback strategy)
# ═══════════════════════════════════════════════════════════════════════════

def run_with_iteration(ctx: Step2bContext, run_mccc_fn):
    """Sparse-aware MCCC driver.

    Allocates the dense ``Ckij/Skij``, then dispatches to ``run_mccc_fn``:

    * **Non-sparse**: one call to ``run_mccc_fn(all_pairs, Ckij, Skij)``.
    * **Sparse**: ``run_mccc_fn(pairs_B, Ckij, Skij)``, then iterate
      ``stability_check`` / ``expand_pairs`` until passed.

    Parameters
    ----------
    ctx : Step2bContext
    run_mccc_fn : callable
        ``run_mccc_fn(pair_set: set, Ckij: ndarray, Skij: ndarray) -> None``.

    Returns
    -------
    (Ckij, Skij, svd_result)
        ``svd_result`` is ``None`` for non-sparse, or the SVD result from
        the last ``stability_check`` for sparse (reused by postprocess).
    """
    from dasfm.picking.sparse_pairs import stability_check

    Ckij, Skij = alloc_dense_ckij(ctx)
    logger = ctx.logger

    if not ctx.sparse:
        all_pairs = {(i, j) for i in range(ctx.n_ev)
                            for j in range(i + 1, ctx.n_ev)}
        logger.info(f"[2/4] MCCC cross-correlation ({ctx.mode_label})")
        run_mccc_fn(all_pairs, Ckij, Skij)
        logger.info(f"  MCCC done ({len(all_pairs)} pairs)")
        return Ckij, Skij, None

    # ── Sparse path ───────────────────────────────────────────────────────
    pairs_done = set(ctx.pairs_B)
    logger.info(f"[2/4] MCCC cross-correlation ({ctx.mode_label})")
    logger.info(f"  Computing {len(pairs_done)} pairs (group B)...")
    run_mccc_fn(pairs_done, Ckij, Skij)
    logger.info(f"  MCCC done ({len(pairs_done)} pairs)")

    pairs_A = set(ctx.pairs_A)
    k_n  = ctx.k_neighbors
    frac = ctx.top_xcorr_frac
    n_r  = ctx.n_remote
    round_i = 0
    svd_res = None
    while True:
        passed, pct, svd_res = stability_check(Ckij, Skij, pairs_A, ctx)
        if passed:
            return Ckij, Skij, svd_res

        round_i += 1
        logger.info(f"  Round {round_i}: expanding pairs (sign_match={pct:.1f}%)")
        new_B, k_n, frac, n_r = expand_pairs(
            ctx.rankings, pairs_done, k_n, frac, n_r, logger,
        )
        new_B = set(new_B)
        extra = new_B - pairs_done
        if not extra:
            logger.info("  expand_pairs produced no new pairs — accepting current state")
            return Ckij, Skij, svd_res

        logger.info(f"  Computing {len(extra)} extra pairs...")
        run_mccc_fn(extra, Ckij, Skij)
        pairs_A = pairs_done
        pairs_done |= extra
