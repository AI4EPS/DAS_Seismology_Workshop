"""polarity_postprocess — SVD polarity extraction + orchestration.

Pure orchestration of the final step2b stage:

    0. Save MCCC cache (Ckij/Skij)           (optional)
    1. SVD decomposition → Pkic matrix       (polarity_svd)
    2. Sign calibration                      (sign_correction)
    3. Save polarity H5                      (polarity_io)
    4. QC plots                              (polarity_qc)
"""
from __future__ import annotations

import time as _time

import h5py
import numpy as np

from dasfm.picking.polarity_svd import solve_polarity_svd
from dasfm.picking.sign_correction import resolve_or_compute_cal, apply_calibration
from dasfm.io.polarity_io import save_polarity_h5, build_pol_valid
from dasfm.picking.polarity_qc import plot_polarity_qc


def postprocess(ctx, Ckij, Skij, svd_result=None):
    """[3/4] SVD + sign correction, [4/4] write H5 + QC plots.

    Parameters
    ----------
    ctx : Step2bContext
    Ckij, Skij : np.ndarray, shape (n_ch, n_ev, n_ev), float32
        Persistent dense matrices, written by ``run_with_iteration``.
    svd_result : dict or None
        If non-None, reuse this SVD result (set by sparse mode's
        ``stability_check``).  Otherwise call ``solve_polarity_svd`` here.
    """
    logger = ctx.logger

    # ── Save MCCC cache (Ckij/Skij) if requested ──
    if ctx.mccc_cache_path is not None:
        ctx.mccc_cache_path.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack([Ckij, Skij], axis=-1)  # (n_ch, n_ev, n_ev, 2)
        with h5py.File(ctx.mccc_cache_path, "w") as f:
            f.create_dataset("pol_LR_mccc_shift", data=stacked,
                             compression="lzf", chunks=True)
        logger.info(f"  MCCC cache saved: {ctx.mccc_cache_path}  "
                    f"({stacked.shape}, {stacked.nbytes/1e6:.1f} MB)")

    # ── [3/4] SVD ──
    if svd_result is not None:
        res = svd_result
        logger.info("[3/4] SVD polarity (reused from stability check)")
    else:
        logger.info("[3/4] SVD polarity")
        res = solve_polarity_svd(Ckij, Skij, logger=logger)
    logger.info(f"  Pkic shape : {res['Pkic'].shape}")
    Pkic = res["Pkic"].copy()

    # ── Sign calibration ──
    cal_path = resolve_or_compute_cal(ctx, logger)
    Pkic, agree, disagree = apply_calibration(Pkic, cal_path, ctx, logger)

    # ── pol_valid + Save H5 ──
    pol_valid = build_pol_valid(ctx.fft_dir, ctx.event_ids, ctx.n_ch)
    save_polarity_h5(ctx.pol_out_path, Pkic, pol_valid, res["svd_info"],
                     ctx.event_ids, ctx.channel_ids, logger=logger)

    # ── [4/4] QC plots ──
    logger.info("[4/4] QC plots")
    plot_polarity_qc(ctx, Pkic, cal_path, agree, disagree)
    logger.info(f"  -> {ctx.fig_dir}  (4 plots)")

    logger.info()
    logger.info("=" * 60)
    logger.info(f"  Done  ({_time.time() - ctx.t0:.1f} s)")
    logger.info("=" * 60)
    logger.close()
