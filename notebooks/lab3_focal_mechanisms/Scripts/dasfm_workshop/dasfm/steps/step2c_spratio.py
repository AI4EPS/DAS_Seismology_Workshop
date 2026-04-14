"""step2c_spratio — Extract DAS S/P amplitude ratios."""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from dasfm.utils.step_utils import Logger, resolve_path
from dasfm.io.das_io import validate_das_geo
from dasfm.io.das_fft import validate_das_win_dir
from dasfm.io.event_catalog_io import validate_event_catalog, validate_per_event_files


def sp_ratio_single_event(args):
    """Process a single event for S/P ratio extraction (worker function)."""
    das_win_dir, eid = args
    from dasfm.io.das_fft import load_das_window_single
    from dasfm.picking.amplitude_windata import pick_sp_ratio_single

    # Pre-flight in run() guarantees the file exists
    single = load_das_window_single(das_win_dir, eid)
    res = pick_sp_ratio_single(single["dasdata"])
    return {
        "eid": eid,
        "p_amp": res["p_amp"],
        "s_amp": res["s_amp"],
        "sp_valid": res["sp_valid"],
        "dt": single["dt"],
        "n_ch": single["n_ch"],
    }


def run(
    project_dir="",
    event_catalog="",
    das_win="",
    das_geo="",
    sp_out_path=None,
    num_cpu_workers=1,
    show_plots=False):
    """Extract DAS S/P amplitude ratios.

    Outputs:
        cache/sp_ratios/sp_ratios.h5
        cache/figs/stage2c/*.png
    """
    _required = {
        "project_dir": project_dir, "event_catalog": event_catalog,
        "das_win": das_win, "das_geo": das_geo, "sp_out_path": sp_out_path,
    }
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    from tqdm import tqdm

    root = Path(project_dir).resolve()
    CACHE_DIR = root / "cache"
    logger = Logger("step2c_spratio", log_dir=str(root / "logs"))

    CAT_FILE    = resolve_path(event_catalog, root)
    DAS_WIN_DIR = resolve_path(das_win,       root)
    DAS_GEO     = resolve_path(das_geo,       root)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(CAT_FILE)
    validate_das_geo(DAS_GEO)
    validate_das_win_dir(DAS_WIN_DIR)
    _eids_pf = pd.read_csv(CAT_FILE)["event_id"].astype(str).tolist()
    validate_per_event_files(_eids_pf, DAS_WIN_DIR, ".h5",
                             label="DAS windowed", upstream_step="step2a_window")

    t0 = _time.time()
    logger.info()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step2c_spratio — Extract DAS S/P amplitude ratios")
    logger.info()
    logger.info("=" * 60)

    catalog   = pd.read_csv(CAT_FILE)
    EVENT_IDS = [str(eid) for eid in catalog["event_id"].values]

    # ── [1/4] Load & pick S/P ratios per event ──────────────────────────────
    logger.info("[1/4] Load & pick S/P ratios per event")

    worker_args = [(DAS_WIN_DIR, eid) for eid in EVENT_IDS]

    if num_cpu_workers > 1:
        import multiprocessing
        logger.info(f"  Parallel: {num_cpu_workers} CPU workers")
        with multiprocessing.Pool(num_cpu_workers) as pool:
            raw_results = list(tqdm(
                pool.imap(sp_ratio_single_event, worker_args),
                total=len(worker_args), desc="S/P ratio", unit="ev"))
    else:
        raw_results = []
        for args in tqdm(worker_args, desc="S/P ratio", unit="ev"):
            raw_results.append(sp_ratio_single_event(args))

    loaded_eids = []
    p_amp_list = []
    s_amp_list = []
    sp_valid_list = []
    dt = None
    n_ch = 0
    for r in raw_results:
        loaded_eids.append(r["eid"])
        p_amp_list.append(r["p_amp"])
        s_amp_list.append(r["s_amp"])
        sp_valid_list.append(r["sp_valid"])
        if dt is None:
            dt = r["dt"]
            n_ch = r["n_ch"]
        logger.log(f"  {r['eid']}  {r['n_ch']} ch")

    n_ev = len(loaded_eids)
    logger.info(f"  Done: {n_ev} ok")
    logger.info(f"  {'Channels':<12}: {n_ch}")
    logger.info(f"  {'dt':<12}: {dt} s")

    # ── [2/4] Assemble S/P ratios ────────────────────────────────────────────
    logger.info("[2/4] Assemble S/P ratios")

    p_amp    = np.column_stack(p_amp_list)     # (n_ch, n_ev)
    s_amp    = np.column_stack(s_amp_list)     # (n_ch, n_ev)
    sp_valid = np.column_stack(sp_valid_list)  # (n_ch, n_ev)
    event_ids = loaded_eids

    sp_valid &= (p_amp > 0) & (s_amp > 0)
    safe_p = np.where(sp_valid, p_amp, 1.0)
    safe_s = np.where(sp_valid, s_amp, 1.0)
    sp_ratios = np.log10(safe_s / safe_p)
    sp_ratios[~sp_valid] = np.nan
    logger.info(f"  {'Valid frac':<12}: {sp_valid.mean():.1%}")

    # ── [3/4] (Channel-event decomposition disabled — method has known issues) ──
    sp_ratios_clean = sp_ratios.copy()
    sp_ratios_clean[~sp_valid] = np.nan

    # NOTE: Original channel-event decomposition kept below for reference only.
    # The method had a systematic bias in the theoretical/observed S/P comparison;
    # disabled until a corrected formulation is available.
    #
    # sp_ratios_raw = sp_ratios.copy()
    # n_ch_sp, n_ev_sp = sp_ratios.shape
    # sp_masked  = np.where(sp_valid, sp_ratios, np.nan)
    # n_valid_per_ch = sp_valid.sum(axis=1).astype(float)
    # mu_ch = np.nanmean(sp_masked, axis=1)
    # decomp_lambda = (n_valid_per_ch * mu_ch / (1.0 + n_valid_per_ch)).sum() \
    #        / (1.0 / (1.0 + n_valid_per_ch)).sum()
    # couple_sp = (n_valid_per_ch * mu_ch - decomp_lambda) / (1.0 + n_valid_per_ch)
    # sp_ratios_clean = sp_ratios - couple_sp[:, np.newaxis]
    # sp_ratios_clean[~sp_valid] = np.nan
    # logger.info(f"  sum(C)      : {couple_sp.sum():.4e}  (should be ≈ 0)")
    # logger.info(f"  couple_sp std: {couple_sp.std():.4f}")

    # ── [4/4] Save & QC plots ─────────────────────────────────────────────────
    logger.info("[4/4] Save & QC plots")

    sp_out_path = resolve_path(sp_out_path, root)
    sp_out_path.parent.mkdir(parents=True, exist_ok=True)
    das_geo_df  = pd.read_csv(DAS_GEO)
    channel_ids = das_geo_df["index"].values.astype(np.int32)

    import h5py
    with h5py.File(sp_out_path, "w") as f:
        f.create_dataset("sp_ratios", data=sp_ratios_clean)
        f.create_dataset("p_amp", data=p_amp)
        f.create_dataset("s_amp", data=s_amp)
        f.create_dataset("sp_valid", data=sp_valid)
        f.create_dataset("event_ids", data=np.array(event_ids, dtype=h5py.string_dtype()))
        f.create_dataset("channel_ids", data=channel_ids)
    logger.info(f"  → {sp_out_path}")

    import matplotlib
    import matplotlib.pyplot as plt

    fig_dir = CACHE_DIR / "figs/stage2c"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_ch_sp = sp_ratios_clean.shape[0]
    n_ev_sp = sp_ratios_clean.shape[1]

    sp_plot = sp_ratios_clean.copy().astype(float)
    sp_plot[~sp_valid] = np.nan
    vmin_sp = np.nanpercentile(sp_plot, 1)
    vmax_sp = np.nanpercentile(sp_plot, 99)

    n_plots = 0

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    im = ax.imshow(sp_plot, aspect="auto", cmap="viridis",
                   vmin=vmin_sp, vmax=vmax_sp, origin="lower")
    ax.set_xlabel("Event index"); ax.set_ylabel("Channel index")
    ax.set_title(f"log₁₀(S/P) heatmap ({n_ch_sp} ch × {n_ev_sp} ev)")
    fig.colorbar(im, ax=ax, label="log₁₀(S/P)")
    fig.tight_layout()
    fig.savefig(fig_dir / "sp_ratios_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "sp_ratios_heatmap.png")))
    n_plots += 1

    sp_flat = sp_plot[sp_valid]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if len(sp_flat) > 0:
        ax.hist(sp_flat, bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel("log₁₀(S/P)"); ax.set_ylabel("Count")
    ax.set_title("Global log₁₀(S/P) distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "sp_ratio_histogram.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "sp_ratio_histogram.png")))
    n_plots += 1

    logger.info(f"  → {fig_dir}  ({n_plots} plots)")
    logger.info()
    logger.info("=" * 60)
    logger.info()
    logger.info(f"  Done  ({_time.time() - t0:.1f} s)")
    logger.info()
    logger.info("=" * 60)
    logger.close()
