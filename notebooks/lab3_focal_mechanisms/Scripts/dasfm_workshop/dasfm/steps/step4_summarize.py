"""step4_summarize — Summarize and compare focal mechanism inversion results."""

from __future__ import annotations

import time as _time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dasfm.utils.step_utils import (
    Logger, resolve_path, parse_inversion_types,
    MODE_COLORS,
)
from dasfm.io.result_io import load_inversion_result, validate_result_dir
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.utils.result_analysis import (
    PLOT_MODES, JOINTPLOT_MODES, DEFAULT_GRADES,
    load_grades, hash_quality_grade,
    load_results, select_result_dir,
)


# ─────────────────────────────────────────────────────────────────────────────


def run(
    project_dir="",
    event_catalog="",
    inversion_types=None,
    compute_uncertainty=False,
    result_dir=None,
    best_only=False,
    show_plots=False):
    """Summarize inversion results: quality grades, metric distributions, SKHASH export.

    Parameters
    ----------
    result_dir : str or None
        Path to a specific result directory (e.g. "result_try2").
        If None, auto-selects the latest valid result_try* dir.
    best_only : bool
        If True (default), grade statistics use only the best candidate per event.
        If False, use all candidates.

    Outputs
    -------
        {result_dir}/summary_figs/*.png
        {result_dir}/skhash_format/{mode}outfile1.csv
        {result_dir}/skhash_format/{mode}outfile2.csv
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog, "inversion_types": inversion_types}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    import matplotlib
    import matplotlib.pyplot as plt

    root = Path(project_dir).resolve()
    logger = Logger("step4_summarize", log_dir=str(root / "logs"))
    grades_df = load_grades(root)

    CAT_FILE = resolve_path(event_catalog, root)
    result_prefix = "result_uncert" if compute_uncertainty else "result"

    if result_dir is not None:
        RESULTroot = resolve_path(result_dir, root)
    else:
        RESULTroot = select_result_dir(root, result_prefix)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(CAT_FILE)
    validate_result_dir(RESULTroot)

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step4_summarize — Summarize inversion results")
    logger.info()
    logger.info("=" * 60)
    logger.info(f"  {'Result dir':<12}: {RESULTroot}")

    FIG_DIR = RESULTroot / "summary_figs"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    inversion_types = parse_inversion_types(inversion_types)
    PLOTPLOT_MODES = [m for m in inversion_types if m in PLOT_MODES]

    # ── [1/4] Load results ──
    logger.info(f"[1/4] Load results")
    catalog   = pd.read_csv(CAT_FILE)
    event_ids = [str(eid) for eid in catalog["event_id"].values]
    n_ev = len(event_ids)
    logger.info(f"  {'Events':<12}: {n_ev}")

    dfs_best, dfs_all = load_results(RESULTroot, event_ids, PLOTPLOT_MODES, grades_df)
    results_best = {m: dfs_best[m].to_dict("records") for m in PLOT_MODES}
    results_all  = {m: dfs_all[m].to_dict("records")  for m in PLOT_MODES}
    results_use  = results_best if best_only else results_all
    stat_label = "best candidate" if best_only else "all candidates"

    for mode in PLOTPLOT_MODES:
        n_sol = int(dfs_best[mode]["has_solution"].sum()) if "has_solution" in dfs_best[mode].columns else 0
        n_all = len(dfs_all[mode])
        logger.info(f"  {mode:20s}: {n_sol}/{n_ev} events with solutions, {n_all} total candidates")
    logger.info(f"  {'Stat mode':<20s}: {stat_label}")

    # ── [2/4] Quality grading ──
    logger.info(f"[2/4] Quality grading")
    grade_order = ["D", "C", "B", "A"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    n_plot = len(PLOTPLOT_MODES)
    width = 0.8 / max(n_plot, 1)
    x = np.arange(len(grade_order))
    for i, mode in enumerate(PLOTPLOT_MODES):
        grades = [r["quality"] for r in results_use[mode]]
        cnt = Counter(grades)
        counts = np.array([cnt.get(g, 0) for g in grade_order])
        offset = (i - (n_plot - 1) / 2) * width
        ax.bar(x + offset, counts, width,
               color=MODE_COLORS[mode], edgecolor="black", linewidth=0.6,
               label=PLOT_MODES[mode]["label"], zorder=3)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(grade_order, fontsize=11)
    ax.set_xlabel("Quality Grade", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Focal Mechanism Quality — {stat_label}", fontsize=14)
    ax.legend(fontsize=9, frameon=True, edgecolor="lightgray")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_comparison_4modes.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(FIG_DIR / "quality_comparison_4modes.png")))
    logger.info(f"  → {FIG_DIR / 'quality_comparison_4modes.png'}")

    # ── [3/4] Grade summary table
    logger.info("\n" + "=" * 60)
    logger.info("Quality Grade Summary")
    logger.info("=" * 60)
    header = f"{'Grade':<14s}"
    for mode in PLOTPLOT_MODES:
        header += f"{PLOT_MODES[mode]['label']:>20s}"
    logger.info(header)
    logger.info("-" * 60)
    for g in ["A", "B", "C", "D"]:
        line = f"{g:<14s}"
        for mode in PLOTPLOT_MODES:
            cnt = sum(1 for r in results_use[mode] if r["quality"] == g)
            line += f"{cnt:>20d}"
        logger.info(line)
    total_line = f"{'Total':<14s}"
    for mode in PLOTPLOT_MODES:
        total_line += f"{len(results_use[mode]):>20d}"
    logger.info("-" * 60)
    logger.info(total_line)

    # corr_factor plot (skipped when sp_decay_c is set — mean alignment disabled, no corr_factor in results)
    cf_modes = [m for m in PLOTPLOT_MODES if m in JOINTPLOT_MODES and m != "STA_pol_sp"]
    if cf_modes:
        cf_data = {}
        for mode in cf_modes:
            cf_all = []
            for eid in event_ids:
                h5_file = RESULTroot / "inv_sol" / eid / f"{mode}.h5"
                if not h5_file.exists(): continue
                data = load_inversion_result(h5_file)
                cf = data.get("corr_factor")
                if cf is None: continue
                cf_all.extend(np.asarray(cf).flatten().tolist())
            cf_data[mode] = cf_all

        has_cf = any(len(v) > 0 for v in cf_data.values())
        if has_cf:
            fig, axes = plt.subplots(1, len(cf_modes), figsize=(6 * len(cf_modes), 4), dpi=200)
            if len(cf_modes) == 1:
                axes = [axes]
            for ax, mode in zip(axes, cf_modes):
                cf_all = cf_data[mode]
                if cf_all:
                    cf_arr = np.array(cf_all)
                    cf_arr = cf_arr[np.isfinite(cf_arr)]
                    if len(cf_arr) > 0:
                        ax.hist(cf_arr, bins=30, color=MODE_COLORS[mode], edgecolor="black",
                                linewidth=0.6, alpha=0.8)
                        ax.axvline(np.median(cf_arr), color="red", ls="--", lw=1.5,
                                   label=f"median={np.median(cf_arr):.3f}")
                        ax.axvline(np.mean(cf_arr), color="blue", ls="--", lw=1.5,
                                   label=f"mean={np.mean(cf_arr):.3f}")
                        ax.legend(fontsize=9)
                ax.set_xlabel("corr_factor (theo_mean − obs_mean)", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.set_title(f"{PLOT_MODES[mode]['label']}\n({len(cf_all)} solutions)", fontsize=12)
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("S/P Ratio Correction Factor Distribution", fontsize=14, y=1.02)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "corr_factor_distribution.png", dpi=300, bbox_inches="tight")
            plt.close("all")
            if show_plots:
                from IPython.display import display, Image
                display(Image(filename=str(FIG_DIR / "corr_factor_distribution.png")))
            logger.info(f"  → {FIG_DIR / 'corr_factor_distribution.png'}")
        else:
            logger.info("  corr_factor plot skipped (sp_decay_c set, no corr_factor in results)")

    # ── [4/4] Export SKHASH ──
    logger.info(f"[4/4] Export SKHASH")
    from dasfm.inversion.moment_tensor import sdr2ns

    SKHASH_DIR = RESULTroot / "skhash_format"
    SKHASH_DIR.mkdir(parents=True, exist_ok=True)

    catalog_full = pd.read_csv(CAT_FILE)
    cat_eid_map = {str(row["event_id"]): row for _, row in catalog_full.iterrows()}
    depth_col = "depth" if "depth" in catalog_full.columns else "depth_km"

    for mode in PLOTPLOT_MODES:
        cfg = PLOT_MODES[mode]
        is_joint = mode in JOINTPLOT_MODES
        out1_rows = []
        out2_rows = []

        for ss, eid in enumerate(event_ids):
            h5_file = RESULTroot / "inv_sol" / eid / f"{mode}.h5"
            if not h5_file.exists(): continue
            data = load_inversion_result(h5_file)
            cand_list = data.get(cfg["sol_key"], [])
            if not cand_list: continue

            cat_row = cat_eid_map.get(eid, {})
            mult_flag = len(cand_list) > 1

            for rank, cand in enumerate(cand_list):
                stk_c = cand["stk_mean"]
                dip_c = cand["dip_mean"]
                rak_c = cand["rak_mean"]
                misfit_pol = cand["misfit_pol_ratio0"]
                kagan_rms = cand["kagan_rms"]
                stdr_val = cand.get("stdr", 0.0)
                accept_ratio = cand["accept_ratio"]
                misfit_amp_rate = cand.get("misfit_amp_rate0", float("nan"))

                quality = hash_quality_grade(
                    misfit_pol, kagan_rms, stdr_val, accept_ratio,
                    amp_rate=misfit_amp_rate,
                    use_amp=is_joint,
                    grades_df=grades_df,
                )
                fp_uncert = cand.get("angleN_rms", cand.get("kagan_rms", np.nan))
                aux_uncert = cand.get("angleS_rms", np.nan)
                sp_misfit = 100 * cand.get("misfit_amp0", 0.0) if is_joint else 0.0

                row1 = {
                    "event_id": eid,
                    "strike": round(stk_c, 1),
                    "dip": round(dip_c, 1),
                    "rake": round(rak_c, 1),
                    "quality": quality,
                    "fault_plane_uncertainty": round(fp_uncert, 1) if not np.isnan(fp_uncert) else "",
                    "aux_plane_uncertainty": round(aux_uncert, 1) if not np.isnan(aux_uncert) else "",
                    "num_das_pol": data.get("num_das_pol", 0),
                    "num_sta_pol": data.get("num_sta_pol", 0),
                    "num_das_sp": data.get("num_das_sp", 0),
                    "num_sta_sp": data.get("num_sta_sp", 0),
                    "polarity_misfit": round(misfit_pol, 4),
                    "prob_mech": round(accept_ratio, 4),
                    "sta_distribution_ratio": round(stdr_val, 4),
                    "sp_misfit": round(sp_misfit, 4) if is_joint else "",
                    "mult_solution_flag": mult_flag,
                }
                for col, out_col in [("time", "time"), ("latitude", "origin_lat"),
                                     ("longitude", "origin_lon"), (depth_col, "origin_depth_km")]:
                    if isinstance(cat_row, dict) and col in cat_row:
                        row1[out_col] = cat_row[col]
                    elif hasattr(cat_row, "__getitem__"):
                        try: row1[out_col] = cat_row[col]
                        except KeyError: pass

                if rank == 0:
                    out1_rows.append(row1)

                stk_all = np.atleast_1d(cand["stk"])
                dip_all = np.atleast_1d(cand["dip"])
                rak_all = np.atleast_1d(cand["rak"])
                for j in range(len(stk_all)):
                    n_vec, s_vec = sdr2ns(float(stk_all[j]), float(dip_all[j]), float(rak_all[j]))
                    out2_rows.append({
                        "event_id": eid, "mech_number": rank,
                        "strike": round(float(stk_all[j]), 1),
                        "dip": round(float(dip_all[j]), 1),
                        "rake": round(float(rak_all[j]), 1),
                        "norm_N": round(float(n_vec[0]), 4),
                        "norm_E": round(float(n_vec[1]), 4),
                        "norm_Z": round(float(n_vec[2]), 4),
                        "norm_aux_N": round(float(s_vec[0]), 4),
                        "norm_aux_E": round(float(s_vec[1]), 4),
                        "norm_aux_Z": round(float(s_vec[2]), 4),
                    })

        if out1_rows:
            df1 = pd.DataFrame(out1_rows)
            outfile1 = SKHASH_DIR / f"{mode}outfile1.csv"
            df1.to_csv(outfile1, index=False)
            logger.info(f"  → {outfile1}  ({len(df1)} events)")
        if out2_rows:
            df2 = pd.DataFrame(out2_rows)
            outfile2 = SKHASH_DIR / f"{mode}outfile2.csv"
            df2.to_csv(outfile2, index=False)
            logger.info(f"  → {outfile2}  ({len(df2)} mechanisms)")

    logger.info("=" * 60)
    logger.info()
    logger.info(f"  Done  ({_time.time() - t0:.1f} s)")
    logger.info()
    logger.info("=" * 60)
    logger.close()
