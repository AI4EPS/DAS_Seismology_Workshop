"""result_analysis — Shared utilities for result summarization and comparison.

Functions used by step4_summarize, step4_compare_runs, step4_compare_modes,
step3_plot, and visualization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dasfm.utils.step_utils import JOINT_MODES, MODE_TITLES, MODE_SOL_KEY, MODE_COLORS
from dasfm.io.result_io import load_inversion_result


# ── Shared constants ──────────────────────────────────────────────────────────

PLOT_MODES = {m: {"label": MODE_TITLES[m], "sol_key": MODE_SOL_KEY[m]}
              for m in MODE_TITLES}

JOINTPLOT_MODES = JOINT_MODES

DEFAULT_GRADES = pd.DataFrame({
    "avg_misfit_max":      [0.15, 0.20, 0.30],
    "rms_uncertainty_max": [25,   35,   45  ],
    "station_ratio_min":   [0.5,  0.4,  0.3 ],
    "mech_prob_min":       [0.8,  0.6,  0.5 ],
    "amp_rate_max":        [0.01, 0.02, 0.05],
}, index=["A", "B", "C"])


def load_grades(root: Path) -> pd.DataFrame:
    f = root / "quality_grades.txt"
    if f.exists():
        return pd.read_csv(f, sep=r'\s+', comment='#').set_index("grade")
    return DEFAULT_GRADES


def hash_quality_grade(avg_misfit, rms_uncertainty, station_ratio, mech_prob,
                        amp_rate=0.0, use_amp=False,
                        grades_df=None):
    if grades_df is None:
        grades_df = DEFAULT_GRADES
    sr = station_ratio
    ar = amp_rate      if use_amp  else 0.0
    for grade in grades_df.index:
        row = grades_df.loc[grade]
        if (avg_misfit       <= row["avg_misfit_max"]
                and rms_uncertainty <= row["rms_uncertainty_max"]
                and sr              >= row["station_ratio_min"]
                and mech_prob       >= row["mech_prob_min"]
                and ar              <= row["amp_rate_max"]):
            return grade
    return "D"


def extract_candidate(cand, eid, ss, rank, mode, grades_df):
    entry = {
        "event_id": eid, "event_idx": ss, "rank": rank,
        "has_solution": True,
        "stk": cand["stk_mean"],
        "dip": cand["dip_mean"],
        "rak": cand["rak_mean"],
        "accept_ratio": cand["accept_ratio"],
        "misfit_pol": cand["misfit_pol_ratio0"],
        "kagan_rms": cand["kagan_rms"],
        "stdr": cand.get("stdr", 0.0),
        "misfit_amp_rate": (float("nan")
                            if np.isnan(cand.get("misfit_amp_rate0", 0.0))
                            else cand["misfit_amp_rate0"]),
    }
    entry["quality"] = hash_quality_grade(
        entry["misfit_pol"], entry["kagan_rms"],
        entry["stdr"], entry["accept_ratio"],
        amp_rate=entry["misfit_amp_rate"],
        use_amp=(mode in JOINTPLOT_MODES),
        grades_df=grades_df,
    )
    return entry


def load_results(result_root, event_ids, plot_modes, grades_df):
    results_best = {m: [] for m in PLOT_MODES}
    results_all  = {m: [] for m in PLOT_MODES}
    for ss, eid in enumerate(event_ids):
        h5_dir = result_root / "inv_sol" / eid
        for mode, cfg in PLOT_MODES.items():
            h5_file = h5_dir / f"{mode}.h5"
            if h5_file.exists():
                data = load_inversion_result(h5_file)
                cand_list = data.get(cfg["sol_key"], [])
                if cand_list:
                    for rank, cand in enumerate(cand_list):
                        results_all[mode].append(
                            extract_candidate(cand, eid, ss, rank, mode, grades_df))
                    results_best[mode].append(
                        extract_candidate(cand_list[0], eid, ss, 0, mode, grades_df))
                    continue
                elif data.get("skipped", False):
                    results_best[mode].append({
                        "event_id": eid, "event_idx": ss,
                        "has_solution": False, "quality": "Skipped",
                    })
                    continue
            results_best[mode].append({
                "event_id": eid, "event_idx": ss,
                "has_solution": False, "quality": "No solution",
            })
    return (
        {m: pd.DataFrame(rows) for m, rows in results_best.items()},
        {m: pd.DataFrame(rows) for m, rows in results_all.items()},
    )


def select_result_dir(base: Path, prefix: str) -> Path:
    """Auto-select latest valid result dir under base."""
    n, valid = 1, []
    while True:
        p = base / f"{prefix}try_dir{n}"
        if not p.exists():
            break
        if any(p.glob("inv_sol/*/*.h5")):
            valid.append(p)
        n += 1
    if not valid:
        return base / f"{prefix}try1"
    return valid[-1]
