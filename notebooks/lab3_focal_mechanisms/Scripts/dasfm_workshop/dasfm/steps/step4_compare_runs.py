"""step4_compare_runs — Compare two inversion result directories."""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from dasfm.utils.step_utils import Logger, resolve_path, parse_inversion_types
from dasfm.utils.result_analysis import (
    PLOT_MODES,
    load_results,
    load_grades,
)
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.result_io import validate_result_dir


def run(
    project_dir="",
    event_catalog="",
    inversion_types=None,
    result_dir_a="",
    result_dir_b="",
    compute_uncertainty=False,
    best_only=True,
    show_plots=False):
    """Compare two inversion result directories.

    Parameters
    ----------
    result_dir_a, result_dir_b : str
        Paths (relative to project_dir or absolute) to two result directories
    best_only : bool
        If True (default), compare only the best candidate per event.
        If False, compare all candidates.
        (e.g. "result_try1", "result_try2").

    Outputs
    -------
        result_compare/{A_name}vs_label{B_name}/grade_transition_matrix.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog, "inversion_types": inversion_types, "result_dir_a": result_dir_a, "result_dir_b": result_dir_b}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    import matplotlib
    import matplotlib.pyplot as plt

    root = Path(project_dir).resolve()
    logger = Logger("step4_compare_runs", log_dir=str(root / "logs"))
    grades_df = load_grades(root)

    CAT_FILE = resolve_path(event_catalog, root)
    dir_a = resolve_path(result_dir_a, root)
    dir_b = resolve_path(result_dir_b, root)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(CAT_FILE)
    validate_result_dir(dir_a)
    validate_result_dir(dir_b)

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step4_compare_runs — Compare two inversion results")
    logger.info()
    logger.info("=" * 60)
    logger.info(f"  {'Dir A':<12}: {dir_a}")
    logger.info(f"  {'Dir B':<12}: {dir_b}")

    COMPARE_DIR = root / "result_compare" / f"{dir_a.name}_vs_{dir_b.name}"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  {'Output':<12}: {COMPARE_DIR}")

    inversion_types = parse_inversion_types(inversion_types)
    PLOTPLOT_MODES = [m for m in inversion_types if m in PLOT_MODES]

    catalog   = pd.read_csv(CAT_FILE)
    event_ids = [str(eid) for eid in catalog["event_id"].values]
    n_ev      = len(event_ids)
    logger.info(f"  {'Events':<12}: {n_ev}")

    # ── [1/3] Load results ──
    stat_label = "best candidate" if best_only else "all candidates"
    logger.info(f"[1/3] Load results from both directories ({stat_label})")
    dfs_best_a, dfs_all_a = load_results(dir_a, event_ids, PLOTPLOT_MODES, grades_df)
    dfs_best_b, dfs_all_b = load_results(dir_b, event_ids, PLOTPLOT_MODES, grades_df)

    # ── [2/3] Quality grade count table ──
    logger.info("[2/3] Quality grade counts")
    grade_full = ["A", "B", "C", "D", "No solution"]

    # Build header: one column per grade, one row per mode, format "A (B)"
    col_w = 12
    label_w = 24
    header = f"  {'Mode':<{label_w}}"
    for grade in grade_full:
        header += f"  {grade:>{col_w}}"
    logger.info("")
    logger.info(f"  Format: dir_a ({dir_a.name}) count,  (dir_b ({dir_b.name}) count)")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    use_a = dfs_best_a if best_only else dfs_all_a
    use_b = dfs_best_b if best_only else dfs_all_b
    for mode in PLOTPLOT_MODES:
        line = f"  {PLOT_MODES[mode]['label']:<{label_w}}"
        for grade in grade_full:
            cnt_a = int((use_a[mode]["quality"] == grade).sum()) if len(use_a[mode]) > 0 else 0
            cnt_b = int((use_b[mode]["quality"] == grade).sum()) if len(use_b[mode]) > 0 else 0
            cell = f"{cnt_a} ({cnt_b})"
            line += f"  {cell:>{col_w}}"
        logger.info(line)

    logger.info("=" * len(header))
    logger.info("")

    # ── [3/3] Grade transition matrix ──
    logger.info("[3/3] Grade transition matrix")
    grade_disp = ["A", "B", "C", "D", "No solution"]
    GRADE_RANK = {g: i for i, g in enumerate(grade_disp)}
    n_g = len(grade_disp)
    n_modes = len(PLOTPLOT_MODES)

    fig, axes = plt.subplots(1, n_modes, figsize=(max(4.5 * n_modes, 5), 5), dpi=200, squeeze=False)
    axes_row = axes[0]
    for ax, mode in zip(axes_row, PLOTPLOT_MODES):
        gmap_a = {row["event_id"]: row.get("quality", "No solution")
                  for row in dfs_best_a[mode].to_dict("records")}
        gmap_b = {row["event_id"]: row.get("quality", "No solution")
                  for row in dfs_best_b[mode].to_dict("records")}
        matrix = np.zeros((n_g, n_g), dtype=int)
        for eid in event_ids:
            ga = gmap_a.get(eid, "No solution")
            gb = gmap_b.get(eid, "No solution")
            matrix[GRADE_RANK.get(ga, n_g - 1), GRADE_RANK.get(gb, n_g - 1)] += 1
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_xticks(range(n_g)); ax.set_yticks(range(n_g))
        ax.set_xticklabels(grade_disp, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(grade_disp, fontsize=8)
        ax.set_xlabel(f"Grade in {dir_b.name}", fontsize=9)
        ax.set_ylabel(f"Grade in {dir_a.name}", fontsize=9)
        ax.set_title(PLOT_MODES[mode]["label"], fontsize=10)
        vmax = matrix.max() or 1
        for i in range(n_g):
            for j in range(n_g):
                if matrix[i, j] > 0:
                    txt_color = "white" if matrix[i, j] > 0.65 * vmax else "black"
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            fontsize=8, color=txt_color)
    fig.suptitle(f"Grade Transition: {dir_a.name} → {dir_b.name}  (best candidate)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / "grade_transition_matrix.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(COMPARE_DIR / "grade_transition_matrix.png")))
    logger.info(f"  → {COMPARE_DIR / 'grade_transition_matrix.png'}")

    logger.info("=" * 60)
    logger.info()
    logger.info(f"  Done  ({_time.time() - t0:.1f} s)")
    logger.info()
    logger.info("=" * 60)
    logger.close()
