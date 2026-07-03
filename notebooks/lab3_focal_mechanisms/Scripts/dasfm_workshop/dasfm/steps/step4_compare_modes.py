"""step4_compare_modes — Compare quality grades between two inversion modes."""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from dasfm.utils.step_utils import Logger, resolve_path, parse_inversion_types
from dasfm.utils.result_analysis import (
    PLOT_MODES, load_results, load_grades,
)
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.result_io import validate_result_dir


def run(
    project_dir="",
    event_catalog="",
    mode_a="",
    mode_b="",
    result_dir=None,
    compute_uncertainty=False,
    show_plots=False):
    """Compare quality grades between two inversion modes within one result dir.

    Parameters
    ----------
    mode_a, mode_b : str
        Two inversion mode strings (e.g. "sta_pol", "sta_pol + das_pol + das_sp").
    result_dir : str or None
        Result directory (e.g. "result_try1"). None = auto-select.
    Outputs
    -------
        {result_dir}/summary_figs/mode_transition_{mode_a}_vs_{mode_b}.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog, "mode_a": mode_a, "mode_b": mode_b}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    import matplotlib
    import matplotlib.pyplot as plt

    root = Path(project_dir).resolve()
    logger = Logger("step4_compare_modes", log_dir=str(root / "logs"))
    grades_df = load_grades(root)

    CAT_FILE = resolve_path(event_catalog, root)

    # Resolve mode names
    modes_a = parse_inversion_types([mode_a])
    modes_b = parse_inversion_types([mode_b])
    if len(modes_a) != 1 or len(modes_b) != 1:
        raise ValueError(f"Expected one mode each, got {modes_a} and {modes_b}")
    ma = modes_a[0]
    mb = modes_b[0]
    if ma not in PLOT_MODES:
        raise ValueError(f"Unknown mode: {ma}")
    if mb not in PLOT_MODES:
        raise ValueError(f"Unknown mode: {mb}")

    # Resolve result dir
    if result_dir is not None:
        result_root = resolve_path(result_dir, root)
    else:
        from dasfm.utils.result_analysis import select_result_dir
        prefix = "result_uncert" if compute_uncertainty else "result"
        result_root = select_result_dir(root, prefix)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(CAT_FILE)
    validate_result_dir(result_root)

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step4_compare_modes — Compare two inversion modes")
    logger.info()
    logger.info("=" * 60)
    label_a = PLOT_MODES[ma]["label"]
    label_b = PLOT_MODES[mb]["label"]
    logger.info(f"  {'Mode A':<12}: {label_a}")
    logger.info(f"  {'Mode B':<12}: {label_b}")
    logger.info(f"  {'Result dir':<12}: {result_root}")

    FIG_DIR = result_root / "summary_figs"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    catalog = pd.read_csv(CAT_FILE)
    event_ids = [str(eid) for eid in catalog["event_id"].values]
    n_ev = len(event_ids)

    dfs_best, dfs_all = load_results(result_root, event_ids, [ma, mb], grades_df)

    # Transition matrix always uses best candidate (one grade per event)
    recs_a = {r["event_id"]: r for r in dfs_best[ma].to_dict("records")}
    recs_b = {r["event_id"]: r for r in dfs_best[mb].to_dict("records")}
    gmap_a = {eid: r.get("quality", "No solution") for eid, r in recs_a.items()}
    gmap_b = {eid: r.get("quality", "No solution") for eid, r in recs_b.items()}

    # Grade transition matrix
    grade_disp = ["A", "B", "C", "D", "No solution"]
    GRADE_RANK = {g: i for i, g in enumerate(grade_disp)}
    n_g = len(grade_disp)

    matrix = np.zeros((n_g, n_g), dtype=int)
    upgraded = degraded = same = 0
    upgraded_events = []
    degraded_events = []
    for eid in event_ids:
        ga = gmap_a.get(eid, "No solution")
        gb = gmap_b.get(eid, "No solution")
        ri = GRADE_RANK.get(ga, n_g - 1)
        rj = GRADE_RANK.get(gb, n_g - 1)
        matrix[ri, rj] += 1
        if rj < ri:
            upgraded += 1
            upgraded_events.append((eid, ga, gb))
        elif rj > ri:
            degraded += 1
            degraded_events.append((eid, ga, gb))
        else:
            same += 1

    # Log summary
    logger.info(f"\n  Events: {n_ev}")
    logger.info(f"  Upgraded (A better in {label_b}): {upgraded}")
    logger.info(f"  Same:                              {same}")
    logger.info(f"  Degraded (A better in {label_a}): {degraded}")

    if upgraded_events:
        logger.info(f"\n  Upgraded events ({label_a} → {label_b}):")
        for eid, ga, gb in upgraded_events:
            logger.info(f"    {eid}: {ga} → {gb}")
    if degraded_events:
        logger.info(f"\n  Degraded events ({label_a} → {label_b}):")
        for eid, ga, gb in degraded_events:
            logger.info(f"    {eid}: {ga} → {gb}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_xticks(range(n_g)); ax.set_yticks(range(n_g))
    ax.set_xticklabels(grade_disp, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(grade_disp, fontsize=10)
    ax.set_ylabel(label_a, fontsize=12)
    ax.set_xlabel(label_b, fontsize=12)
    ax.set_title(f"Grade Transition: {label_a} → {label_b}\n"
                 f"(↑{upgraded}  ={same}  ↓{degraded})", fontsize=12)
    vmax = matrix.max() or 1
    for i in range(n_g):
        for j in range(n_g):
            if matrix[i, j] > 0:
                txt_color = "white" if matrix[i, j] > 0.65 * vmax else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=10, color=txt_color)
    fig.tight_layout()
    fname = f"mode_transition_{ma}_vs_{mb}.png"
    fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(FIG_DIR / fname)))
    logger.info(f"\n  → {FIG_DIR / fname}")

    # Export per-event TSV with per-metric grade strings
    from dasfm.utils.result_analysis import JOINTPLOT_MODES as _JM, DEFAULT_GRADES

    def _per_metric_grades(rec, mode):
        """Return a string like 'AABA' — one letter per active metric."""
        if not rec or not rec.get("has_solution", False):
            return "-"
        pol = rec.get("misfit_pol", 1.0)
        kag = rec.get("kagan_rms", 99.0)
        prob = rec.get("accept_ratio", 0.0)
        stdr_val = rec.get("stdr", 0.0)
        amp = rec.get("misfit_amp_rate", 0.0)
        if np.isnan(amp):
            amp = 0.0

        # Per-metric grade
        checks = [
            ("avg_misfit_max", pol, "le"),
            ("rms_uncertainty_max", kag, "le"),
            ("mech_prob_min", prob, "ge"),
            ("station_ratio_min", stdr_val, "ge"),
        ]
        if mode in _JM:
            checks.append(("amp_rate_max", amp, "le"))

        letters = []
        for thresh_key, val, direction in checks:
            g = "D"
            for grade in DEFAULT_GRADES.index:
                thresh = DEFAULT_GRADES.loc[grade, thresh_key]
                if (direction == "le" and val <= thresh) or (direction == "ge" and val >= thresh):
                    g = grade
                    break
            letters.append(g)
        return "".join(letters)

    # Build header: metric order for each mode
    def _metric_order(m):
        names = ["pol", "kag", "prob"]
        if m == "STA_pol":
            names.append("stdr")
        if m in _JM:
            names.append("sp")
        return "".join(names)

    rows = []
    for eid in event_ids:
        ra = recs_a.get(eid, {})
        rb = recs_b.get(eid, {})
        rows.append({
            "event_id": eid,
            ma: _per_metric_grades(ra, ma),
            mb: _per_metric_grades(rb, mb),
        })

    csv_name = f"mode_transition_{ma}_vs_{mb}.csv"
    csv_path = FIG_DIR / csv_name
    with open(csv_path, "w") as f:
        f.write(f"# {ma} columns: {_metric_order(ma)}\n")
        f.write(f"# {mb} columns: {_metric_order(mb)}\n")
        f.write(f"# pol=polarity_misfit, kag=kagan_uncertainty, prob=solution_prob, stdr=STDR, sp=sp_misfit_rate\n")
        f.write(f"# Grade per metric: A/B/C/D, -=No solution\n")
    pd.DataFrame(rows).to_csv(csv_path, index=False, mode="a")
    logger.info(f"  → {csv_path}")

    logger.info("=" * 60)
    logger.info(f"  Done  ({_time.time() - t0:.1f} s)")
    logger.info("=" * 60)
    logger.close()
