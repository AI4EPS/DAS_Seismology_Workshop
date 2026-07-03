"""step3_plot — Generate beachball plots from step3_invert results.

Reads .h5 files saved by step3_invert and generates beachball PNG figures.
All observation data (DAS/STA polarity, S/P ratios, azimuth, takeoff) is
embedded in the result files, so no raw input files are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from dasfm.utils.step_utils import (
    Logger, resolve_path, parse_inversion_types,
    JOINT_MODES, MODE_TITLES, MODE_INDEX,
)
from dasfm.io.result_io import load_inversion_result, validate_result_dir
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.utils.visualization import plot_beachball_on_ax


def _grid_for_n_modes(n_modes: int) -> tuple[int, int, tuple[float, float]]:
    """Pick (nrows, ncols, figsize) so all `n_modes` subplots are visible
    and the figure has no wasted columns.

    Each subplot is sized 4×4 inches.  Layout table:

        n_modes  →  (nrows, ncols)  figsize
        1            (1, 1)         ( 4,  4)
        2            (1, 2)         ( 8,  4)
        3            (1, 3)         (12,  4)
        4            (1, 4)         (16,  4)
        5–6          (2, 3)         (12,  8)
        7–8          (2, 4)         (16,  8)  ← legacy default
        9–12         (3, 4)         (16, 12)
        13+          (ceil(n/4), 4) (16, 4*nrows)

    Picking ncols=3 for 5-6 keeps the figure tighter than the
    one-size-fits-all "always 4 columns" rule (6 cells vs 8 cells for
    n=5).  Returning a non-degenerate grid keeps ``fig.suptitle``
    horizontally centered over the actual content instead of over a
    fixed 16-inch canvas.
    """
    if n_modes <= 0:
        return 1, 1, (4.0, 4.0)
    if n_modes <= 4:
        ncols = n_modes
    elif n_modes <= 6:
        ncols = 3
    else:
        ncols = 4
    nrows = (n_modes + ncols - 1) // ncols
    figsize = (4.0 * ncols, 4.0 * nrows)
    return nrows, ncols, figsize


def plot_single_event(args):
    """Plot beachball for a single event (worker function for multiprocessing)."""
    import matplotlib
    import matplotlib.pyplot as plt

    (eid, ev_sol_dir, fig_dir, inversion_types, grades_df) = args

    if not ev_sol_dir.exists():
        return {"eid": eid, "status": "skip", "n_plotted": 0}

    fig_dir.mkdir(parents=True, exist_ok=True)

    n_modes = len(inversion_types)
    nrows, ncols, figsize = _grid_for_n_modes(n_modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    axes_flat = np.atleast_1d(axes).ravel()
    n_plotted = 0

    for idx, mode in enumerate(inversion_types):
        h5_file = ev_sol_dir / f"{mode}.h5"
        ax = axes_flat[idx]

        if not h5_file.exists():
            ax.set_visible(False)
            continue

        data = load_inversion_result(h5_file)
        if data.get("skipped"):
            ax.set_visible(False)
            continue

        is_joint = mode in JOINT_MODES
        sol_key = "solution_joint" if is_joint else "solution_pol"
        cand_key = "candidates_joint" if is_joint else "candidates_pol"
        sol = data.get(sol_key, [])
        cand = data.get(cand_key, [])

        if not isinstance(sol, dict) or not cand:
            ax.set_visible(False)
            continue

        obs = data.get("obs", {})
        sta_kw = {}
        if "sta_az" in obs:
            sta_kw = dict(sta_az=obs["sta_az"], sta_takeoff=obs["sta_takeoff"],
                          sta_pol=obs["sta_pol"])
        sta_sp_kw = {}
        if "sta_sp_az" in obs:
            sta_sp_kw = dict(sta_sp_az=obs["sta_sp_az"],
                             sta_sp_takeoff=obs["sta_sp_takeoff"],
                             sta_sp=obs["sta_sp"])

        plot_beachball_on_ax(
            ax, sol, cand, mode,
            das_az=obs.get("das_az"), das_takeoff=obs.get("das_takeoff"),
            das_pol=obs.get("das_pol"), das_sp=obs.get("das_sp"),
            grades_df=grades_df,
            **sta_kw, **sta_sp_kw)
        n_plotted += 1

    for idx in range(n_modes, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Event: {eid}", fontsize=14, y=1.01)
    fig.tight_layout()
    combined_path = fig_dir / f"{eid}_all_modes.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"eid": eid, "status": "ok", "n_plotted": n_plotted}


def run(
    project_dir="",
    event_catalog="",
    inversion_types=None,
    result_dir="",
    fig_dir=None,
    inversion_type_index=None,
    num_cpu_workers=1,
    show_plots=False,
):
    """Generate beachball plots from step3_invert .h5 results.

    Parameters
    ----------
    result_dir : str
        Directory containing inv_sol/{event_id}/{mode}.h5 files.
    event_catalog : str
        Event catalog CSV (used only for event ID list).
    inversion_types : list[str]
        List of inversion modes to plot.
    fig_dir : str, optional
        Output directory for beachball plots (relative to project_dir).
        Default: ``{result_dir}/event_figs``.
    num_cpu_workers : int
        Number of CPU workers for parallel plotting (1 = serial).
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog, "inversion_types": inversion_types, "result_dir": result_dir}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    import pandas as pd

    root = Path(project_dir).resolve()
    logger = Logger("step3_plot", log_dir=str(root / "logs"))
    RESULT_ROOT = resolve_path(result_dir, root)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(resolve_path(event_catalog, root))
    validate_result_dir(RESULT_ROOT)

    from dasfm.utils.result_analysis import load_grades
    grades_df = load_grades(root)

    inversion_types = parse_inversion_types(inversion_types)
    inv_index = {**MODE_INDEX, **(inversion_type_index or {})}

    logger.info("=" * 60)
    logger.info(f"  step3_plot — Generate beachball plots")
    logger.info("=" * 60)
    logger.info(f"  Result dir: {RESULT_ROOT}")

    # Event list
    CAT_FILE = resolve_path(event_catalog, root)
    catalog = pd.read_csv(CAT_FILE)
    all_event_ids = [str(eid) for eid in catalog["event_id"].values]
    n_ev = len(all_event_ids)

    logger.info(f"  Events: {n_ev}")

    # ── Generate plots ──
    import matplotlib

    if fig_dir is not None:
        fig_dir = resolve_path(fig_dir, root)
    else:
        fig_dir = RESULT_ROOT / "event_figs"
    worker_args = [
        (eid, RESULT_ROOT / "inv_sol" / eid, fig_dir,
         inversion_types, grades_df)
        for eid in all_event_ids
    ]

    if num_cpu_workers > 1:
        import multiprocessing
        logger.info(f"  Parallel: {num_cpu_workers} workers (fork Pool)")
        with multiprocessing.Pool(num_cpu_workers) as pool:
            results = list(tqdm(
                pool.imap(plot_single_event, worker_args),
                total=n_ev, desc="plotting", unit="ev", leave=True))
    else:
        results = []
        for args in tqdm(worker_args, desc="plotting", unit="ev", leave=True):
            results.append(plot_single_event(args))

    n_plotted = sum(r["n_plotted"] for r in results)

    # Display first event's combined plot
    if show_plots:
        combined = fig_dir / f"{all_event_ids[0]}_all_modes.png"
        if combined.exists():
            from IPython.display import display, Image
            display(Image(filename=str(combined)))

    logger.info(f"  Generated {n_plotted} subplot plots across {n_ev} events")
    logger.info("=" * 60)
    logger.close()
