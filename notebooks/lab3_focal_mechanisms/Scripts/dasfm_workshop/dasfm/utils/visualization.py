"""
Focal mechanism beachball plots with polarity / amplitude markers.

Adapted from old_code/util_plot.py.
Uses inv_fm["kagan_rms"] (RMS Kagan angle between cluster average and all accepted solutions).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from obspy.imaging.beachball import beach

from ..inversion.tensor_utils import dict_to_numpy
from ..utils.step_utils import log_or_print


# ── helpers ──────────────────────────────────────────────────────────────────

def project_to_beachball(az_deg, takeoff_deg):
    """Equal-area lower-hemisphere projection (az, takeoff → x, y)."""
    az_rad = np.deg2rad(az_deg)
    takeoff_rad = np.deg2rad(takeoff_deg)

    # upper hemisphere → flip
    upper = takeoff_deg > 90
    if np.ndim(upper) == 0:
        if upper:
            az_rad = np.deg2rad((az_deg + 180) % 360)
            takeoff_rad = np.deg2rad(180 - takeoff_deg)
    else:
        az_rad = np.where(upper, np.deg2rad((az_deg + 180) % 360), az_rad)
        takeoff_rad = np.where(upper, np.deg2rad(180 - takeoff_deg), takeoff_rad)

    r = np.sqrt(2) * np.sin(takeoff_rad / 2.0)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)
    return x, y


def draw_background_solutions(ax, solution_dict, max_lines=200, plot_style="dasfm"):
    """Draw grey beachball outlines for all accepted solutions."""
    solution_np = dict_to_numpy(solution_dict)
    if solution_np.get("stk") is None:
        return solution_np
    c_stk = solution_np["stk"]
    c_dip = solution_np["dip"]
    c_rak = solution_np["rak"]

    if len(c_stk) <= max_lines:
        idx = np.arange(len(c_stk))
    else:
        idx = np.random.choice(len(c_stk), max_lines, replace=False)

    if plot_style == "skhash":
        lw, ec, al = 0.15, "0.5", 0.4
    else:  # dasfm
        lw, ec, al = 0.5, "gray", 0.5

    for i in idx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bball = beach(
                [c_stk[i], c_dip[i], c_rak[i]],
                xy=(0, 0), width=2, linewidth=lw,
                edgecolor=ec, alpha=al, nofill=True,
            )
        bball.set_zorder(0)
        ax.add_collection(bball)

    return solution_np


def draw_best_solution(ax, inv_fm, plot_style="dasfm"):
    """Draw the best-fit beachball (single cluster, legacy signature)."""
    draw_cluster_solutions(ax, [inv_fm], plot_style=plot_style)


def draw_cluster_solutions(ax, solution_list, plot_style="dasfm"):
    """Draw cluster-average beachballs.

    Parameters
    ----------
    plot_style : "skhash" | "dasfm"
        ``"skhash"``: all cluster averages drawn with decreasing opacity
        (SKHASH convention). Secondary clusters are visible as semi-
        transparent overlaps.
        ``"dasfm"``: only the best cluster drawn as a single grey-filled
        beachball with a red outline (original dasfm style).
    """
    if plot_style == "dasfm":
        # Original dasfm style: one filled beachball for the best cluster
        fm = solution_list[0]["fm_mean"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bball = beach(
                fm, xy=(0, 0), width=2,
                facecolor=[0.5, 0.5, 0.5], edgecolor="r",
                linewidth=2, alpha=0.5,
            )
        bball.set_zorder(10)
        ax.add_collection(bball)
        return

    # SKHASH style: all clusters with decreasing opacity
    n = len(solution_list)
    for rank, cand in enumerate(solution_list):
        fm = cand["fm_mean"]
        is_best = (rank == 0)
        alpha = min(1.0, 4.0 / (3.0 * n)) if n > 1 else 0.6
        if not is_best:
            alpha *= 0.6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bball = beach(
                fm, xy=(0, 0), width=2,
                facecolor="0.15" if is_best else "0.3",
                edgecolor="r" if is_best else "0.4",
                linewidth=2 if is_best else 0.8,
                alpha=alpha,
            )
        bball.set_zorder(10 - rank)
        ax.add_collection(bball)


def draw_sta_polarity(ax, sx, sy, sta_pos, plot_style="dasfm"):
    """Draw station polarity markers on the beachball.

    Parameters
    ----------
    plot_style : "skhash" | "dasfm"
        ``"skhash"``: black ``+`` for up, black open ``○`` for down
        (standard seismological convention).
        ``"dasfm"``: red filled ``▲`` for up, blue filled ``▼`` for down
        (original dasfm coloured style).
    """
    if plot_style == "skhash":
        if sta_pos.any():
            ax.scatter(sx[sta_pos], sy[sta_pos], c="k", marker="+", s=60,
                       linewidths=0.8, zorder=20)
        if (~sta_pos).any():
            ax.scatter(sx[~sta_pos], sy[~sta_pos], marker="o", s=50,
                       linewidths=0.5, zorder=20, edgecolors="k", facecolors="none")
    else:  # dasfm
        if sta_pos.any():
            ax.scatter(sx[sta_pos], sy[sta_pos], c="red", marker="^", s=30,
                       linewidth=0.5, zorder=20, edgecolors="black")
        if (~sta_pos).any():
            ax.scatter(sx[~sta_pos], sy[~sta_pos], c="blue", marker="v", s=30,
                       linewidth=0.5, zorder=20, edgecolors="black")


def draw_border(ax):
    """Draw the outer circle border and hide axis frame/ticks."""
    border = patches.Circle(
        (0, 0), radius=1.0,
        edgecolor="black", facecolor="none",
        linewidth=2, zorder=100,
    )
    ax.add_patch(border)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


# ── public API ───────────────────────────────────────────────────────────────

def save_pol_all_lines(
    solution_sum: dict,
    solution_list: list[dict],
    az: np.ndarray,
    takeoff: np.ndarray,
    pol: np.ndarray,
    title: str,
    ss: int,
    fig_dir: str | Path = "inv_figs",
    skip: int = 20,
    sta_az: np.ndarray | None = None,
    sta_takeoff: np.ndarray | None = None,
    sta_pol: np.ndarray | None = None,
    fig_name: str | None = None,
    show: bool = False,
    mode: str | None = None,
    grades_df=None,
    plot_style="dasfm",
    logger=None,
):
    """
    Plot polarity-only inversion result on a beachball.

    Parameters
    ----------
    solution_sum : dict   — full solution dict (stk/dip/rak tensors).
    solution_list : list  — candidate list from clustering.
    az, takeoff, pol  : 1-D arrays — DAS azimuth [deg], takeoff [deg], polarity sign.
    title : str           — plot title prefix.
    ss : int              — event index (0-based).
    fig_dir : path        — output directory.
    skip : int            — plot every N-th DAS channel.
    sta_az, sta_takeoff, sta_pol : optional 1-D arrays — station picks (plotted as triangles).
    fig_name : str        — override output filename (without extension).
    """
    nsol = len(solution_list)
    if nsol == 0:
        log_or_print(logger, f"[Warning] save_pol_all_lines: No solutions for {title} (S{ss+1}). Skip.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    solution_np = draw_background_solutions(ax, solution_sum, plot_style=plot_style)
    inv_fm = solution_list[0]
    draw_cluster_solutions(ax, solution_list, plot_style=plot_style)
    draw_border(ax)

    # plot DAS polarity markers (vectorised)
    idx = np.arange(0, len(az), skip)
    x, y = project_to_beachball(az[idx], takeoff[idx])
    pol_sign = np.sign(pol[idx])
    pos = pol_sign > 0
    if pos.any():
        ax.scatter(x[pos], y[pos], c="red", marker="o", s=10,
                   linewidth=0, edgecolors="none", zorder=10)
    if (~pos).any():
        ax.scatter(x[~pos], y[~pos], c="blue", marker="o", s=10,
                   linewidth=0, edgecolors="none", zorder=10)

    # plot station polarity markers
    if sta_az is not None and sta_takeoff is not None and sta_pol is not None:
        sx, sy = project_to_beachball(sta_az, sta_takeoff)
        sta_pos = np.sign(sta_pol) > 0
        draw_sta_polarity(ax, sx, sy, sta_pos, plot_style=plot_style)

    # title info
    prob_inv = inv_fm["accept_ratio"] * 100
    mm_pol = inv_fm["misfit_pol_ratio0"] * 100
    stdr = inv_fm["stdr"]
    rms_unc = inv_fm["kagan_rms"]

    from dasfm.utils.result_analysis import hash_quality_grade
    grade = hash_quality_grade(
        mm_pol / 100, rms_unc, stdr, inv_fm["accept_ratio"],
        grades_df=grades_df,
    )
    fig.suptitle(
        f"{title}\n"
        f"Quality: {grade}\n"
        f"prob: {prob_inv:.2f}%   pol: {mm_pol:.2f}%\n"
        f"kagan: {rms_unc:.2f}   stdr: {stdr:.2f}",
        fontsize=9, y=0.78,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.72])

    if fig_dir:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fname = fig_name if fig_name else f"S{ss+1}_{title.replace(' ', '_')}"
        _path = fig_dir / f"{fname}.png"
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        if show:
            from IPython.display import display, Image
            display(Image(filename=str(_path)))
        plt.close(fig)
    else:
        plt.show()


def save_amp_all_lines(
    solution_sum: dict,
    solution_list: list[dict],
    az: np.ndarray,
    takeoff: np.ndarray,
    sp_ratio: np.ndarray,
    title: str,
    ss: int,
    fig_dir: str | Path = "inv_figs",
    skip: int = 20,
    show: bool = False,
    mode: str | None = None,
    grades_df=None,
    plot_style="dasfm",
    logger=None,
):
    """
    Plot S/P amplitude inversion result — marker size ∝ S/P ratio.

    Parameters
    ----------
    sp_ratio : 1-D array — log10(S/P) per channel (used for marker sizing).
    """
    nsol = len(solution_list)
    if nsol == 0:
        log_or_print(logger, f"[Warning] save_amp_all_lines: No solutions for {title} (S{ss+1}). Skip.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    solution_np = draw_background_solutions(ax, solution_sum, plot_style=plot_style)
    inv_fm = solution_list[0]
    draw_cluster_solutions(ax, solution_list, plot_style=plot_style)
    draw_border(ax)

    # plot amplitude markers (size ∝ sp_ratio, vectorised)
    idx = np.arange(0, len(az), skip)
    x, y = project_to_beachball(az[idx], takeoff[idx])
    pp_size = (sp_ratio[idx] - np.nanmin(sp_ratio)) ** 5 + 1
    ax.scatter(x, y, c="blue", marker="o", s=pp_size, linewidth=2, zorder=10)

    # title info
    prob_inv = inv_fm["accept_ratio"] * 100
    mm_pol = inv_fm["misfit_pol_ratio0"] * 100
    mm_amp = inv_fm["misfit_amp_rate0"] * 100
    stdr = inv_fm["stdr"]
    rms_unc = inv_fm["kagan_rms"]

    from dasfm.utils.result_analysis import hash_quality_grade, JOINTPLOT_MODES
    is_joint = mode in JOINTPLOT_MODES if mode else True
    grade = hash_quality_grade(
        mm_pol / 100, rms_unc, stdr, inv_fm["accept_ratio"],
        amp_rate=mm_amp / 100,
        use_amp=is_joint, grades_df=grades_df,
    )
    line3 = f"kagan: {rms_unc:.2f}   stdr: {stdr:.2f}   sp: {mm_amp:.2f}%"
    fig.suptitle(
        f"{title}\n"
        f"Quality: {grade}\n"
        f"prob: {prob_inv:.2f}%   pol: {mm_pol:.2f}%\n"
        f"{line3}",
        fontsize=9, y=0.78,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.72])

    if fig_dir:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_title = title.replace(" ", "_")
        _path = fig_dir / f"{save_title}.png"
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        if show:
            from IPython.display import display, Image
            display(Image(filename=str(_path)))
        plt.close(fig)
    else:
        plt.show()


def save_joint_all_lines(
    solution_sum: dict,
    solution_list: list[dict],
    az: np.ndarray,
    takeoff: np.ndarray,
    pol: np.ndarray,
    sp_ratio: np.ndarray,
    title: str,
    ss: int,
    fig_dir: str | Path = "inv_figs",
    skip: int = 20,
    sta_az: np.ndarray | None = None,
    sta_takeoff: np.ndarray | None = None,
    sta_pol: np.ndarray | None = None,
    sta_sp_az: np.ndarray | None = None,
    sta_sp_takeoff: np.ndarray | None = None,
    sta_sp: np.ndarray | None = None,
    fig_name: str | None = None,
    show: bool = False,
    mode: str | None = None,
    grades_df=None,
    plot_style="dasfm",
    logger=None,
):
    """
    Plot joint polarity + amplitude inversion result.

    DAS markers: colour = polarity (red/blue circle), size ∝ S/P ratio.
    Station polarity markers (optional): larger triangles with black edge.
    Station S/P markers (optional): green hollow circles, size ∝ S/P ratio.

    Parameters
    ----------
    az, takeoff, pol, sp_ratio : 1-D arrays — DAS picks (must have same length).
    sta_az, sta_takeoff, sta_pol : optional 1-D arrays — station polarity picks.
    sta_sp_az, sta_sp_takeoff, sta_sp : optional 1-D arrays — station S/P ratio picks.
    fig_name : str        — override output filename (without extension).
    """
    nsol = len(solution_list)
    if nsol == 0:
        log_or_print(logger, f"[Warning] save_joint_all_lines: No solutions for {title} (S{ss+1}). Skip.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    solution_np = draw_background_solutions(ax, solution_sum, plot_style=plot_style)
    inv_fm = solution_list[0]
    draw_cluster_solutions(ax, solution_list, plot_style=plot_style)
    draw_border(ax)

    # plot DAS joint markers (size ∝ S/P ratio, vectorised)
    if len(az) > 0 and len(sp_ratio) > 0:
        idx = np.arange(0, len(az), skip)
        x, y = project_to_beachball(az[idx], takeoff[idx])
        has_pol = len(pol) > 0
        pp_size = (sp_ratio[idx] - np.nanmin(sp_ratio)) ** 4 * 1.5 + 1
        if not has_pol:
            ax.scatter(x, y, c="orange", marker="o", s=pp_size,
                       linewidth=0, edgecolors="none", zorder=10)
        else:
            pol_sign = np.sign(pol[idx])
            pos = pol_sign > 0
            if pos.any():
                ax.scatter(x[pos], y[pos], c="red", marker="o", s=pp_size[pos],
                           linewidth=0, edgecolors="none", zorder=10)
            if (~pos).any():
                ax.scatter(x[~pos], y[~pos], c="blue", marker="o", s=pp_size[~pos],
                           linewidth=0, edgecolors="none", zorder=10)

    # plot station polarity markers
    if sta_az is not None and sta_takeoff is not None and sta_pol is not None:
        sx, sy = project_to_beachball(sta_az, sta_takeoff)
        sta_pos = np.sign(sta_pol) > 0
        draw_sta_polarity(ax, sx, sy, sta_pos, plot_style=plot_style)

    # plot station S/P ratio markers (orange hollow circles, vectorised)
    if sta_sp_az is not None and sta_sp_takeoff is not None and sta_sp is not None:
        sp_x, sp_y = project_to_beachball(sta_sp_az, sta_sp_takeoff)
        sp_range = np.nanmax(sta_sp) - np.nanmin(sta_sp)
        sp_norm = (sta_sp - np.nanmin(sta_sp)) / max(sp_range, 1e-10)
        sp_size = (sp_norm ** 2 * 200) + 10
        ax.scatter(sp_x, sp_y, facecolors="none", edgecolors="orange",
                   marker="o", s=sp_size, linewidth=1.5, zorder=15)

    # title info
    prob_inv = inv_fm["accept_ratio"] * 100
    mm_pol = inv_fm["misfit_pol_ratio0"] * 100
    mm_amp = inv_fm["misfit_amp_rate0"] * 100
    stdr = inv_fm["stdr"]
    rms_unc = inv_fm["kagan_rms"]

    from dasfm.utils.result_analysis import hash_quality_grade, JOINTPLOT_MODES
    is_joint = mode in JOINTPLOT_MODES if mode else True
    grade = hash_quality_grade(
        mm_pol / 100, rms_unc, stdr, inv_fm["accept_ratio"],
        amp_rate=mm_amp / 100,
        use_amp=is_joint, grades_df=grades_df,
    )
    line3 = f"kagan: {rms_unc:.2f}   stdr: {stdr:.2f}   sp: {mm_amp:.2f}%"
    fig.suptitle(
        f"{title}\n"
        f"Quality: {grade}\n"
        f"prob: {prob_inv:.2f}%   pol: {mm_pol:.2f}%\n"
        f"{line3}",
        fontsize=9, y=0.78,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.72])

    if fig_dir:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fname = fig_name if fig_name else f"S{ss+1}_{title.replace(' ', '_')}"
        _path = fig_dir / f"{fname}.png"
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        if show:
            from IPython.display import display, Image
            display(Image(filename=str(_path)))
        plt.close(fig)
    else:
        plt.show()


# ── Combined multi-mode beachball on a single figure ────────────────────────

def plot_beachball_on_ax(ax, solution_sum, solution_list, mode,
                          das_az=None, das_takeoff=None, das_pol=None,
                          das_sp=None, skip=20,
                          sta_az=None, sta_takeoff=None, sta_pol=None,
                          sta_sp_az=None, sta_sp_takeoff=None, sta_sp=None,
                          grades_df=None, plot_style="dasfm"):
    """Draw a single beachball with markers on the given axes.

    Returns (grade, label) or (None, None) if no solutions.
    """
    from dasfm.utils.result_analysis import hash_quality_grade, JOINTPLOT_MODES
    from ..utils.step_utils import MODE_TITLES

    if not solution_list:
        ax.set_visible(False)
        return None, None

    solution_np = draw_background_solutions(ax, solution_sum, plot_style=plot_style)
    inv_fm = solution_list[0]
    draw_cluster_solutions(ax, solution_list, plot_style=plot_style)
    draw_border(ax)

    is_joint = mode in JOINTPLOT_MODES
    has_das_sp = das_sp is not None and len(das_sp) > 0

    # DAS markers (vectorised)
    if das_az is not None and len(das_az) > 0:
        idx = np.arange(0, len(das_az), skip)
        x, y = project_to_beachball(das_az[idx], das_takeoff[idx])
        if is_joint and has_das_sp:
            has_pol = das_pol is not None and len(das_pol) > 0
            pp_size = (das_sp[idx] - np.nanmin(das_sp)) ** 4 * 1.5 + 1
            if not has_pol:
                ax.scatter(x, y, c="orange", marker="o", s=pp_size,
                           linewidth=0, edgecolors="none", zorder=10)
            else:
                pol_sign = np.sign(das_pol[idx])
                pos = pol_sign > 0
                if pos.any():
                    ax.scatter(x[pos], y[pos], c="red", marker="o", s=pp_size[pos],
                               linewidth=0, edgecolors="none", zorder=10)
                if (~pos).any():
                    ax.scatter(x[~pos], y[~pos], c="blue", marker="o", s=pp_size[~pos],
                               linewidth=0, edgecolors="none", zorder=10)
        elif das_pol is not None and len(das_pol) > 0:
            pol_sign = np.sign(das_pol[idx])
            pos = pol_sign > 0
            if pos.any():
                ax.scatter(x[pos], y[pos], c="red", marker="o", s=10,
                           linewidth=0, edgecolors="none", zorder=10)
            if (~pos).any():
                ax.scatter(x[~pos], y[~pos], c="blue", marker="o", s=10,
                           linewidth=0, edgecolors="none", zorder=10)

    # STA polarity markers
    if sta_az is not None and sta_takeoff is not None and sta_pol is not None:
        sx, sy = project_to_beachball(sta_az, sta_takeoff)
        sta_pos = np.sign(sta_pol) > 0
        draw_sta_polarity(ax, sx, sy, sta_pos, plot_style=plot_style)

    # STA S/P markers (vectorised)
    if sta_sp_az is not None and sta_sp_takeoff is not None and sta_sp is not None:
        sp_x, sp_y = project_to_beachball(sta_sp_az, sta_sp_takeoff)
        sp_range = np.nanmax(sta_sp) - np.nanmin(sta_sp)
        sp_norm_val = (sta_sp - np.nanmin(sta_sp)) / max(sp_range, 1e-10)
        sp_size = (sp_norm_val ** 2 * 200) + 10
        ax.scatter(sp_x, sp_y, facecolors="none", edgecolors="orange",
                   marker="o", s=sp_size, linewidth=1.5, zorder=15)

    # Compute quality grade and title
    prob_inv = inv_fm["accept_ratio"] * 100
    mm_pol = inv_fm["misfit_pol_ratio0"] * 100
    stdr = inv_fm["stdr"]
    rms_unc = inv_fm["kagan_rms"]
    mm_amp = inv_fm.get("misfit_amp_rate0", 0.0) * 100

    grade = hash_quality_grade(
        mm_pol / 100, rms_unc, stdr, inv_fm["accept_ratio"],
        amp_rate=mm_amp / 100 if is_joint else 0.0,
        use_amp=is_joint, grades_df=grades_df,
    )

    label = MODE_TITLES.get(mode, mode)
    line1 = f"{label}  (Quality: {grade})"
    line2 = f"prob:{prob_inv:.0f}%  pol:{mm_pol:.1f}%"
    line3 = f"kagan:{rms_unc:.1f}  stdr:{stdr:.2f}"
    if is_joint:
        line3 += f"  sp:{mm_amp:.1f}%"
    ax.set_title(f"{line1}\n{line2}\n{line3}", fontsize=9, pad=4)

    return grade, label
