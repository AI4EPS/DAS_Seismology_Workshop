"""
Hierarchical clustering of focal mechanism solutions (no overlap, numpy).

Extracted from ytf_cluster.py (old_code). Converted from torch to numpy.
"""
from __future__ import annotations

import numpy as np
import scipy.cluster.hierarchy as sch
from . import moment_tensor as pmt
from dasfm.utils.step_utils import log_or_print
from .kagan import (
    sdr2kagan_pdist,
    dcm2kagan,
    get_weight_from_misfit,
    sdr_weighted_average,
    sdr_weighted_average_iter,
)


def cluster_solution_nooverlap(
    solution: dict,
    cluster_threshold: float = 45.0,
    kagan_threshold: float = 30.0,
    logger=None,
) -> list[dict]:
    """
    Hierarchical clustering with strict non-overlapping membership.

    Steps:
        1. Hierarchical clustering -> initial partitions.
        2. Refine center iteratively per partition.
        3. Strict filter: keep only members within kagan_threshold AND
           belonging to the original partition.

    Returns:
        list of candidate cluster dicts (one per surviving cluster, no
        sort, no top-5 cut).  Each candidate carries only geometric /
        statistical fields (``fm_mean``, ``kagan_rms``, ``angleN_rms``,
        ``angleS_rms``, ``accept_ratio``, ``stk/dip/rak/mask``).  Misfit
        fields (``misfit_pol0``, ``misfit_amp0``, etc.) are intentionally
        **not** computed here — they are filled in later by
        :func:`dasfm.inversion.repr_misfit.recompute_repr_misfits` with
        the true forward-model misfit of each cluster's representative
        ``fm_mean``, after which the candidate list is sorted and
        truncated to top-5.
    """
    stk = solution["stk"]
    dip = solution["dip"]
    rak = solution["rak"]

    # Misfits are used only as **weights** for the sdr_weighted_average below;
    # they are NOT written into the returned candidate dicts.  Final misfit
    # fields are filled in by recompute_repr_misfits using the true
    # forward-model misfit of fm_mean.  Use amplitude misfit when polarity is
    # unavailable (amp-only mode).
    misfit_pol = solution["misfit_pol"]
    misfit_amp = solution["misfit_amp"]
    misfit_sub = misfit_amp if np.isnan(misfit_pol).all() else misfit_pol
    accept_count = solution.get("accept_count")  # None for legacy / NMC=1

    nsol = stk.size
    if nsol == 0:
        return []

    # pairwise Kagan distance
    kagan_p = sdr2kagan_pdist(stk, dip, rak)

    if nsol == 1:
        iclust = np.array([1], dtype=int)
    else:
        Z = sch.linkage(kagan_p, method="average")
        iclust = sch.fcluster(Z, t=cluster_threshold, criterion="distance").astype(int)

    unique_cluster_id = np.sort(np.unique(iclust))

    # precompute DCMs
    dcm_all = np.asarray(pmt.sdr2dcm(stk, dip, rak), dtype=np.float32)

    solution_candidate_list = []

    log_or_print(logger, f"  cluster_nooverlap: {len(unique_cluster_id)} clusters", file_only=True)
    for cluster_id in unique_cluster_id:
        mask_cluster_member = iclust == cluster_id
        if np.count_nonzero(mask_cluster_member) < 5:
            continue

        stk_sub = stk[mask_cluster_member]
        dip_sub = dip[mask_cluster_member]
        rak_sub = rak[mask_cluster_member]

        weight_c = get_weight_from_misfit(misfit_sub[mask_cluster_member])
        nvec0, svec0, _, _, _ = sdr_weighted_average(stk_sub, dip_sub, rak_sub, weight=weight_c)
        stk0, dip0, rak0 = pmt.ns2sdr(nvec0, svec0)

        # coarse selection
        dcm0 = np.asarray(pmt.sdr2dcm(stk0, dip0, rak0), dtype=np.float32)
        if dcm0.ndim == 2:
            dcm0 = dcm0[np.newaxis]
        dcm0_exp = np.broadcast_to(dcm0, (nsol,) + dcm0.shape[1:])
        kagan_all_np = dcm2kagan(dcm0_exp, dcm_all)

        isel_k = (kagan_all_np < max(45.0, cluster_threshold)) & mask_cluster_member
        if not np.any(isel_k):
            continue

        # refinement
        weight_ref = get_weight_from_misfit(misfit_sub[isel_k])
        nvec0, svec0, _, _, _ = sdr_weighted_average_iter(
            stk[isel_k], dip[isel_k], rak[isel_k], weight=weight_ref, iter_max=3,
        )
        stk0, dip0, rak0 = pmt.ns2sdr(nvec0, svec0)

        dcm0 = np.asarray(pmt.sdr2dcm(stk0, dip0, rak0), dtype=np.float32)
        if dcm0.ndim == 2:
            dcm0 = dcm0[np.newaxis]
        dcm0_exp = np.broadcast_to(dcm0, (nsol,) + dcm0.shape[1:])
        kagan_all_np = dcm2kagan(dcm0_exp, dcm_all)

        # final selection
        isel_final = (kagan_all_np < kagan_threshold) & mask_cluster_member
        if not np.any(isel_final):
            continue

        # statistics — RMS Kagan angle between cluster average and ALL
        # accepted solutions (not just cluster members), measuring how
        # well the cluster average represents the full solution space.
        kagan_rms0 = float(np.sqrt(np.mean(kagan_all_np ** 2)))

        _, _, _, angle_N_stats, angle_S_stats = sdr_weighted_average_iter(
            stk[isel_final], dip[isel_final], rak[isel_final],
            weight=get_weight_from_misfit(misfit_sub[isel_final]),
            iter_max=1,
        )

        angleN_rms0 = (
            float(np.sqrt(np.mean(np.asarray(angle_N_stats) ** 2)))
            if angle_N_stats is not None and len(angle_N_stats) > 0
            else np.nan
        )
        angleS_rms0 = (
            float(np.sqrt(np.mean(np.asarray(angle_S_stats) ** 2)))
            if angle_S_stats is not None and len(angle_S_stats) > 0
            else np.nan
        )

        if accept_count is not None:
            # Weighted by MC trial acceptance count: mechanisms accepted by
            # more trials contribute more to the ratio.
            accept_ratio = float(np.sum(accept_count[isel_final]) / np.sum(accept_count))
        else:
            accept_ratio = float(np.count_nonzero(isel_final) / nsol)

        solution_candidate_list.append({
            "stk_mean": float(stk0),
            "dip_mean": float(dip0),
            "rak_mean": float(rak0),
            "fm_mean": [float(stk0), float(dip0), float(rak0)],
            "kagan_rms": kagan_rms0,
            "angleN_rms": angleN_rms0,
            "angleS_rms": angleS_rms0,
            "accept_ratio": accept_ratio,
            "stk": stk[isel_final].copy(),
            "dip": dip[isel_final].copy(),
            "rak": rak[isel_final].copy(),
            "mask": np.where(isel_final)[0],
        })

    return solution_candidate_list
