"""sparse_pairs — Sparse pair selection and stability check for large-scale MCCC.

Three pair-selection strategies combined (union):
1. K nearest neighbors (spatial distance)
2. Top m fraction xcorr (fast cross-correlation quality)
3. L remote connections per event (global connectivity)

Plus automatic bridge pairs for disconnected components and SVD-based
stability check (:func:`stability_check`) for convergence testing.

Usage:
    rankings = precompute_pair_rankings(catalog_df, fft_dir, event_ids, ...)
    pairs = select_from_rankings(rankings, k=10, m=0.1, L=5)
    # Expand: just increase k/m/L, no recomputation
    pairs_B = select_from_rankings(rankings, k=12, m=0.12, L=7)
"""
from __future__ import annotations

import random
import numpy as np


def precompute_pair_rankings(
    catalog_df,
    fft_dir,
    event_ids,
    dt,
    maxlag=0.5,
    xcorr_subsample=10,
    device="cpu",
    remote_seed=None,
):
    """Compute full rankings once. Returns a dict reused by select_from_rankings.

    Parameters
    ----------
    remote_seed : int or None
        Seed for deterministic remote pair ordering. If None, random.

    Returns
    -------
    dict with keys:
        knn_ranked : list[list[int]] — per-event neighbors sorted by distance
        xcorr_ranked : list[(i, j)] — all pairs sorted by xcorr quality (descending)
        remote_ranked : list[list[int]] — per-event shuffled candidates
        n_ev : int
        n_total : int — n_ev*(n_ev-1)//2
    """
    n_ev = len(event_ids)

    # 1. KNN: full distance ranking per event
    knn_ranked = _knn_full_ranking(catalog_df)

    # 2. Xcorr: all pairs ranked by quality
    xcorr_ranked = _xcorr_full_ranking(fft_dir, event_ids, dt, maxlag,
                                        xcorr_subsample, device)

    # 3. Remote: deterministic shuffle per event
    remote_ranked = _remote_full_ranking(n_ev, remote_seed)

    return {
        "knn_ranked": knn_ranked,
        "xcorr_ranked": xcorr_ranked,
        "remote_ranked": remote_ranked,
        "n_ev": n_ev,
        "n_total": n_ev * (n_ev - 1) // 2,
    }


def select_from_rankings(rankings, k_neighbors=10, top_xcorr_frac=0.1, n_remote=3):
    """Select pairs by truncating precomputed rankings. B ⊇ A guaranteed.

    Parameters
    ----------
    rankings : dict from precompute_pair_rankings()
    k_neighbors : int — take top-k nearest neighbors per event
    top_xcorr_frac : float — take top fraction of xcorr-ranked pairs
    n_remote : int — take top-L remote connections per event

    Returns sorted list of (i, j) tuples with i < j.
    """
    n_ev = rankings["n_ev"]

    # 1. KNN: take first k neighbors per event
    knn = set()
    for i, neighbors in enumerate(rankings["knn_ranked"]):
        for j in neighbors[:k_neighbors]:
            knn.add((min(i, j), max(i, j)))

    # 2. Xcorr: take top fraction
    n_keep = max(1, int(len(rankings["xcorr_ranked"]) * top_xcorr_frac))
    xcorr = set(rankings["xcorr_ranked"][:n_keep])

    # 3. Remote: take first L per event
    remote = set()
    for i, others in enumerate(rankings["remote_ranked"]):
        for j in others[:min(n_remote, len(others))]:
            remote.add((min(i, j), max(i, j)))

    all_pairs = sorted(knn | xcorr | remote)
    _check_and_bridge(all_pairs, n_ev)

    return all_pairs


# ═══════════════════════════════════════════════════════════════════════════
#  Internal ranking functions
# ═══════════════════════════════════════════════════════════════════════════

def _knn_full_ranking(catalog_df):
    """Return per-event list of all other events sorted by distance."""
    from scipy.spatial import cKDTree

    coords = catalog_df[["longitude", "latitude"]].values
    n_ev = len(coords)
    tree = cKDTree(coords)
    # Query all neighbors (k=n_ev includes self at index 0)
    _, indices = tree.query(coords, k=n_ev)
    # indices[i] = [self, nearest, 2nd nearest, ...], skip self
    return [indices[i, 1:].tolist() for i in range(n_ev)]


def _xcorr_full_ranking(fft_dir, event_ids, dt, maxlag, subsample, device):
    """Return all pairs sorted by xcorr quality (descending)."""
    import torch
    from dasfm.picking.mccc import xcorr_from_freq
    from dasfm.io.das_fft import load_das_fft_single

    n_ev = len(event_ids)
    use_gpu = isinstance(device, str) and device.startswith("cuda")

    ffts = {}
    for i, eid in enumerate(event_ids):
        fft = load_das_fft_single(fft_dir, eid)
        if fft is not None:
            t = torch.as_tensor(fft[::subsample])
            ffts[i] = t.to(device) if use_gpu else t

    scores = []
    for i in range(n_ev):
        for j in range(i + 1, n_ev):
            if i not in ffts or j not in ffts:
                continue
            xcor, _ = xcorr_from_freq(ffts[i], ffts[j], dt, maxlag=maxlag)
            cc_max = xcor.abs().max(dim=-1).values.mean().item()
            scores.append((cc_max, i, j))

    scores.sort(reverse=True)
    return [(i, j) for (_, i, j) in scores]


def _remote_full_ranking(n_ev, seed=None):
    """Return per-event deterministic shuffle of all other events."""
    if seed is None:
        seed = random.randint(0, 2**31)
    ranked = []
    for i in range(n_ev):
        others = [j for j in range(n_ev) if j != i]
        rng = random.Random(seed + i)
        rng.shuffle(others)
        ranked.append(others)
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
#  Connectivity check
# ═══════════════════════════════════════════════════════════════════════════

def _check_and_bridge(pairs, n_ev):
    """Check graph connectivity; add bridge pairs if disconnected."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if not pairs:
        return
    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_ev, n_ev))
    adj = adj + adj.T
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp > 1:
        for c in range(1, n_comp):
            i = int(np.where(labels == 0)[0][0])
            j = int(np.where(labels == c)[0][0])
            pairs.append((min(i, j), max(i, j)))


# ═══════════════════════════════════════════════════════════════════════════
#  Adaptive expansion (used by sparse iteration loop in mccc_context)
# ═══════════════════════════════════════════════════════════════════════════

def expand_pairs(rankings, pairs_A, k_neighbors, top_xcorr_frac, n_remote, logger):
    """Expand sparse pair set with adaptive delta targeting 1-2% of total pairs.

    Uses precomputed rankings — just re-truncates, no recomputation.
    Controls len(B) - len(A) to be 1-2% of total pairs.

    Returns (new_B, k_B, m_B, L_B)
    """
    n_total = rankings["n_total"]
    target_lo = int(n_total * 0.01)   # 1%
    target_hi = int(n_total * 0.02)   # 2%
    target_mid = (target_lo + target_hi) / 2
    n_A = len(pairs_A)

    delta_k = 1
    delta_m = 0.001
    delta_l = 1
    best = None  # (distance, new_B_set, k, m, L)

    for attempt in range(10):
        new_k = k_neighbors + delta_k
        new_m = min(top_xcorr_frac + delta_m, 1.0)
        new_l = n_remote + delta_l

        new_B = set(select_from_rankings(
            rankings, k_neighbors=new_k,
            top_xcorr_frac=new_m, n_remote=new_l,
        ))

        n_diff = len(new_B) - n_A

        dist = abs(n_diff - target_mid)
        if best is None or dist < best[0]:
            best = (dist, new_B, new_k, new_m, new_l)

        if target_lo <= n_diff <= target_hi:
            logger.info(f"    expand: B-A={n_diff} ({n_diff/n_total*100:.1f}% of total)")
            return new_B, new_k, new_m, new_l
        elif n_diff < target_lo:
            delta_k = max(1, delta_k * 2)
            delta_m = delta_m * 2
            delta_l = max(1, delta_l * 2)
        else:
            delta_k = max(1, delta_k // 2)
            delta_m = max(0.001, delta_m / 2)
            delta_l = max(1, delta_l // 2)

    # Use best attempt
    _, new_B, new_k, new_m, new_l = best
    n_diff = len(new_B) - n_A
    logger.info(f"    expand: B-A={n_diff} ({n_diff/n_total*100:.1f}% of total) "
                f"(best of {attempt+1} attempts)")
    return new_B, new_k, new_m, new_l


# ═══════════════════════════════════════════════════════════════════════════
#  SVD-based stability check for sparse MCCC convergence
# ═══════════════════════════════════════════════════════════════════════════

def stability_check(Ckij, Skij, pairs_A: set, ctx):
    """Compare SVD on the full set vs SVD on subset pairs_A.

    Builds a temporary mask ``(n_ev, n_ev)`` for ``pairs_A``, applies it
    element-wise to ``Ckij/Skij`` to get the subset matrices, runs SVD on
    both, then compares signs of the resulting ``Pkic`` columns.

    The "global +/-1 flip" ambiguity in SVD is handled by ``max(raw, 100-raw)``
    so a pair_A run that comes out fully sign-flipped from the full run still
    counts as 100% match.

    Parameters
    ----------
    Ckij, Skij : np.ndarray (n_ch, n_ev, n_ev)
    pairs_A : set of (int, int)
        Subset of event pairs to compare against.
    ctx : Step2bContext
        Needs ``n_ev``, ``stability_threshold``, ``logger``.

    Returns
    -------
    (passed, sign_match_pct, res_full)
        ``res_full`` is the SVD result on the full Ckij/Skij — postprocess
        can reuse this instead of running SVD again.
    """
    n_ev = ctx.n_ev
    logger = ctx.logger

    mask = np.eye(n_ev, dtype=bool)
    for (i, j) in pairs_A:
        mask[i, j] = True
        mask[j, i] = True
    mask3 = mask[None, :, :]

    Ckij_A = Ckij * mask3
    Skij_A = Skij * mask3

    from dasfm.picking.polarity_svd import solve_polarity_svd
    n_pairs_full = int(((Ckij[0] != 0).sum() - n_ev) // 2)
    logger.info(f"  Stability check:")
    logger.info(f"    Group A: {len(pairs_A)} pairs")
    logger.info(f"    Full   : {n_pairs_full} pairs")
    logger.info(f"    SVD group A...")
    res_A = solve_polarity_svd(Ckij_A, Skij_A)
    logger.info(f"    SVD full...")
    res_full = solve_polarity_svd(Ckij, Skij)

    del Ckij_A, Skij_A, mask3, mask

    raw_match  = (np.sign(res_A["Pkic"]) == np.sign(res_full["Pkic"])).mean() * 100
    sign_match = max(raw_match, 100.0 - raw_match)
    passed     = sign_match >= ctx.stability_threshold * 100

    flip_note = " → flipped" if raw_match < 50 else ""
    logger.info(
        f"    Result: sign_match={sign_match:.1f}% (raw={raw_match:.1f}%{flip_note})  "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    return passed, sign_match, res_full
