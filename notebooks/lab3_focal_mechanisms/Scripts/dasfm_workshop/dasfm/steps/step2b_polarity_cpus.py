"""step2b_polarity_cpus — Multi-CPU MCCC backend (fork Pool, block-pair parallel).

Defines a closure ``_run_mccc(pair_set, Ckij, Skij)`` that dispatches block-pair
work to a fork ``multiprocessing.Pool`` of ``compute_block_pair_kernel`` workers.
Each worker returns a list of ``(gi, gj, si, cc)`` tuples; the main process
streams them into the persistent dense ``Ckij/Skij`` matrices.
"""
from __future__ import annotations

import multiprocessing
from tqdm import tqdm

from dasfm.picking.mccc_kernel import compute_block_pair_kernel, BlockPairTask
from dasfm.picking.mccc_context import setup_context, run_with_iteration
from dasfm.picking.polarity_postprocess import postprocess
from dasfm.utils.memory import estimate_step2b_vram_cpus


def _init_pool_worker():
    """Pool initializer: limit thread pools, set torch threads to 1.

    Runs once per fork child.  Ensures math libraries don't oversubscribe
    when many workers run concurrently.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _block_events(block_id: int, n_blocks: int, n_ev: int) -> list:
    """Even-ish split of [0, n_ev) into n_blocks chunks (remainder spread on the left)."""
    base = n_ev // n_blocks
    rem  = n_ev % n_blocks
    if block_id < rem:
        start = block_id * (base + 1)
        end   = start + base + 1
    else:
        start = rem * (base + 1) + (block_id - rem) * base
        end   = start + base
    return list(range(start, end))


def run(**kwargs):
    """Entry point — called by the dispatcher with all 25 user kwargs."""
    ctx = setup_context(
        mode_label=f"{int(kwargs.get('num_cpu_workers', 1))} CPU workers",
        **kwargs,
    )
    logger = ctx.logger
    n_ev, n_ch = ctx.n_ev, ctx.n_ch
    W = ctx.num_cpu_workers

    # ── Memory budget + block sizing (cpus-specific estimator) ────────────
    mem = estimate_step2b_vram_cpus(
        n_ev=n_ev, n_ch=n_ch, nfast=ctx.nfast,
        use_lr=ctx.use_lr, num_workers=W,
    )
    n_blocks_mem = mem["n_blocks"]

    # Parallelism constraint: n_blocks*(n_blocks+1)/2 ≥ W block-pairs so every
    # worker has at least one block-pair to chew on.
    n_blocks_par = 1
    while n_blocks_par * (n_blocks_par + 1) // 2 < W:
        n_blocks_par += 1
    n_blocks = min(max(n_blocks_mem, n_blocks_par, 1), n_ev)

    block_pairs = [(k, l) for k in range(n_blocks) for l in range(k, n_blocks)]
    n_block_pairs = len(block_pairs)

    logger.info(
        f"  Estimated RAM peak ({W} workers × "
        f"{mem['per_worker_bytes']/1e9:.2f} GB): "
        f"{mem['peak_bytes']/1e9:.2f} GB  /  "
        f"{mem['available_bytes']/1e9:.2f} GB available"
    )
    logger.info(f"  {'Blocks':<12}: {n_blocks}  "
                f"(memory: {n_blocks_mem}, parallelism: {n_blocks_par})")
    logger.info(f"  {'Block pairs':<12}: {n_block_pairs}")
    logger.info(f"  {'Workers':<12}: {W}")

    # ── Pool created AFTER setup_context (which may have imported torch
    #    via sparse precompute, but only with device='cpu' so no CUDA init).
    #    Spawning the Pool here is fork-safe as long as no CUDA context exists
    #    in the parent process.
    pool = multiprocessing.Pool(W, initializer=_init_pool_worker)

    try:
        # ── Backend-specific MCCC runner ──────────────────────────────────
        def _run_mccc(pair_set, Ckij, Skij):
            """Build BlockPairTasks for ``pair_set``, dispatch to pool, stream results."""
            allowed = set(pair_set) if pair_set is not None else None
            tasks = [
                BlockPairTask(
                    block_k=bk, block_l=bl,
                    ev_indices_k=_block_events(bk, n_blocks, n_ev),
                    ev_indices_l=_block_events(bl, n_blocks, n_ev),
                    event_ids=ctx.event_ids,
                    fft_dir=str(ctx.fft_dir),
                    dt=ctx.dt,
                    maxlag=ctx.mccc_max_lag_sec,
                    mccc_maxwin=ctx.mccc_maxwin,
                    mccc_damp=ctx.mccc_damp,
                    max_shift=ctx.mccc_max_shift,
                    polarity_method=ctx.polarity_method,
                    has_lr=ctx.has_lr,
                    n_ch=n_ch,
                    device="cpu",
                    smooth_window=ctx.polarity_smooth_window,
                    allowed_pairs=allowed,
                )
                for (bk, bl) in block_pairs
            ]

            pbar = tqdm(total=len(tasks), desc="MCCC blocks",
                        unit="bp", leave=True)
            # Stream results into the main process's Ckij/Skij — workers
            # return Python lists, the main process is the only writer.
            for results in pool.imap_unordered(compute_block_pair_kernel, tasks):
                for (gi, gj, si, cc) in results:
                    target = Ckij if si == 0 else Skij
                    target[:, gi, gj] = cc
                    target[:, gj, gi] = cc
                pbar.update(1)
            pbar.close()

        Ckij, Skij, svd_result = run_with_iteration(ctx, _run_mccc)
    finally:
        pool.close()
        pool.join()

    postprocess(ctx, Ckij, Skij, svd_result)
