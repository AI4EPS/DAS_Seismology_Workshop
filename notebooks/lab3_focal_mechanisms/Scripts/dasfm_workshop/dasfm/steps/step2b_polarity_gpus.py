"""step2b_polarity_gpus — Multi-GPU MCCC backend (ThreadPoolExecutor, per-GPU thread).

Defines a closure ``_run_mccc(pair_set, Ckij, Skij)`` that round-robin distributes
block pairs across ``num_gpu`` GPU threads.  All threads share the main process's
dense ``Ckij/Skij`` (no IPC needed); writes are serialised by ``threading.Lock``.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dasfm.picking.mccc_kernel import compute_block_pair_kernel, BlockPairTask
from dasfm.picking.mccc_context import setup_context, run_with_iteration
from dasfm.picking.polarity_postprocess import postprocess
from dasfm.steps.step2b_polarity_cpus import _block_events   # share even-split
from dasfm.utils.memory import estimate_step2b_vram


def run(**kwargs):
    """Entry point — called by the dispatcher with all 25 user kwargs."""
    ctx = setup_context(
        mode_label=f"{int(kwargs.get('num_gpu', 0)) or len(kwargs.get('device', []))} GPUs",
        **kwargs,
    )
    logger = ctx.logger
    n_ev, n_ch = ctx.n_ev, ctx.n_ch
    W = ctx.num_gpu

    # ── Memory budget — each GPU thread is its own single-GPU view, so the
    #    serial estimator applies to one card.  Pass device=gpu_devices[0]
    #    assuming all cards in the pool are the same model.
    mem = estimate_step2b_vram(
        n_ev=n_ev, n_ch=n_ch, nfast=ctx.nfast,
        use_lr=ctx.use_lr, use_gpu=True, device=ctx.gpu_devices[0],
    )
    n_blocks_mem = mem["n_blocks"]

    # Parallelism constraint: n_blocks*(n_blocks+1)/2 ≥ W block-pairs.
    n_blocks_par = 1
    while n_blocks_par * (n_blocks_par + 1) // 2 < W:
        n_blocks_par += 1
    n_blocks = min(max(n_blocks_mem, n_blocks_par, 1), n_ev)

    block_pairs = [(k, l) for k in range(n_blocks) for l in range(k, n_blocks)]
    n_block_pairs = len(block_pairs)

    logger.info(
        f"  Estimated VRAM peak (per GPU): "
        f"{mem['peak_bytes']/1e9:.2f} GB  /  "
        f"{mem['available_bytes']/1e9:.2f} GB available"
    )
    logger.info(f"  {'Blocks':<12}: {n_blocks}  "
                f"(memory: {n_blocks_mem}, parallelism: {n_blocks_par})")
    logger.info(f"  {'Block pairs':<12}: {n_block_pairs}")
    logger.info(f"  {'Workers':<12}: {W} GPUs")

    # ── Backend-specific MCCC runner ──────────────────────────────────────
    def _run_mccc(pair_set, Ckij, Skij):
        """Round-robin block_pairs across GPUs, stream results into Ckij/Skij."""
        allowed = set(pair_set) if pair_set is not None else None

        # Round-robin distribute block_pairs across num_gpu GPUs
        gpu_tasks = [[] for _ in range(W)]
        for i, (bk, bl) in enumerate(block_pairs):
            gpu_tasks[i % W].append((bk, bl))

        write_lock = threading.Lock()
        pbar = tqdm(total=n_block_pairs, desc="MCCC blocks",
                    unit="bp", leave=True)

        def _gpu_worker(gpu_id: int, gpu_dev: str, tasks: list):
            """One thread per GPU.  Computes block-pairs and writes to Ckij/Skij.

            Threads share Ckij/Skij directly (no IPC); ``write_lock`` makes
            the per-block-pair flush atomic.  Different (gi, gj) writes never
            overlap so the lock mainly serialises tqdm updates and protects
            against any non-thread-safe numpy quirk.
            """
            for (bk, bl) in tasks:
                task = BlockPairTask(
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
                    device=gpu_dev,
                    smooth_window=ctx.polarity_smooth_window,
                    allowed_pairs=allowed,
                )
                results = compute_block_pair_kernel(task)
                with write_lock:
                    for (gi, gj, si, cc) in results:
                        target = Ckij if si == 0 else Skij
                        target[:, gi, gj] = cc
                        target[:, gj, gi] = cc
                    pbar.update(1)

        with ThreadPoolExecutor(max_workers=W) as ex:
            futs = [
                ex.submit(_gpu_worker, gi, ctx.gpu_devices[gi], gpu_tasks[gi])
                for gi in range(W)
            ]
            for f in futs:
                f.result()
        pbar.close()

    Ckij, Skij, svd_result = run_with_iteration(ctx, _run_mccc)
    postprocess(ctx, Ckij, Skij, svd_result)
