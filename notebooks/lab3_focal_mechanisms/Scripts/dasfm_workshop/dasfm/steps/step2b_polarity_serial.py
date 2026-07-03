"""step2b_polarity_serial — Serial MCCC backend (single CPU or single GPU).

Defines a closure ``_run_mccc(pair_set, Ckij, Skij)`` that streams MCCC results
directly into the dense ``Ckij/Skij`` matrices via block-pair α/β windowing.
The closure is handed off to :func:`run_with_iteration` which drives both the
sparse-iteration and non-sparse paths.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from dasfm.io.das_fft import load_das_fft_single, load_das_fft_lr_single
from dasfm.picking.mccc_kernel import compute_one_pair_into
from dasfm.picking.mccc_context import setup_context, run_with_iteration
from dasfm.picking.polarity_postprocess import postprocess
from dasfm.utils.memory import estimate_step2b_vram


def run(**kwargs):
    """Entry point — called by the dispatcher with all 25 user kwargs."""
    ctx = setup_context(mode_label="serial", **kwargs)
    logger = ctx.logger

    # ── Memory budget + block sizing ──────────────────────────────────────
    mem = estimate_step2b_vram(
        n_ev=ctx.n_ev, n_ch=ctx.n_ch, nfast=ctx.nfast,
        use_lr=ctx.use_lr, use_gpu=ctx.use_gpu, device=ctx.device,
    )
    block_size = mem["block_size"]
    n_blocks = mem["n_blocks"]
    logger.info(
        f"  Estimated {'VRAM' if ctx.use_gpu else 'RAM'} peak : "
        f"{mem['peak_bytes']/1e9:.2f} GB  /  "
        f"{mem['available_bytes']/1e9:.2f} GB available"
    )
    if block_size >= ctx.n_ev:
        logger.info(f"  Single block ({ctx.n_ev} events fit in cache)")
    else:
        logger.info(f"  Block streaming: {n_blocks} blocks × {block_size} events")

    # ── Pick the MCCCPicker class once ────────────────────────────────────
    if ctx.use_gpu:
        from dasfm.picking.mccc_gpu import MCCCPickerGPU as Picker
    else:
        from dasfm.picking.mccc import MCCCPicker as Picker

    # ── Backend-specific MCCC runner (closure captures ctx + block_size) ──
    def _run_mccc(pair_set, Ckij, Skij):
        """Stream MCCC for ``pair_set`` into ``Ckij/Skij`` via α/β block windowing.

        Iteration order keeps at most 2 blocks of FFT cache resident at once
        (the current α block plus the moving β block).
        """
        n_ev = ctx.n_ev
        use_lr = ctx.use_lr
        device = ctx.device

        # Bucket pairs by which (α, β) block-pair each (i, j) belongs to.
        pairs_by_block: dict = {}
        for (i, j) in pair_set:
            a, b = i // block_size, j // block_size
            if a > b:
                a, b = b, a
            pairs_by_block.setdefault((a, b), []).append((i, j))

        pbar = tqdm(total=len(pair_set), desc="MCCC pairs",
                    unit="pair", leave=True)
        freqs: dict = {}
        lr:    dict = {}

        def _load(idx):
            if idx in freqs:
                return
            f = load_das_fft_single(ctx.fft_dir, ctx.event_ids[idx])
            if f is None:
                return
            t = torch.as_tensor(f)
            freqs[idx] = t.to(device) if ctx.use_gpu else t
            if use_lr:
                _lr = load_das_fft_lr_single(ctx.fft_dir, ctx.event_ids[idx])
                if _lr is not None:
                    lr[idx] = {
                        k: (torch.as_tensor(v).to(device) if ctx.use_gpu
                            else torch.as_tensor(v))
                        for k, v in _lr.items()
                    }

        def _drop(idx):
            freqs.pop(idx, None)
            lr.pop(idx, None)

        def _block_events(b):
            return list(range(b * block_size, min((b + 1) * block_size, n_ev)))

        for alpha in range(n_blocks):
            a_events = _block_events(alpha)
            for ev in a_events:
                _load(ev)

            for beta in range(alpha, n_blocks):
                if beta != alpha:
                    b_events = _block_events(beta)
                    for ev in b_events:
                        _load(ev)

                for (i, j) in pairs_by_block.get((alpha, beta), []):
                    if i in freqs and j in freqs:
                        compute_one_pair_into(
                            i, j, freqs, lr,
                            dt=ctx.dt,
                            maxlag=ctx.mccc_max_lag_sec,
                            mccc_maxwin=ctx.mccc_maxwin,
                            mccc_damp=ctx.mccc_damp,
                            max_shift=ctx.mccc_max_shift,
                            smooth_window=ctx.polarity_smooth_window,
                            use_lr=use_lr, Picker=Picker,
                            Ckij=Ckij, Skij=Skij,
                        )
                    pbar.update(1)

                if beta != alpha:
                    for ev in b_events:
                        _drop(ev)

            for ev in a_events:
                _drop(ev)

        pbar.close()
        # Free GPU caching allocator once at the end (NOT in inner loop —
        # empty_cache is a synchronising op and kills throughput).
        if ctx.use_gpu:
            torch.cuda.empty_cache()

    # ── Hand off to shared driver, then postprocess ───────────────────────
    Ckij, Skij, svd_result = run_with_iteration(ctx, _run_mccc)
    postprocess(ctx, Ckij, Skij, svd_result)
