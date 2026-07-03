"""Memory-budget estimation helpers for dasfm steps.

The functions here predict peak GPU/CPU memory consumption *before* a
step starts allocating, so the dispatcher can choose between full-cache
and on-demand strategies and the user gets a single-line "headline"
estimate in the log.

The estimates are intentionally upper bounds — they err on the side of
safety so a passing check guarantees the step will fit, while a failing
check triggers the on-demand fallback.

Public functions
----------------
* :func:`estimate_step2b_vram`      — single-process peak (used by
  :func:`dasfm.steps.step2b_polarity_serial.run` and the multi-GPU
  :func:`dasfm.steps.step2b_polarity_gpus.run`, where each GPU thread
  is its own single-GPU view).
* :func:`estimate_step2b_vram_cpus` — multi-process peak for the fork
  Pool path in :func:`dasfm.steps.step2b_polarity_cpus.run`.  Divides
  the host-RAM budget by ``num_workers`` so each forked child gets its
  own slice.
"""
from __future__ import annotations


# MCCCPicker internal knobs.  Mirror the defaults in
# dasfm.picking.mccc.MCCCPicker.__init__ — step2b never overrides them, so
# the estimator can hard-code the same values here.  If you ever change the
# MCCCPicker defaults, update this file as well.
_MCCC_CHUNK_SIZE   = 50_000
_MCCC_SCALE_FACTOR = 10

# CUDA framework overhead: cuBLAS / cuDNN / cuFFT init + caching-allocator
# fragmentation.  Calibrated against measured nvidia-smi values for the
# step2b workload (see git history of this module for the data points).
_CUDA_ALLOC_HEADROOM = 1.3   # PyTorch caching allocator fragmentation
_CUDA_CONTEXT_BYTES  = 0.3e9  # cuBLAS/cuDNN/cuFFT init constant


def estimate_step2b_vram(
    n_ev: int,
    n_ch: int,
    nfast: int,
    use_lr: bool,
    use_gpu: bool,
    device: str | int = "cuda:0",
) -> dict:
    """Estimate peak memory for step2b MCCC.

    Parameters
    ----------
    n_ev : int
        Number of events to be processed.
    n_ch : int
        Number of DAS channels.
    nfast : int
        FFT length used in step2a's das_fft cache (already a power of 2).
    use_lr : bool
        True when hybrid polarity is enabled and ``p_left/p_right`` were
        actually saved by step2a — adds 2 extra FFT tensors per event.
    use_gpu : bool
        True for CUDA, False for CPU.  CUDA path adds framework overhead
        (cuBLAS/cuDNN/cuFFT init + caching-allocator fragmentation).
    device : str or int, default "cuda:0"
        Specific GPU to query for `total_memory`.  Accepts ``"cuda:N"``,
        ``"cuda"`` (uses current device), or an integer index.  Ignored
        when ``use_gpu`` is False.

    Returns
    -------
    dict with keys:
        ``per_event_bytes``     — bytes for one event's freqs_p (cache unit)
        ``cache_bytes``         — bytes for the persistent FFT cache (all
                                  events)
        ``mccc_working_bytes``  — bytes for MCCCPicker._form_coo working set
        ``form_coo_ratio``      — working set as a multiplier of cache_unit
        ``peak_bytes``          — total peak when caching all events at once
        ``available_bytes``     — half of total VRAM (or RAM on CPU); the
                                  pipeline only uses ``< available_bytes``
                                  to leave headroom for other allocations
        ``block_size``          — recommended events per block for the
                                  block-streaming path in
                                  ``step2b_polarity_serial``.  Equals
                                  ``n_ev`` when the full cache fits;
                                  otherwise the largest ``b`` such that
                                  *two* blocks of ``b`` events plus the
                                  working set still fit in budget.
        ``n_blocks``            — ``ceil(n_ev / block_size)``
    """
    mccc_chunk_size   = _MCCC_CHUNK_SIZE
    mccc_scale_factor = _MCCC_SCALE_FACTOR
    # 1. Persistent FFT cache: rfft is one-sided so freqs_p is
    #    (n_ch, nfast/2 + 1) complex64 (8 B).  Hybrid mode also keeps the
    #    left + right FFT, so the full cache is 3× per-event.
    n_freq = nfast // 2 + 1
    per_event_bytes = n_ch * n_freq * 8
    n_tensors_per_event = 3 if use_lr else 1
    cache_bytes = n_ev * per_event_bytes * n_tensors_per_event

    # 2. Transient working set inside MCCCPicker._form_coo: each call
    #    builds an `xcf + xct` chunk of (chunk_size pairs) ×
    #    (scale_factor × nfast samples).  Express as a multiplier of one
    #    event's freqs_p — derivation:
    #
    #        xcf+xct  ≈ 2 × chunk_size × scale_factor × nfast × 4  bytes
    #        per_event ≈        n_ch  ×                nfast × 4  bytes
    #        ratio    = 2 × chunk_size × scale_factor / n_ch
    #
    #    The ratio doesn't depend on `nfast`, only on `n_ch` and the two
    #    MCCCPicker constants.  This is intentionally a ~25% upper bound
    #    relative to the actual nextpow2 geometry — the headroom is the
    #    safety margin we want.
    form_coo_ratio = 2 * mccc_chunk_size * mccc_scale_factor / max(n_ch, 1)
    mccc_working_bytes = form_coo_ratio * per_event_bytes

    # 3. Add CUDA framework overhead (GPU only) and look up available memory.
    if use_gpu:
        import torch
        peak_bytes = (cache_bytes + mccc_working_bytes) * _CUDA_ALLOC_HEADROOM \
                     + _CUDA_CONTEXT_BYTES
        # Resolve the target device — accept "cuda:N", "cuda", or an int.
        # The CUDA-allocator headroom (×1.3) and context constant already
        # leave fragmentation slack, so use 90% of the card's total memory
        # as the budget instead of the legacy 50% half-budget.
        if isinstance(device, str):
            if device == "cuda":
                device_idx = torch.cuda.current_device()
            elif device.startswith("cuda:"):
                device_idx = int(device.split(":", 1)[1])
            else:
                device_idx = torch.cuda.current_device()
        else:
            device_idx = int(device)
        available_bytes = (
            torch.cuda.get_device_properties(device_idx).total_memory * 0.9
        )
    else:
        # CPU mode: be more conservative — host RAM is shared with the OS,
        # other Python processes, multiprocessing workers, page cache, etc.
        import psutil
        peak_bytes = cache_bytes + mccc_working_bytes
        available_bytes = psutil.virtual_memory().available * 0.5

    # 4. Block size for the streaming path in step2b_polarity_serial.
    #    The block-streaming loop holds at most 2 blocks of events in cache
    #    at once (the current α block plus the moving β block).  Solve
    #    for the largest `b` such that
    #
    #        (2 * b * per_event * n_tensors + working) * 1.3 + ctx ≤ avail   (GPU)
    #        (2 * b * per_event * n_tensors + working)             ≤ avail   (CPU)
    #
    #    When the full cache already fits, no streaming is needed → b = n_ev.
    if peak_bytes <= available_bytes:
        block_size = n_ev
    else:
        if use_gpu:
            cache_2blocks_budget = (
                (available_bytes - _CUDA_CONTEXT_BYTES) / _CUDA_ALLOC_HEADROOM
                - mccc_working_bytes
            )
        else:
            cache_2blocks_budget = available_bytes - mccc_working_bytes
        block_size = int(
            cache_2blocks_budget / (2 * per_event_bytes * n_tensors_per_event)
        )
        # Working set itself fits on every reasonable GPU (~2-16 GB), so
        # the budget is always positive — clamp to [1, n_ev] just in case.
        block_size = max(1, min(block_size, n_ev))

    n_blocks = (n_ev + block_size - 1) // block_size

    return {
        "per_event_bytes":    per_event_bytes,
        "cache_bytes":        cache_bytes,
        "mccc_working_bytes": mccc_working_bytes,
        "form_coo_ratio":     form_coo_ratio,
        "peak_bytes":         peak_bytes,
        "available_bytes":    available_bytes,
        "block_size":         block_size,
        "n_blocks":           n_blocks,
    }


def estimate_step2b_vram_cpus(
    n_ev: int,
    n_ch: int,
    nfast: int,
    use_lr: bool,
    num_workers: int,
) -> dict:
    """Estimate peak host RAM for step2b multi-CPU MCCC.

    Each forked worker independently loads FFT and runs MCCCPicker, so
    total RAM ≈ ``num_workers × per-worker peak``.  This function divides
    the half-RAM budget by ``num_workers`` and reports the recommended
    ``block_size`` that lets each worker fit within its slice.

    Parameters
    ----------
    n_ev : int
        Number of events to be processed.
    n_ch : int
        Number of DAS channels.
    nfast : int
        FFT length used in step2a's das_fft cache (already a power of 2).
    use_lr : bool
        True when hybrid polarity is enabled and ``p_left/p_right`` were
        actually saved by step2a — adds 2 extra FFT tensors per event.
    num_workers : int
        Number of fork ``multiprocessing.Pool`` workers running
        :func:`dasfm.picking.mccc_kernel.compute_block_pair_kernel` concurrently.

    Returns
    -------
    dict with the same keys as :func:`estimate_step2b_vram`, plus:
        ``per_worker_bytes``  — single worker's expected peak (cache +
                                working set, no fragmentation headroom)
        ``num_workers``       — echoed for clarity
    """
    # 1. Per-event FFT cache (rfft is one-sided).
    n_freq = nfast // 2 + 1
    per_event_bytes = n_ch * n_freq * 8
    n_tensors_per_event = 3 if use_lr else 1

    # 2. Working set inside MCCCPicker._form_coo (xcf + xct chunk).
    form_coo_ratio = 2 * _MCCC_CHUNK_SIZE * _MCCC_SCALE_FACTOR / max(n_ch, 1)
    mccc_working_bytes = form_coo_ratio * per_event_bytes

    # 3. Per-worker model: each fork child holds its own full event cache
    #    (worst case: cache_bytes for the assigned block-pair) plus the
    #    transient working set during _form_coo.  No CUDA framework
    #    overhead because workers run on CPU.
    per_worker_full_cache = n_ev * per_event_bytes * n_tensors_per_event
    per_worker_peak       = per_worker_full_cache + mccc_working_bytes

    # 4. Aggregate across all W workers.
    cache_bytes = num_workers * per_worker_full_cache
    peak_bytes  = num_workers * per_worker_peak

    import psutil
    available_bytes = psutil.virtual_memory().available * 0.5

    # 5. Solve for the largest `block_size` such that
    #
    #        num_workers × (2*b*per_event*n_tensors + working) ≤ available
    #
    #    When the full cache fits, no streaming is needed → block_size = n_ev.
    if peak_bytes <= available_bytes:
        block_size = n_ev
    else:
        per_worker_budget    = available_bytes / num_workers
        cache_2blocks_budget = per_worker_budget - mccc_working_bytes
        block_size = int(
            cache_2blocks_budget / (2 * per_event_bytes * n_tensors_per_event)
        )
        block_size = max(1, min(block_size, n_ev))

    n_blocks = (n_ev + block_size - 1) // block_size

    return {
        "per_event_bytes":    per_event_bytes,
        "per_worker_bytes":   per_worker_peak,
        "cache_bytes":        cache_bytes,
        "mccc_working_bytes": mccc_working_bytes,
        "form_coo_ratio":     form_coo_ratio,
        "peak_bytes":         peak_bytes,
        "available_bytes":    available_bytes,
        "block_size":         block_size,
        "n_blocks":           n_blocks,
        "num_workers":        num_workers,
    }
