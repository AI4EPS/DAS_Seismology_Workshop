"""step3_invert_cpus — Multi-CPU parallel inversion via fork Pool.

Architecture
------------
Main process:
    1. ``step3_invert.run()`` loads data via ``_load_data()`` (no torch).
    2. ``run(data, num_workers, ...)`` sets ``_shared_data`` module global.
    3. Forks the Pool — children inherit ``_shared_data`` via copy-on-write
       (zero-copy: the OS shares the parent's pages until a write happens).
    4. Each worker initializer builds its per-worker torch state on ``cpu``.
    5. Main loop dispatches events via ``imap_unordered`` and updates the
       progress bar in the main process (no inter-process pbar locking).

Critical fork-safety constraints
--------------------------------
* This file's top-level imports MUST NOT trigger torch import.  Verified by
  the A.1 fork-safety smoke test (see ``step3_invert.py`` docstring).
* ``_load_data()`` (called from the parent) is torch-free — audited at Phase 0.5
  and enforced by the import discipline of ``step3_invert_serial.py``.
* The torch import happens inside ``_init_worker()`` (post-fork), so each
  worker imports torch into a fresh interpreter — no shared CUDA contexts,
  no doubled OMP thread pools.

Linux-only
----------
Uses ``multiprocessing.get_context("fork")`` explicitly.  Windows raises
(no fork start method).  macOS technically supports fork but Apple has
deprecated it for ObjC/libdispatch reasons; ``dasfm`` supports macOS only
as a development environment, production multi-CPU runs are Linux.
"""
from __future__ import annotations

import multiprocessing

from tqdm import tqdm


# Module-level state.
#   ``_shared_data``: SET BY PARENT in ``run()`` BEFORE forking.  Inherited by
#       fork children via copy-on-write.  Standard Python multiprocessing.Pool
#       idiom for sharing large read-only data — see Python docs example for
#       Pool initializer.  DO NOT pass _shared_data via initargs (that pickles
#       and copies it, defeating the COW purpose).
#   ``_worker_ctx``: SET BY EACH WORKER in ``_init_worker()`` after fork.
#       Holds the per-worker torch state (cpu tensors built once per worker
#       and reused across all events the worker processes).
_shared_data = None
_worker_ctx = None


def _init_worker():
    """Fork child initializer: limit threads, build per-worker torch ctx.

    ``_shared_data`` is automatically inherited from the parent via fork COW.
    Reading it (no writes) keeps the OS pages shared — true zero-copy.

    Each worker is single-threaded for BLAS/OMP — multi-threaded BLAS inside
    Pool workers fights the Python-level parallelism for cores.
    """
    global _worker_ctx
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from dasfm.steps.step3_invert_serial import _build_torch_state
    _worker_ctx = _build_torch_state(_shared_data, device="cpu")

    # Force single-threaded torch — parent may have initialized with more
    # threads before fork; env vars alone don't override an already-init'd pool.
    import torch
    torch.set_num_threads(1)
    _worker_ctx["_quiet"] = True
    _worker_ctx["logger"] = None    # workers must not write to the parent's log


def _process_one_event(iev: int) -> int:
    """Worker function: process one event using the per-worker ctx."""
    from dasfm.inversion.pipeline import process_event
    process_event(iev, _worker_ctx)
    return iev


def run(data: dict, num_workers: int, event_indices=None) -> None:
    """Multi-CPU run.  Called by ``step3_invert.run()`` with pre-loaded data.

    Parameters
    ----------
    data : dict
        Output of ``_load_data()`` — pure numpy.  Shared with workers via
        fork COW (zero-copy).
    num_workers : int
        Fork pool size.
    event_indices : list[int] or None
        Subset of events to process.  ``None`` = all events.  Subsets are
        dynamically dispatched via ``imap_unordered`` — no pre-partitioning.
    """
    global _shared_data
    _shared_data = data    # parent sets BEFORE fork; children inherit via COW

    # Explicit fork context — required for COW + module global pattern.
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "step3_invert_cpus requires multiprocessing 'fork' start method "
            "(Linux only).  Use step3_invert_serial or step3_invert_gpus on "
            "Windows.")
    ctx_mp = multiprocessing.get_context("fork")

    if event_indices is None:
        event_indices = list(range(data["n_ev"]))
    if not event_indices:
        return

    logger = data.get("logger")
    if logger is not None:
        logger.info("=" * 60)
        logger.info(f"  [cpus] num_workers={num_workers}  events={len(event_indices)}")
        logger.info("=" * 60)

    pbar = tqdm(total=len(event_indices), desc="events", unit="ev",
                leave=True)
    pool = ctx_mp.Pool(num_workers, initializer=_init_worker)
    try:
        # imap_unordered iterates the event list directly — workers grab the
        # next event as they finish their current one (dynamic load balancing).
        # No partitioning needed; subsets work for free.
        for _ in pool.imap_unordered(_process_one_event, event_indices):
            pbar.update(1)
    finally:
        pool.close()
        pool.join()
        pbar.close()

    if logger is not None:
        import time as _time
        logger.info("=" * 60)
        logger.info(f"  Done  ({_time.time() - data['t0']:.1f} s)")
        logger.info(f"  -> {data['RESULT_ROOT']}")
        logger.info("=" * 60)
        logger.close()
