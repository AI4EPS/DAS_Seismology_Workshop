"""step3_invert_gpus — Multi-GPU parallel inversion via ThreadPoolExecutor.

Architecture
------------
Main process:
    1. ``step3_invert.run()`` loads data via ``_load_data()`` (no torch yet).
    2. ``run(data, devices, ...)`` spawns a ``ThreadPoolExecutor`` with one
       thread per GPU.  Threads share the parent's memory directly — no fork,
       no pickling, no copy.
    3. Each thread builds its per-device torch state on its assigned GPU and
       processes its slice of events sequentially.
    4. Threads update a single shared ``tqdm`` progress bar via
       ``threading.Lock``.

Why threads (not processes) for multi-GPU?
------------------------------------------
* torch CUDA ops release the GIL → threads run in parallel on separate GPUs.
* numpy bulk ops also release the GIL.
* No fork-safety constraints — threads share the parent process.
* No pickling overhead for ``data``.
* The Python-level pre/post processing per event is small enough that the
  remaining GIL contention is negligible compared to GPU compute time.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from dasfm.steps._runner_common import partition_events_round_robin


def run(data: dict, devices: list[str], event_indices=None) -> None:
    """Multi-GPU run.  Called by ``step3_invert.run()`` with pre-loaded data.

    Parameters
    ----------
    data : dict
        Output of ``_load_data()`` — pure numpy.  Shared with worker threads
        via closure (no copy — threads see the same memory).
    devices : list[str]
        e.g. ``["cuda:0", "cuda:1"]`` — must contain ≥ 2 cuda devices.
    event_indices : list[int] or None
        Subset of events to process.  ``None`` = all events.  The subset is
        round-robin partitioned across GPUs.
    """
    if not isinstance(devices, list):
        raise ValueError(
            f"step3_invert_gpus: device must be a list of GPU devices for "
            f"multi-GPU mode, got {devices!r}")
    gpu_devices = [d for d in devices if d.startswith("cuda")]
    num_gpu = len(gpu_devices)
    if num_gpu < 2:
        raise ValueError(
            f"step3_invert_gpus: need ≥2 GPUs for multi-GPU mode, got "
            f"{gpu_devices!r}")

    if event_indices is None:
        event_indices = list(range(data["n_ev"]))
    if not event_indices:
        return

    event_lists = partition_events_round_robin(event_indices, num_gpu)

    logger = data.get("logger")
    if logger is not None:
        logger.info("=" * 60)
        logger.info(f"  [gpus] devices={gpu_devices}  events={len(event_indices)}")
        for gi, ev_list in enumerate(event_lists):
            logger.info(f"    {gpu_devices[gi]}: {len(ev_list)} events")
        logger.info("=" * 60)

    pbar = tqdm(total=len(event_indices), desc="events", unit="ev",
                leave=True)
    pbar_lock = threading.Lock()

    def _gpu_thread_main(events: list[int], device: str) -> None:
        """Per-thread: build torch state on this GPU, process assigned events."""
        # Lazy import — pulls torch.  Threads share the parent's torch state
        # safely (CUDA contexts are per-thread per-device).
        from dasfm.steps.step3_invert_serial import _build_torch_state
        from dasfm.inversion.pipeline import process_event
        ctx = _build_torch_state(data, device=device)
        ctx["_quiet"] = True
        ctx["logger"] = None    # threads must not double-log to parent
        for iev in events:
            process_event(iev, ctx)
            with pbar_lock:
                pbar.update(1)

    try:
        with ThreadPoolExecutor(max_workers=num_gpu) as ex:
            futures = [ex.submit(_gpu_thread_main, event_lists[gi], gpu_devices[gi])
                       for gi in range(num_gpu)]
            # Wait + propagate any thread exceptions to the main process.
            for f in futures:
                f.result()
    finally:
        pbar.close()

    if logger is not None:
        import time as _time
        logger.info("=" * 60)
        logger.info(f"  Done  ({_time.time() - data['t0']:.1f} s)")
        logger.info(f"  -> {data['RESULT_ROOT']}")
        logger.info("=" * 60)
        logger.close()
