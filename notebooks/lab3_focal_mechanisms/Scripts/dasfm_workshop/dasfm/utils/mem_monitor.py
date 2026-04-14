"""mem_monitor — Background memory + VRAM sampler.

A small context manager that spawns a daemon thread sampling host RAM
(psutil) and GPU VRAM (pynvml) at fixed intervals.  Records the full
timeline so you can extract peaks, plot trajectories, or save a CSV.

Usage
-----

Wrap a step / function call::

    from dasfm.utils.mem_monitor import MemoryMonitor

    with MemoryMonitor(interval=0.5, gpu_indices=[1, 2], label="step2b") as mon:
        step2b_polarity(...)

    print(mon.summary())          # one-line peak report
    mon.save_csv("trace.csv")     # full timeline
    print(mon.peak_rss_gb())      # 12.34
    print(mon.peak_gpu_gb(1))     # 3.17

Notes
-----
* CPU RSS includes the parent process **and all descendants**, so the
  multi-CPU fork pool is monitored correctly (each forked worker has its
  own RSS that gets summed).
* GPU monitoring uses ``pynvml`` (NVIDIA Management Library) which sees
  total VRAM usage on the card across all processes — exactly what
  ``nvidia-smi`` reports.  If ``pynvml`` is not installed, GPU sampling
  is silently skipped.
* Sampling runs in a daemon thread, so unhandled exceptions in the
  monitored block still propagate cleanly.  The monitor stops in
  ``__exit__``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class _Sample:
    """One memory snapshot."""
    t:            float           # seconds since monitor start
    rss_total_gb: float           # parent + descendants RSS (GB)
    rss_main_gb:  float           # main process RSS only (GB)
    gpu_used_gb:  dict            # {gpu_idx: used_GB}


class MemoryMonitor:
    """Background sampler for host RAM + GPU VRAM.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 0.5).
    gpu_indices : list[int] | None
        GPUs to monitor.  ``None`` = all GPUs visible to nvml.
        Empty list = no GPU sampling.
    pid : int | None
        Root process PID.  ``None`` = current process.  Children are
        auto-discovered every sample (so newly forked workers are picked
        up automatically).
    label : str
        Optional label printed in summary.
    """

    def __init__(
        self,
        interval: float = 0.5,
        gpu_indices: list | None = None,
        pid: int | None = None,
        label: str = "",
    ) -> None:
        self.interval = interval
        self.gpu_indices = gpu_indices
        self.label = label
        self.pid = pid or psutil.Process().pid

        self._samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float = 0.0
        self._nvml_handles: dict = {}
        self._nvml_inited = False

    # ── Context manager ───────────────────────────────────────────────
    def __enter__(self):
        self._init_nvml()
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.interval)
        self._shutdown_nvml()
        return False  # never suppress an exception

    # ── nvml lifecycle ────────────────────────────────────────────────
    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            n_total = pynvml.nvmlDeviceGetCount()
            indices = (self.gpu_indices if self.gpu_indices is not None
                       else list(range(n_total)))
            self._nvml_handles = {
                i: pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in indices if 0 <= i < n_total
            }
            self._nvml_inited = True
        except Exception:
            self._nvml_handles = {}
            self._nvml_inited = False

    def _shutdown_nvml(self):
        if self._nvml_inited:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ── Sampling loop ─────────────────────────────────────────────────
    def _loop(self):
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop.is_set():
            try:
                # Tree RSS (parent + every descendant; new forks auto-included)
                main_rss = root.memory_info().rss
                tree_rss = main_rss
                for child in root.children(recursive=True):
                    try:
                        tree_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # GPU usage (all processes on the card, like nvidia-smi)
                gpu_used: dict = {}
                if self._nvml_inited:
                    import pynvml
                    for idx, handle in self._nvml_handles.items():
                        try:
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_used[idx] = info.used / 1e9
                        except Exception:
                            pass

                self._samples.append(_Sample(
                    t=time.monotonic() - self._t0,
                    rss_total_gb=tree_rss / 1e9,
                    rss_main_gb=main_rss / 1e9,
                    gpu_used_gb=gpu_used,
                ))
            except Exception:
                pass
            self._stop.wait(self.interval)

    # ── Reporting ─────────────────────────────────────────────────────
    def summary(self) -> str:
        """One-line peak summary, e.g.:

        ``[step2b] elapsed=42.3s  RSS peak: tree=2.45 GB, main=2.31 GB
        GPU1 peak 3.17 GB  GPU2 peak 3.18 GB  (85 samples)``
        """
        if not self._samples:
            return f"[{self.label or 'mem'}] no samples"
        peak_tree = max(s.rss_total_gb for s in self._samples)
        peak_main = max(s.rss_main_gb for s in self._samples)
        end_t = self._samples[-1].t

        gpu_lines = []
        first_gpus = self._samples[0].gpu_used_gb
        if first_gpus:
            for idx in sorted(first_gpus.keys()):
                peak = max(s.gpu_used_gb.get(idx, 0.0) for s in self._samples)
                gpu_lines.append(f"GPU{idx} peak {peak:.2f} GB")
        gpu_str = "  ".join(gpu_lines) if gpu_lines else "(no GPU)"

        return (
            f"[{self.label or 'mem'}] elapsed={end_t:.1f}s  "
            f"RSS peak: tree={peak_tree:.2f} GB, main={peak_main:.2f} GB  "
            f"{gpu_str}  ({len(self._samples)} samples)"
        )

    def peak_rss_gb(self) -> float:
        """Peak total RSS (parent + all descendants) over the monitored window."""
        return max((s.rss_total_gb for s in self._samples), default=0.0)

    def peak_rss_main_gb(self) -> float:
        """Peak RSS of the root process only (excludes fork pool children)."""
        return max((s.rss_main_gb for s in self._samples), default=0.0)

    def peak_gpu_gb(self, idx: int) -> float:
        """Peak VRAM used on a specific GPU index (all processes)."""
        return max((s.gpu_used_gb.get(idx, 0.0) for s in self._samples), default=0.0)

    def all_peak_gpus_gb(self) -> dict:
        """Map ``{gpu_idx: peak_GB}`` across every monitored GPU."""
        if not self._samples:
            return {}
        keys = set()
        for s in self._samples:
            keys.update(s.gpu_used_gb.keys())
        return {idx: self.peak_gpu_gb(idx) for idx in sorted(keys)}

    # ── Persistence ───────────────────────────────────────────────────
    def save_csv(self, path) -> None:
        """Write the full timeline to ``path`` as CSV."""
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._samples:
            return
        gpu_keys = sorted(self._samples[0].gpu_used_gb.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["t_sec", "rss_total_gb", "rss_main_gb"]
                + [f"gpu{i}_gb" for i in gpu_keys]
            )
            for s in self._samples:
                row = [
                    f"{s.t:.3f}",
                    f"{s.rss_total_gb:.4f}",
                    f"{s.rss_main_gb:.4f}",
                ]
                for k in gpu_keys:
                    row.append(f"{s.gpu_used_gb.get(k, 0.0):.4f}")
                w.writerow(row)

    def save_plot(self, path) -> None:
        """Save a 2-panel matplotlib plot (CPU RSS + GPU VRAM over time)."""
        if not self._samples:
            return
        import matplotlib
        import matplotlib.pyplot as plt

        ts        = [s.t for s in self._samples]
        rss_total = [s.rss_total_gb for s in self._samples]
        rss_main  = [s.rss_main_gb  for s in self._samples]
        gpu_keys  = sorted(self._samples[0].gpu_used_gb.keys())

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=120, sharex=True)
        axes[0].plot(ts, rss_total, label="tree RSS", color="C0")
        axes[0].plot(ts, rss_main,  label="main RSS", color="C1", linestyle="--")
        axes[0].set_ylabel("RAM [GB]")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f"{self.label or 'mem trace'}")

        if gpu_keys:
            for idx in gpu_keys:
                ys = [s.gpu_used_gb.get(idx, 0.0) for s in self._samples]
                axes[1].plot(ts, ys, label=f"GPU{idx}")
            axes[1].set_ylabel("VRAM [GB]")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "(no GPU monitored)",
                         ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xlabel("time [s]")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
