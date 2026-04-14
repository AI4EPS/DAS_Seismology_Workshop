"""mccc_kernel — Pure-function MCCC inner kernels for serial / parallel backends.

Module-level functions (not closures) so multiprocessing.Pool can pickle them.

Public API
----------
* :class:`BlockPairTask`
* :func:`compute_block_pair_kernel`  — used by cpus + gpus backends
* :func:`compute_one_pair_into`      — used by serial backend (in-place write)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d


# ═══════════════════════════════════════════════════════════════════════════
#  Task definition
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BlockPairTask:
    """All inputs needed to compute one (block_k, block_l) MCCC chunk.

    Used by :mod:`dasfm.steps.step2b_polarity_cpus` and
    :mod:`dasfm.steps.step2b_polarity_gpus` to dispatch block-pair work to
    fork workers / GPU threads.  Replaces the legacy 16-element positional
    tuple with a self-documenting dataclass.
    """
    block_k:        int
    block_l:        int
    ev_indices_k:   list
    ev_indices_l:   list
    event_ids:      list
    fft_dir:        str
    dt:             float
    maxlag:         float
    mccc_maxwin:    int
    mccc_damp:      float
    max_shift:      int
    polarity_method: str
    has_lr:         bool
    n_ch:           int
    device:         str
    smooth_window:  int
    allowed_pairs:  Optional[set] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Public kernels
# ═══════════════════════════════════════════════════════════════════════════

def compute_block_pair_kernel(task: BlockPairTask) -> list:
    """Compute MCCC for one block pair.

    Pure function — no module-level state, fork-safe.  Returns a list of
    ``(gi, gj, si, cc)`` tuples where ``cc`` is a ``(n_ch,)`` float32 array.
    The caller (main process) is responsible for streaming these into the
    persistent dense ``Ckij/Skij`` matrices.
    """
    import torch  # local import — keep module top-level lightweight for fork

    use_lr  = task.polarity_method == "hilbert" and task.has_lr
    use_gpu = isinstance(task.device, str) and task.device.startswith("cuda")
    if use_gpu:
        from dasfm.picking.mccc_gpu import MCCCPickerGPU as Picker
    else:
        from dasfm.picking.mccc import MCCCPicker as Picker

    # Load FFT cache for block_k events
    freqs_k, lr_k = _load_block_ffts(
        task.ev_indices_k, task.event_ids, task.fft_dir,
        use_lr, task.device, use_gpu,
    )

    # Block_l: reuse freqs_k if same block, else load
    if task.block_k == task.block_l:
        freqs_l, lr_l = freqs_k, lr_k
    else:
        freqs_l, lr_l = _load_block_ffts(
            task.ev_indices_l, task.event_ids, task.fft_dir,
            use_lr, task.device, use_gpu,
        )

    pairs = _build_pair_list(
        task.block_k, task.block_l,
        task.ev_indices_k, task.ev_indices_l,
        task.allowed_pairs,
    )

    results = []
    for (gi, gj) in pairs:
        if gi not in freqs_k or gj not in freqs_l:
            continue
        for si in range(task.max_shift):
            cc = _mccc_one_pair_inner(
                freqs_k[gi], freqs_l[gj],
                lr_k.get(gi) if use_lr else None,
                lr_l.get(gj) if use_lr else None,
                task.dt, task.maxlag, task.mccc_maxwin, task.mccc_damp,
                use_lr, Picker, si, task.smooth_window,
            )
            results.append((gi, gj, si, cc))

    return results


def compute_one_pair_into(
    i, j, freqs, lr, *,
    dt, maxlag, mccc_maxwin, mccc_damp, max_shift, smooth_window,
    use_lr, Picker, Ckij, Skij,
):
    """Compute MCCC for a single (i, j) pair and stream-write into Ckij/Skij.

    Used by the serial backend's block-streaming loop.  Writes ``cc`` directly
    into the persistent dense ``Ckij`` (si=0) and ``Skij`` (si>=1) matrices —
    no intermediate Python list.
    """
    fi, fj = freqs[i], freqs[j]
    lr_i = lr.get(i) if use_lr else None
    lr_j = lr.get(j) if use_lr else None
    for si in range(max_shift):
        cc = _mccc_one_pair_inner(
            fi, fj, lr_i, lr_j,
            dt, maxlag, mccc_maxwin, mccc_damp,
            use_lr, Picker, si, smooth_window,
        )
        target = Ckij if si == 0 else Skij
        target[:, i, j] = cc
        target[:, j, i] = cc


# ═══════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mccc_one_pair_inner(
    fi, fj, lr_i, lr_j,
    dt, maxlag, mccc_maxwin, mccc_damp,
    use_lr, Picker, si, smooth_window,
):
    """Core MCCC computation for one (i, j, si) triplet.

    Returns a ``(n_ch,)`` float32 numpy array of channel-wise correlations.
    """
    from dasfm.picking.mccc import xcorr_from_freq

    xcor, _ = xcorr_from_freq(fi, fj, dt, maxlag=maxlag, channel_shift=si)
    if use_lr and lr_i is not None and lr_j is not None:
        x1, _ = xcorr_from_freq(lr_i["left"],  lr_j["left"],  dt,
                                maxlag=maxlag, channel_shift=si)
        x2, _ = xcorr_from_freq(lr_i["right"], lr_j["right"], dt,
                                maxlag=maxlag, channel_shift=si)
        xcor = xcor + x1 + x2

    mp = Picker(xcor, dt, mccc_maxwin=mccc_maxwin, damp=0, mccc_damp=mccc_damp)
    sol = mp.solve()
    cc = sol["cc_main"].cpu().numpy().astype(np.float32)
    if smooth_window > 1:
        cc = uniform_filter1d(cc, size=smooth_window)
    return cc


def _load_block_ffts(ev_indices, event_ids, fft_dir, use_lr, device, use_gpu):
    """Load freqs_p (and optionally L/R) for a list of event indices.

    Returns ``(freqs_dict, lr_dict)`` where keys are global event indices and
    values are torch tensors on the requested device.
    """
    import torch
    from dasfm.io.das_fft import load_das_fft_single, load_das_fft_lr_single

    freqs: dict = {}
    lr:    dict = {}
    for idx in ev_indices:
        eid = event_ids[idx]
        f = load_das_fft_single(fft_dir, eid)
        if f is None:
            continue
        t = torch.as_tensor(f)
        freqs[idx] = t.to(device) if use_gpu else t
        if use_lr:
            _lr = load_das_fft_lr_single(fft_dir, eid)
            if _lr is not None:
                lr[idx] = {
                    k: (torch.as_tensor(v).to(device) if use_gpu
                        else torch.as_tensor(v))
                    for k, v in _lr.items()
                }
    return freqs, lr


def _build_pair_list(block_k, block_l, ev_k, ev_l, allowed_pairs):
    """Generate the list of (gi, gj) pairs for one block pair.

    Diagonal block (block_k == block_l): only the upper triangle within ev_k.
    Off-diagonal block: full Cartesian product ev_k × ev_l.
    Optionally filtered by ``allowed_pairs`` (sparse mode).
    """
    if block_k == block_l:
        pairs = [(ev_k[i], ev_k[j])
                 for i in range(len(ev_k))
                 for j in range(i + 1, len(ev_k))]
    else:
        pairs = [(ik, jl) for ik in ev_k for jl in ev_l]
    if allowed_pairs is not None:
        pairs = [(i, j) for (i, j) in pairs
                 if (min(i, j), max(i, j)) in allowed_pairs]
    return pairs
