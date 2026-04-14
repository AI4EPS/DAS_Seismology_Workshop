"""GPU-accelerated 2-D Hilbert-transform wave separation for DAS data.

Uses PyTorch FFT ops (torch.fft) to compute the Hilbert transforms on GPU,
then cuts P/S windows via vectorised advanced indexing — no Python loops over
channels.  The Tukey-taper construction and per-channel L1 normalisation are
likewise done on GPU.

The bandpass filter (``preprocess_raw``) is intentionally kept on CPU
because no stable, dependency-free ``sosfiltfilt`` equivalent exists in
PyTorch.  Data are transferred to GPU after filtering.

Public function
---------------
* :func:`hilbert_separate_gpu`
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey


# ===========================================================================
# Public API
# ===========================================================================

def hilbert_separate_gpu(
    waveforms: np.ndarray,
    p_arr: np.ndarray,
    s_arr: np.ndarray,
    dt: float,
    cut_half: int = 100,
    tukey_alpha_t: float = 0.10,
    tukey_alpha_x: float = 0.03,
    device: str = "cuda:0",
) -> dict:
    """GPU-accelerated 2-D Hilbert separation via torch.fft.

    Drop-in replacement for :func:`dasfm.processing.hilbert.hilbert_separate`.
    The interface and return dict are identical; all heavy computation runs on
    the requested CUDA device.

    Parameters
    ----------
    waveforms : np.ndarray, shape (n_ch, n_t)
        Bandpass-filtered DAS waveforms (CPU numpy array; transferred to GPU
        inside this function).
    p_arr : np.ndarray, shape (n_ch,), int
        P-wave arrival sample indices.  ``-1`` = no pick.
    s_arr : np.ndarray, shape (n_ch,), int
        S-wave arrival sample indices.  ``-1`` = no pick.
    dt : float
        Sampling interval [s].
    cut_half : int
        Half-window length in samples.  Output window = ``2*cut_half+1``.
    tukey_alpha_t : float
        Tukey taper fraction along time.
    tukey_alpha_x : float
        Tukey taper fraction along space (channels).
    device : str
        PyTorch device string, e.g. ``"cuda:0"``.

    Returns
    -------
    dict  (same keys as :func:`hilbert_separate`)
        ``p_original``, ``p_right``, ``p_left``  — (n_ch, n_win) float32
        ``s_original``, ``s_right``, ``s_left``  — (n_ch, n_win) float32
        ``t_cut``        — (n_win,) float64 time axis relative to arrival [s]
        ``cut_half``     — int
        ``taper_mask_x`` — (n_ch,) bool
        ``taper_weights_x`` — (n_ch,) float32
        ``taper_n_x``    — int
    """
    import torch
    dev = torch.device(device)

    n_ch, n_t = waveforms.shape
    n_win = 2 * cut_half
    t_cut = (np.arange(n_win) - cut_half) * dt

    # ── Step 1: 2-D Tukey taper (build on CPU, move to GPU) ──────────────────
    wx_np = tukey(n_ch, alpha=tukey_alpha_x).astype(np.float32)   # (n_ch,)
    wt_np = tukey(n_t,  alpha=tukey_alpha_t).astype(np.float32)   # (n_t,)
    wx   = torch.from_numpy(wx_np).to(dev)
    wt   = torch.from_numpy(wt_np).to(dev)
    wall = torch.outer(wx, wt)                                      # (n_ch, n_t)

    # Transfer filtered waveforms to GPU
    wav_gpu = torch.from_numpy(waveforms.astype(np.float32)).to(dev)  # (n_ch, n_t)
    data    = wav_gpu * wall

    # ── Step 2: per-channel L1 normalisation ─────────────────────────────────
    amp = torch.sum(torch.abs(data), dim=1)   # (n_ch,)
    amp = torch.where(amp == 0.0, torch.ones_like(amp), amp)
    data_norm = data / amp.unsqueeze(1)               # (n_ch, n_t)

    # ── Step 3: 2-D Hilbert separation via torch.fft ─────────────────────────
    #   axis=1 (time) first, then axis=0 (space)
    ht  = torch.imag(torch_analytic(data_norm, dim=1))   # (n_ch, n_t) real
    hxt = torch_analytic(ht, dim=0)                       # (n_ch, n_t) complex

    sr_r_norm = data_norm + torch.imag(hxt)
    sr_l_norm = data_norm - torch.imag(hxt)

    # ── Step 4: restore per-channel L1 amplitude ─────────────────────────────
    data_out = data_norm * amp.unsqueeze(1)
    sr_r     = sr_r_norm * amp.unsqueeze(1)
    sr_l     = sr_l_norm * amp.unsqueeze(1)

    taper_mask_x = wx_np < (1.0 - 1e-6)
    taper_n_x    = int(np.round(tukey_alpha_x / 2 * n_ch))

    # ── Step 5: vectorised window cutting on GPU ──────────────────────────────
    p_arr_t = torch.from_numpy(p_arr.astype(np.int64)).to(dev)
    s_arr_t = torch.from_numpy(s_arr.astype(np.int64)).to(dev)

    p_orig, p_right, p_left = cut_windows_torch(
        data_out, sr_r, sr_l, p_arr_t, n_t, cut_half, data_orig=wav_gpu)
    s_orig, s_right, s_left = cut_windows_torch(
        data_out, sr_r, sr_l, s_arr_t, n_t, cut_half, data_orig=wav_gpu)

    return {
        "p_original":      p_orig.cpu().numpy(),
        "p_right":         p_right.cpu().numpy(),
        "p_left":          p_left.cpu().numpy(),
        "s_original":      s_orig.cpu().numpy(),
        "s_right":         s_right.cpu().numpy(),
        "s_left":          s_left.cpu().numpy(),
        "cut_half":        cut_half,
        "taper_weights_x": wx_np,
        "taper_mask_x":    taper_mask_x,
        "taper_n_x":       taper_n_x,
    }


# ===========================================================================
# Internal helpers
# ===========================================================================

def torch_analytic(x: "torch.Tensor", dim: int) -> "torch.Tensor":
    """Analytic signal along ``dim`` (mirrors ``scipy.signal.hilbert``).

    Uses ``torch.fft.fft`` → apply one-sided frequency filter → ``ifft``.
    Works for both real and complex input tensors.
    """
    import torch
    n  = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)               # complex

    # One-sided filter weights (same as scipy's implementation)
    h = torch.zeros(n, dtype=torch.float32, device=x.device)
    if n % 2 == 0:
        h[0] = 1.0
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0

    shape = [1] * x.ndim
    shape[dim % x.ndim] = n
    return torch.fft.ifft(Xf * h.view(shape), dim=dim)   # complex output


def cut_windows_torch(
    data_out: "torch.Tensor",
    sr_r: "torch.Tensor",
    sr_l: "torch.Tensor",
    arr_idx: "torch.Tensor",
    n_t: int,
    cut_half: int,
    data_orig: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Vectorised window cutting on GPU (no Python loop over channels).

    Equivalent to :func:`dasfm.processing.hilbert.cut_windows` but uses
    advanced (gather-style) indexing so all channels are processed in a
    single kernel launch.
    """
    import torch
    n_ch  = data_out.shape[0]
    n_win = 2 * cut_half
    src   = data_orig if data_orig is not None else data_out

    # Build 2-D index grids: row = channel, col = sample position
    row_idx = torch.arange(n_ch, device=data_out.device).unsqueeze(1)           # (n_ch, 1)
    col_off = torch.arange(n_win, device=data_out.device).unsqueeze(0) - cut_half  # (1, n_win)
    col_idx = arr_idx.unsqueeze(1) + col_off                                     # (n_ch, n_win)

    # Validity mask: arr_idx >= 0 and all column indices within [0, n_t)
    arr_valid = (arr_idx >= 0).unsqueeze(1)                    # (n_ch, 1)
    col_valid = (col_idx >= 0) & (col_idx < n_t)               # (n_ch, n_win)
    mask      = arr_valid & col_valid                           # (n_ch, n_win)

    col_safe = col_idx.clamp(0, n_t - 1)                       # safe for indexing
    zero     = torch.zeros(1, device=data_out.device)

    orig_cut = torch.where(mask, src[row_idx,     col_safe], zero).float()
    r_cut    = torch.where(mask, sr_r[row_idx,    col_safe], zero).float()
    l_cut    = torch.where(mask, sr_l[row_idx,    col_safe], zero).float()

    return orig_cut, r_cut, l_cut
