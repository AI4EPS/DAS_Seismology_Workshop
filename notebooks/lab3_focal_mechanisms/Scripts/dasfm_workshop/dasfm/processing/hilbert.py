"""2-D Hilbert-transform wave separation for DAS data.

Reference implementation: hilbert_real_all.py

Algorithm
---------
1. Apply 2-D Tukey taper (space × time) to the full filtered waveform.
2. Normalise each channel by its L1 norm.
3. 2-D Hilbert separation **once** on the full trace:
      Ht  = imag( hilbert( data_norm, axis=time  ) )
      Hxt = hilbert( Ht, axis=space )
      SR_R = data_norm + imag( Hxt )   # right-going
      SR_L = data_norm - imag( Hxt )   # left-going
4. Restore per-channel L1 amplitude.
   The taper is intentionally **not** undone — dividing by near-zero taper
   weights at the edges amplifies noise and has no physical justification.
   Instead, channels within the tapered fringe are flagged (``taper_mask_x``).
5. Cut P windows and S windows from the single pair of separated traces.

Axis convention throughout: ``(n_channels, n_samples)``.

Public function
---------------
* :func:`hilbert_separate`
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey


# ===========================================================================
# Public API
# ===========================================================================

def hilbert_separate(
    waveforms: np.ndarray,
    p_arr: np.ndarray,
    s_arr: np.ndarray,
    dt: float,
    cut_half: int = 100,
    tukey_alpha_t: float = 0.10,
    tukey_alpha_x: float = 0.03,
) -> dict:
    """Separate left- and right-going waves via 2-D Hilbert transform.

    The 2-D Hilbert transform is computed **once** on the full filtered
    trace.  P and S windows are then cut from the single pair of separated
    arrays, avoiding redundant computation.

    Parameters
    ----------
    waveforms : np.ndarray, shape (n_ch, n_t)
        Bandpass-filtered DAS waveforms.
    p_arr : np.ndarray, shape (n_ch,), int
        P-wave arrival sample indices per channel.  Set to ``-1`` for
        channels without a pick; their output window will be all zeros.
    s_arr : np.ndarray, shape (n_ch,), int
        S-wave arrival sample indices per channel.  Set to ``-1`` for
        channels without a pick.
    dt : float
        Sampling interval [s].
    cut_half : int
        Half-width of the output cut around the smoothed arrival [samples].
        Output window length = ``2 * cut_half``.  Default 100.
    tukey_alpha_t : float
        Tukey taper fraction along the time axis.  Default 0.10.
    tukey_alpha_x : float
        Tukey taper fraction along the space (channel) axis.  Default 0.03.
        Each edge tapers over ``alpha_x / 2 × n_ch`` channels.  Reduce this
        value to shrink the flagged fringe, at the cost of more spectral
        leakage in the spatial Hilbert transform.

    Returns
    -------
    dict with keys:
        ``p_original``, ``p_right``, ``p_left``  — (n_ch, n_win)  P windows
        ``s_original``, ``s_right``, ``s_left``  — (n_ch, n_win)  S windows
        ``t_cut``        — time axis relative to arrival [s], shape (n_win,)
        ``cut_half``     — the *cut_half* value used
        ``taper_mask_x`` — (n_ch,) bool, True = channel in spatial taper fringe
        ``taper_n_x``    — int, number of channels flagged on **each** edge
    """
    n_ch, n_t = waveforms.shape
    n_win = 2 * cut_half
    t_cut = (np.arange(n_win) - cut_half) * dt

    # ── Step 1: 2-D Tukey taper ───────────────────────────────────────────
    wx   = tukey(n_ch, alpha=tukey_alpha_x)     # (n_ch,)
    wt   = tukey(n_t,  alpha=tukey_alpha_t)     # (n_t,)
    wall = np.outer(wx, wt)                     # (n_ch, n_t)
    data = waveforms * wall                     # (n_ch, n_t)

    # ── Step 2: per-channel L1 normalisation ─────────────────────────────
    amp = np.sum(np.abs(data), axis=1)  # (n_ch,)
    amp = np.where(amp == 0.0, 1.0, amp)
    data_norm = np.nan_to_num(data / amp[:, np.newaxis])

    # ── Step 3: 2-D Hilbert separation (single pass) ──────────────────────
    #   axis=1 → along time;  axis=0 → along space (channels)
    ht  = np.imag(hilbert(data_norm, axis=1))   # (n_ch, n_t)
    hxt = hilbert(ht, axis=0)                   # (n_ch, n_t) complex

    sr_r_norm = data_norm + np.imag(hxt)
    sr_l_norm = data_norm - np.imag(hxt)

    # ── Step 4: restore per-channel L1 amplitude ──────────────────────────
    # The Tukey taper is intentionally kept in the data (NOT undone).
    # Channels in the tapered fringe carry attenuated signal; they are
    # flagged in taper_mask_x so callers can down-weight or exclude them.
    data_out = data_norm * amp[:, np.newaxis]
    sr_r     = sr_r_norm * amp[:, np.newaxis]
    sr_l     = sr_l_norm * amp[:, np.newaxis]

    # Taper fringe: channels where the spatial weight is not in the flat top.
    # Number of affected channels per edge = round(tukey_alpha_x / 2 * n_ch).
    taper_mask_x = wx < (1.0 - 1e-6)                       # (n_ch,) bool
    taper_n_x    = int(np.round(tukey_alpha_x / 2 * n_ch)) # channels per edge

    # ── Step 5: cut P and S windows from the single separated traces ──────
    # p_original / s_original are cut from the UNTAPERED waveforms so that
    # amplitude measurements are not biased by the spatial Tukey taper.
    # p_right / p_left / s_right / s_left use the tapered data_out, which is
    # required for a clean 2-D Hilbert transform.
    p_orig, p_right, p_left = cut_windows(
        data_out, sr_r, sr_l, p_arr, n_t, cut_half, data_orig=waveforms)
    s_orig, s_right, s_left = cut_windows(
        data_out, sr_r, sr_l, s_arr, n_t, cut_half, data_orig=waveforms)

    return {
        "p_original":      p_orig,
        "p_right":         p_right,
        "p_left":          p_left,
        "s_original":      s_orig,
        "s_right":         s_right,
        "s_left":          s_left,
        "cut_half":        cut_half,
        "taper_weights_x": wx.astype(np.float32),   # (n_ch,) continuous 0→1
        "taper_mask_x":    taper_mask_x,             # (n_ch,) bool
        "taper_n_x":       taper_n_x,
    }


# ===========================================================================
# Internal helper
# ===========================================================================

def cut_windows(
    data_out: np.ndarray,
    sr_r: np.ndarray,
    sr_l: np.ndarray,
    arr_idx: np.ndarray,
    n_t: int,
    cut_half: int,
    data_orig: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut windows centred on arrival indices.

    Parameters
    ----------
    data_out : (n_ch, n_t)
        Tapered waveforms (used for r_cut and l_cut, and for orig_cut when
        *data_orig* is ``None``).
    sr_r, sr_l : (n_ch, n_t)
        Right-going and left-going separated waveforms.
    arr_idx : (n_ch,) int
        Arrival sample indices in the full trace.
    n_t : int
        Total number of time samples.
    cut_half : int
        Half-window length in samples.
    data_orig : (n_ch, n_t) or None
        If supplied, ``orig_cut`` is taken from this array (typically the
        **untapered** raw waveforms) rather than from *data_out*.

    Returns
    -------
    (orig_cut, r_cut, l_cut), each (n_ch, 2*cut_half)
    """
    n_ch = data_out.shape[0]
    n_win = 2 * cut_half
    src = data_orig if data_orig is not None else data_out

    orig_cut = np.zeros((n_ch, n_win), dtype=np.float32)
    r_cut    = np.zeros((n_ch, n_win), dtype=np.float32)
    l_cut    = np.zeros((n_ch, n_win), dtype=np.float32)

    for i in range(n_ch):
        t1 = arr_idx[i] - cut_half
        t2 = arr_idx[i] + cut_half
        if 0 <= t1 and t2 <= n_t:
            orig_cut[i] = src[i, t1:t2]
            r_cut[i]    = sr_r[i, t1:t2]
            l_cut[i]    = sr_l[i, t1:t2]

    return orig_cut, r_cut, l_cut
