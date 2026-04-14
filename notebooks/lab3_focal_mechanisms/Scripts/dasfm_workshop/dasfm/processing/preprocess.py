"""Waveform preprocessing for DASData and DASRawData objects.

Preprocessing is deliberately separated from I/O so that:
* readers stay pure (no side-effects on the waveforms).
* Users can apply different preprocessing pipelines to the same raw data.
* Each step is individually testable.

Two public functions:

* :func:`preprocess`     — operates on :class:`~dasfm.io.data.DASData`
  (pre-extracted P/S windows).  Bandpass on short windows will trigger a
  warning; prefer ``lowpass`` + ``demean`` + ``detrend`` for those.

* :func:`preprocess_raw` — operates on :class:`~dasfm.io.data.DASRawData`
  (continuous traces).  Bandpass is fully appropriate here.

Typical usage::

    from dasfm.io.das_io import load_das_window, load_das_raw
    from dasfm.processing.preprocess import preprocess, preprocess_raw

    # pre-extracted windows  →  use lowpass (short record)
    win = load_das_window("event.h5")
    win_lp = preprocess(win, filter_type="lowpass", freq_max=20.0,
                        demean=True, detrend=True)

    # continuous trace  →  bandpass is fine (long record)
    raw = load_das_raw("continuous.h5")
    raw_bp = preprocess_raw(raw, filter_type="bandpass",
                            freq_min=1.0, freq_max=20.0,
                            demean=True, detrend=True)
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Literal, Optional

import numpy as np
from scipy import signal

from dasfm.io.data import DASData, DASRawData


def preprocess(
    data: DASData,
    filter_type: Optional[Literal["bandpass", "highpass", "lowpass"]] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    filter_order: int = 4,
    demean: bool = False,
    detrend: bool = False,
) -> DASData:
    """Apply demean → detrend → zero-phase filter to a :class:`~dasfm.io.data.DASData`.

    Returns a **new** ``DASData`` instance; the original is never modified.
    All-zero channels (e.g. previously NaN-fixed) are skipped so they cannot
    bias neighbouring channels during filtering.

    Parameters
    ----------
    data : DASData
        Raw or previously loaded DAS data.
    filter_type : {'bandpass', 'highpass', 'lowpass'} or None
        Butterworth filter type.  ``None`` skips filtering entirely.
    freq_min : float, optional
        Low-cut frequency in Hz.  Required for ``'bandpass'`` and ``'highpass'``.
    freq_max : float, optional
        High-cut frequency in Hz.  Required for ``'bandpass'`` and ``'lowpass'``.
    filter_order : int
        Filter order (default ``4``).
    demean : bool
        Remove per-channel mean before filtering (default ``False``).
    detrend : bool
        Remove linear trend before filtering (default ``False``).

    Returns
    -------
    DASData
        New instance with processed ``p_data`` and ``s_data``.
        All other fields (SNR, traveltime, metadata, valid_mask …) are
        copied unchanged from *data*.

    Raises
    ------
    ValueError
        If *filter_type* is specified without the required corner frequencies.
    """
    validate_filter_args(filter_type, freq_min, freq_max)

    fs = data.sampling_rate
    warn_short_window(data.p_data.shape[1], fs, filter_type, freq_min)

    p_proc = apply_filter(data.p_data, fs, demean, detrend,
                    filter_type, freq_min, freq_max, filter_order)
    s_proc = apply_filter(data.s_data, fs, demean, detrend,
                    filter_type, freq_min, freq_max, filter_order)

    # record what was applied
    proc_meta = dict(data.metadata)
    proc_meta["preprocess"] = {
        "filter_type": filter_type,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "filter_order": filter_order,
        "demean": demean,
        "detrend": detrend,
    }

    return dataclasses.replace(data, p_data=p_proc, s_data=s_proc,
                               metadata=proc_meta)


def preprocess_raw(
    data: DASRawData,
    filter_type: Optional[Literal["bandpass", "highpass", "lowpass"]] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    filter_order: int = 4,
    demean: bool = False,
    detrend: bool = False,
) -> DASRawData:
    """Apply demean → detrend → zero-phase filter to a :class:`~dasfm.io.data.DASRawData`.

    Returns a **new** ``DASRawData`` instance; the original is never modified.
    Bandpass filtering is appropriate here because continuous traces are
    typically long enough (tens of seconds) to avoid short-window edge effects.

    Parameters
    ----------
    data : DASRawData
        Continuous DAS trace loaded by :func:`~dasfm.io.das_io.load_das_raw`.
    filter_type : {'bandpass', 'highpass', 'lowpass'} or None
        Butterworth filter type.  ``None`` skips filtering entirely.
    freq_min : float, optional
        Low-cut frequency in Hz.  Required for ``'bandpass'`` and ``'highpass'``.
    freq_max : float, optional
        High-cut frequency in Hz.  Required for ``'bandpass'`` and ``'lowpass'``.
    filter_order : int
        Filter order (default ``4``).
    demean : bool
        Remove per-channel mean before filtering (default ``False``).
    detrend : bool
        Remove linear trend before filtering (default ``False``).

    Returns
    -------
    DASRawData
        New instance with processed ``waveforms``.
        All other fields (metadata, valid_mask, coords …) are copied unchanged.

    Raises
    ------
    ValueError
        If *filter_type* is specified without the required corner frequencies.
    """
    validate_filter_args(filter_type, freq_min, freq_max)

    fs = data.sampling_rate
    warn_short_window(data.n_samples, fs, filter_type, freq_min)

    waveforms_proc = apply_filter(data.waveforms, fs, demean, detrend,
                            filter_type, freq_min, freq_max, filter_order)

    proc_meta = dict(data.metadata)
    proc_meta["preprocess"] = {
        "filter_type": filter_type,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "filter_order": filter_order,
        "demean": demean,
        "detrend": detrend,
    }

    return dataclasses.replace(data, waveforms=waveforms_proc,
                               metadata=proc_meta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def warn_short_window(
    n_samples: int,
    fs: float,
    filter_type: Optional[str],
    freq_min: Optional[float],
) -> None:
    """Warn if the window is too short for stable high-pass / bandpass filtering.

    Rule of thumb: the window should contain at least 4 full cycles of the
    low-cut frequency.  For pre-extracted P/S windows, consider using
    ``lowpass`` only (with ``demean=True, detrend=True``).
    """
    if filter_type not in ("bandpass", "highpass") or freq_min is None:
        return
    window_duration = n_samples / fs          # seconds
    min_cycles = 4.0 / freq_min              # seconds needed for 4 cycles
    if window_duration < min_cycles:
        warnings.warn(
            f"Window is {window_duration:.2f} s but {freq_min} Hz high-pass "
            f"needs ≥{min_cycles:.1f} s (4 cycles) for stable filtering. "
            f"Edge-effect artefacts (low-frequency upturn) are likely. "
            f"For short pre-extracted windows, use filter_type='lowpass' "
            f"with demean=True, detrend=True instead.",
            UserWarning,
            stacklevel=3,
        )


def validate_filter_args(
    filter_type: Optional[str],
    freq_min: Optional[float],
    freq_max: Optional[float],
) -> None:
    if filter_type is None:
        return
    if filter_type in ("bandpass", "highpass") and freq_min is None:
        raise ValueError(f"filter_type='{filter_type}' requires freq_min.")
    if filter_type in ("bandpass", "lowpass") and freq_max is None:
        raise ValueError(f"filter_type='{filter_type}' requires freq_max.")


def apply_filter(
    data: np.ndarray,
    fs: float,
    demean: bool,
    detrend: bool,
    filter_type: Optional[str],
    freq_min: Optional[float],
    freq_max: Optional[float],
    filter_order: int,
) -> np.ndarray:
    """Apply demean → detrend → filter to ``(n_channels, n_samples)``."""
    data = data.copy()
    active = np.any(data != 0, axis=1)   # skip all-zero rows

    if demean:
        data[active] -= data[active].mean(axis=1, keepdims=True)

    if detrend:
        data[active] = signal.detrend(data[active], axis=1)

    if filter_type is not None:
        sos = build_sos_filter(filter_type, freq_min, freq_max, filter_order, fs)
        data[active] = signal.sosfiltfilt(sos, data[active], axis=1)

    return data


def build_sos_filter(
    filter_type: str,
    freq_min: Optional[float],
    freq_max: Optional[float],
    order: int,
    fs: float,
) -> np.ndarray:
    nyq = fs / 2.0
    if filter_type == "bandpass":
        Wn = [freq_min / nyq, freq_max / nyq]
    elif filter_type == "highpass":
        Wn = freq_min / nyq
    else:
        Wn = freq_max / nyq
    return signal.butter(order, Wn, btype=filter_type, output="sos")


# ─────────────────────────────────────────────────────────────────────────────
#  Window cutting
# ─────────────────────────────────────────────────────────────────────────────

def cut_window(waveforms, arr_idx, cut_half):
    """Cut symmetric window around arrival indices.

    Parameters
    ----------
    waveforms : (n_ch, n_t) float32 — filtered waveform matrix.
    arr_idx : (n_ch,) int64 — arrival sample indices (-1 = missing pick).
    cut_half : int — half-window size in samples.

    Returns
    -------
    out : (n_ch, 2*cut_half) float32 — windowed waveforms.
        Channels with missing picks (arr_idx < 0) or out-of-range
        windows are filled with zeros.
    """
    n_ch, n_t = waveforms.shape
    n_win = 2 * cut_half
    out = np.zeros((n_ch, n_win), dtype=np.float32)
    for i in range(n_ch):
        t1 = arr_idx[i] - cut_half
        t2 = arr_idx[i] + cut_half
        if arr_idx[i] >= 0 and 0 <= t1 and t2 <= n_t:
            out[i] = waveforms[i, t1:t2]
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Phase pick conditioning  (operates on float seconds, NaN = missing)
#
#  Pipeline (run in order in step2a_window.process_single_event):
#    1. reject_outlier_picks  — replace points far from a local median by NaN
#    2. interpolate_missing_picks — linear-interp NaN gaps between head/tail
#    3. smooth_picks          — uniform_filter1d on the inner valid segment
#  Then divide by dt and round to integer sample indices for window cutting.
#
#  All three helpers take and return float arrays (NaN = missing) so the
#  sub-sample precision of phase_index*dt is preserved through the chain.
# ═══════════════════════════════════════════════════════════════════════════

def reject_outlier_picks(arr, medfilt_window=100, residual_threshold=0.5):
    """Mark picks far from a local median as missing (NaN).

    Replaces the previous "blanket median filter" step that overwrote
    every channel with the local median.  Now only channels whose
    residual to the local median exceeds *residual_threshold* (in
    seconds, after subtracting the median trend) are flagged as NaN,
    so the surrounding good picks are passed through unchanged and the
    later interpolation step can fill the rejected positions cleanly.

    Parameters
    ----------
    arr : (n_ch,) float
        Pick times in seconds.  NaN = already missing.
    medfilt_window : int
        Median filter kernel size (in channels) used to estimate the
        local trend.  Default 100 (matches the previous behaviour).
    residual_threshold : float
        Channels with ``|arr - local_median| > residual_threshold`` are
        marked NaN.  Default 0.5 s — generous enough that real fiber
        bends do not get clipped, but tight enough to catch isolated
        mis-picks (PhaseNet S labelled as P, etc.).

    Returns
    -------
    (n_ch,) float
        Copy of *arr* with outliers replaced by NaN.  Channels that
        were already NaN stay NaN.
    """
    valid = ~np.isnan(arr)
    if not valid.any():
        return arr.copy()
    from scipy.ndimage import median_filter
    result = arr.copy()
    # median_filter ignores NaN poorly — fill missing with the global
    # median temporarily for the trend estimate, but only update the
    # *originally valid* positions back into result.
    filled = arr.copy()
    filled[~valid] = float(np.nanmedian(arr))
    trend = median_filter(filled, size=min(medfilt_window, len(arr)))
    residual = np.abs(result - trend)
    bad = valid & (residual > residual_threshold)
    result[bad] = np.nan
    return result


def interpolate_missing_picks(arr):
    """Interpolate missing picks between first and last valid channel.

    Parameters
    ----------
    arr : (n_ch,) float
        Pick times (any unit; seconds in step2a).  NaN = missing.

    Returns
    -------
    (n_ch,) float
        Gaps between head and tail filled by linear interpolation.
        Channels before first valid / after last valid remain NaN.
    """
    valid = ~np.isnan(arr)
    if valid.all() or not valid.any():
        return arr.copy()
    ch_idx = np.arange(len(arr))
    valid_idx = ch_idx[valid]
    head, tail = valid_idx[0], valid_idx[-1]
    result = arr.copy()
    inner = ch_idx[(ch_idx >= head) & (ch_idx <= tail)]
    result[inner] = np.interp(inner, valid_idx, arr[valid_idx])
    return result


def smooth_picks(arr, smooth_window=20):
    """Smooth pick array on the valid range with a uniform filter.

    No median filter here — outlier rejection now happens upstream in
    :func:`reject_outlier_picks` before interpolation, so this step is
    purely a final low-pass to remove residual jitter.

    Parameters
    ----------
    arr : (n_ch,) float
        Pick times (any unit).  NaN = missing outside valid range.
    smooth_window : int
        Uniform smoothing kernel size (in channels).

    Returns
    -------
    (n_ch,) float
        Smoothed.  Channels outside ``[head, tail]`` remain NaN.
    """
    valid = ~np.isnan(arr)
    if not valid.any():
        return arr.copy()
    from scipy.ndimage import uniform_filter1d
    ch_idx = np.arange(len(arr))
    valid_idx = ch_idx[valid]
    head, tail = valid_idx[0], valid_idx[-1]
    result = arr.copy()
    seg = result[head:tail+1].astype(float)
    seg = uniform_filter1d(seg, size=min(smooth_window, len(seg)))
    result[head:tail+1] = seg
    return result
