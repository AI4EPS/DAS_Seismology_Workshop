"""step2a_window — DAS waveform windowing via Hilbert separation (CPU or GPU)."""

from __future__ import annotations

import time as _time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey as _tukey

from dasfm.utils.step_utils import Logger, resolve_path, resolve_device
from tqdm import tqdm
from dasfm.io.das_io import load_das_raw, validate_das_raw, validate_das_geo, validate_phase_picks
from dasfm.io.event_catalog_io import validate_event_catalog, validate_per_event_files
from dasfm.io.middle_io import write_win_middle
from dasfm.processing.preprocess import (
    preprocess_raw, reject_outlier_picks, interpolate_missing_picks, smooth_picks,
)


def _normalize_and_fft(arr):
    """Normalize a (n_ch, n_t) waveform tensor and return its rfft.

    Pipeline (must match :func:`dasfm.picking.mccc.xcorr_from_freq` input
    contract):
        1. global RMS normalize
        2. Tukey 0.8 taper + tiny offset
        3. per-channel demean + L2 normalize
        4. next-pow-2 rfft (axis=-1) → complex64

    Returns
    -------
    freqs : np.ndarray
        complex64 of shape ``(n_ch, nfast//2 + 1)``.
    nfast : int
        Full FFT length (next power of two ≥ ``2 * n_t - 1``).
    """
    arr = np.asarray(arr, dtype=np.float32)
    rms = np.sqrt(np.mean(arr ** 2))
    if rms > 0:
        arr = arr / rms
    taper = _tukey(arr.shape[-1], 0.8).astype(np.float32)
    arr = arr * taper + 1e-6
    arr = arr - arr.mean(axis=-1, keepdims=True)
    norms = np.sqrt((arr ** 2).sum(axis=-1, keepdims=True))
    norms[norms == 0] = 1.0
    arr = arr / norms
    nfast = 1
    while nfast < 2 * arr.shape[-1] - 1:
        nfast *= 2
    return np.fft.rfft(arr, n=nfast, axis=-1).astype(np.complex64), nfast


def process_single_event(args):
    """Process a single event for step2a (worker function for multiprocessing)."""
    (eid, raw_data_dir, raw_time_dir, das_win_dir, das_fft_dir,
     good_ch_idx, n_good, freq_min, freq_max,
     cut_half_sec, hilbert, use_gpu, device,
     medfilt_window, smooth_window) = args

    h5_path  = raw_data_dir / f"{eid}.h5"
    csv_path = raw_time_dir / f"{eid}.csv"

    raw = load_das_raw(h5_path)
    n_raw = raw.waveforms.shape[0]
    if n_raw > n_good:
        raw.waveforms  = raw.waveforms[good_ch_idx]
        raw.valid_mask = raw.valid_mask[good_ch_idx]

    filt = preprocess_raw(raw, filter_type="bandpass",
                          freq_min=freq_min, freq_max=freq_max)

    dt_s = raw.dt
    event_time_idx = raw.event_time_index

    picks = pd.read_csv(csv_path)
    p_map = picks[picks["phase_type"] == "P"].set_index("channel_index")["phase_index"]
    s_map = picks[picks["phase_type"] == "S"].set_index("channel_index")["phase_index"]

    # ── Pick conditioning in seconds (NaN = missing) ─────────────────────
    # 1. Build float-second arrays from CSV phase_index, NaN where absent.
    # 2. Outlier rejection by residual to local median (turns mis-picks
    #    into NaN so the next step can fill them).
    # 3. Linear interpolate NaN gaps between first and last valid channel.
    # 4. Boxcar smooth the inner segment to remove residual jitter.
    # The whole chain stays in seconds — only divided by dt at the very
    # end when we need integer sample indices for window cutting.
    def _build_sec_array(pmap):
        out = np.full(n_good, np.nan, dtype=np.float64)
        for i in range(n_good):
            if i in pmap.index:
                out[i] = float(pmap[i]) * dt_s
        return out

    p_sec = _build_sec_array(p_map)
    s_sec = _build_sec_array(s_map)

    p_sec = reject_outlier_picks(p_sec, medfilt_window=medfilt_window)
    s_sec = reject_outlier_picks(s_sec, medfilt_window=medfilt_window)
    p_sec = interpolate_missing_picks(p_sec)
    s_sec = interpolate_missing_picks(s_sec)
    p_sec = smooth_picks(p_sec, smooth_window=smooth_window)
    s_sec = smooth_picks(s_sec, smooth_window=smooth_window)

    # Per-channel travel time (seconds, relative to event origin) — kept
    # as the absolute, sub-sample-accurate value for the das_win H5 field.
    p_tt = p_sec - event_time_idx * dt_s
    s_tt = s_sec - event_time_idx * dt_s

    # Convert seconds → integer sample indices for waveform cutting.
    # NaN (= missing or out-of-range) becomes -1, matching cut_window's
    # missing-pick sentinel.
    def _sec_to_int_idx(sec):
        out = np.full(sec.shape, -1, dtype=np.int64)
        valid = ~np.isnan(sec)
        out[valid] = np.round(sec[valid] / dt_s).astype(np.int64)
        return out

    p_arr = _sec_to_int_idx(p_sec)
    s_arr = _sec_to_int_idx(s_sec)

    cut_half = round(cut_half_sec / raw.dt)

    # Trim waveforms to P-2s ~ P+8s to reduce Hilbert memory (complex128)
    valid_p = p_arr[p_arr >= 0]
    if len(valid_p) > 0:
        pre_sec, post_sec = 2.0, 8.0
        trim_start = max(0, int(valid_p.min() - pre_sec / raw.dt))
        trim_end = min(filt.waveforms.shape[1], int(valid_p.max() + post_sec / raw.dt))
        filt.waveforms = filt.waveforms[:, trim_start:trim_end]
        p_arr = np.where(p_arr >= 0, p_arr - trim_start, -1)
        s_arr = np.where(s_arr >= 0, s_arr - trim_start, -1)

    # Drop picks whose cut_half window would extend past the data range,
    # so the cut step doesn't silently zero-pad out-of-bound samples.
    n_t_trim = filt.waveforms.shape[1]
    p_arr = np.where((p_arr >= cut_half) & (p_arr + cut_half < n_t_trim),
                     p_arr, -1)
    s_arr = np.where((s_arr >= cut_half) & (s_arr + cut_half < n_t_trim),
                     s_arr, -1)

    if hilbert:
        if use_gpu:
            from dasfm.processing.hilbert_gpu import hilbert_separate_gpu as hilbert_fn
        else:
            from dasfm.processing.hilbert import hilbert_separate as hilbert_fn
        if use_gpu:
            res = hilbert_fn(waveforms=filt.waveforms, p_arr=p_arr, s_arr=s_arr,
                             dt=raw.dt, cut_half=cut_half, device=device)
        else:
            res = hilbert_fn(waveforms=filt.waveforms, p_arr=p_arr, s_arr=s_arr,
                             dt=raw.dt, cut_half=cut_half)
    else:
        from dasfm.processing.preprocess import cut_window
        res = {
            "p_original": cut_window(filt.waveforms, p_arr, cut_half),
            "p_right": None, "p_left": None,
            "s_original": cut_window(filt.waveforms, s_arr, cut_half),
            "s_right": None, "s_left": None,
            "taper_weights_x": None, "taper_n_x": 0,
        }

    p_valid = (p_arr >= 0)
    s_valid = (s_arr >= 0)

    write_win_middle(
        out_path=das_win_dir / f"{eid}.h5",
        hilbert_res={
            "p_original": res["p_original"], "p_right": res["p_right"],
            "taper_weights_x": res["taper_weights_x"],
            "taper_n_x": res["taper_n_x"],
            "p_left": res["p_left"], "s_original": res["s_original"],
            "s_right": res["s_right"], "s_left": res["s_left"],
            "cut_half": cut_half,
        },
        p_traveltime=p_tt, s_traveltime=s_tt, dt=raw.dt,
        event_id=eid, event_time=raw.event_time or "",
        magnitude=raw.magnitude, p_valid=p_valid, s_valid=s_valid,
    )

    # Pre-compute and save FFT cache for step2b MCCC.  step2b reads only this
    # cache — it never recomputes FFTs from das_win.  Hybrid mode therefore
    # requires us to also persist the left/right (Hilbert) FFTs here.  All
    # event metadata (dt, has_lr, p_valid, event_id) lives alongside so that
    # step2b is fully self-sufficient and never has to touch das_win.
    freqs_p, nfast = _normalize_and_fft(res["p_original"])
    has_lr = (
        bool(hilbert)
        and res.get("p_left") is not None
        and res.get("p_right") is not None
    )
    if has_lr:
        freqs_left,  _ = _normalize_and_fft(res["p_left"])
        freqs_right, _ = _normalize_and_fft(res["p_right"])

    with h5py.File(das_fft_dir / f"{eid}.h5", "w") as f:
        f.create_dataset("freqs_p", data=freqs_p)
        f.create_dataset("p_valid", data=p_valid.astype(bool))
        f.attrs["nfast"]    = nfast
        f.attrs["dt"]       = float(raw.dt)
        f.attrs["has_lr"]   = has_lr
        f.attrs["event_id"] = eid
        if has_lr:
            f.create_dataset("freqs_left",  data=freqs_left)
            f.create_dataset("freqs_right", data=freqs_right)

    return {
        "eid": eid, "status": "ok",
        "n_ch": filt.waveforms.shape[0], "dt": raw.dt,
        "taper_n": res["taper_n_x"], "magnitude": raw.magnitude,
        "p_count": int(p_valid.sum()), "s_count": int(s_valid.sum()),
    }


def run(
    project_dir="",
    event_catalog="",
    das_raw_data="",
    das_raw_time="",
    das_geo="",
    das_win="",
    das_fft="",
    bandpass_freq_min_hz=1.0,
    bandpass_freq_max_hz=10.0,
    window_half_duration_sec=1.0,
    hilbert=True,
    device="cpu",
    num_cpu_workers=1,
    show_plots=False):
    """Load raw DAS data, bandpass filter, save windowed P/S arrivals.

    Parameters
    ----------
    das_win : str
        Output directory for windowed time-domain H5 files (consumed by step2c).
    das_fft : str
        Output directory for per-event FFT cache (consumed by step2b).  Holds
        ``freqs_p`` always, plus ``freqs_left`` / ``freqs_right`` when
        ``hilbert=True``.  Each file also persists ``dt``, ``has_lr``,
        ``event_id`` and ``p_valid`` so step2b is fully self-sufficient.
    hilbert : bool
        If True (default), apply 2-D Hilbert wave separation and save
        right/left-going components.  If False, only cut windows around
        P/S arrivals; p_right/p_left/s_right/s_left are saved as zeros.
    device : str or list[str]
        "cpu", "cuda:N", or ["cuda:0", "cuda:1", ...].
        Multi-GPU: events are distributed across GPUs (one thread per GPU).

    Outputs:
        {das_win}/{event_id}.h5
        {das_fft}/{event_id}.h5
        cache/figs/stage2a/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog, "das_raw_data": das_raw_data, "das_raw_time": das_raw_time, "das_geo": das_geo, "das_win": das_win, "das_fft": das_fft}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    root = Path(project_dir).resolve()
    logger = Logger("step2a_window", log_dir=str(root / "logs"))

    CAT_FILE     = resolve_path(event_catalog, root)
    RAW_DATA_DIR = resolve_path(das_raw_data,  root)
    RAW_TIME_DIR = resolve_path(das_raw_time,  root)
    DAS_INFO     = resolve_path(das_geo,       root)
    DAS_WIN_DIR  = resolve_path(das_win,       root)
    DAS_FFT_DIR  = resolve_path(das_fft,       root)

    # ── Pre-flight: validate every input file/directory we will read ─────
    validate_event_catalog(CAT_FILE)
    validate_das_geo(DAS_INFO)
    _eids_pf = pd.read_csv(CAT_FILE)["event_id"].astype(str).tolist()
    validate_per_event_files(_eids_pf, RAW_DATA_DIR, ".h5",
                             label="DAS raw H5")
    validate_per_event_files(_eids_pf, RAW_TIME_DIR, ".csv",
                             label="DAS phase picks CSV")
    # Spot-check structure of the first per-event H5 + picks CSV; per-event
    # full validation would re-open all files, which is wasteful.
    if _eids_pf:
        validate_das_raw(RAW_DATA_DIR / f"{_eids_pf[0]}.h5")
        validate_phase_picks(RAW_TIME_DIR / f"{_eids_pf[0]}.csv")

    # Normalize device → canonical form (also accepts ["cuda:3"], ["cpu"], None, ...)
    _, num_gpu, gpu_devices = resolve_device(device)
    use_gpu = hilbert and num_gpu > 0
    primary_device = gpu_devices[0] if gpu_devices else "cpu"
    mode = ("GPU" if use_gpu else "CPU") if hilbert else "no Hilbert"
    t0 = _time.time()
    logger.info()
    logger.info("=" * 60)
    logger.info()
    logger.info(f"  step2a_window — windowed DAS waveforms ({mode})")
    logger.info()
    logger.info("=" * 60)
    logger.info("[1/2] Process & save windowed waveforms")

    catalog   = pd.read_csv(CAT_FILE)
    EVENT_IDS = [str(eid) for eid in catalog["event_id"].values]

    das_geo_df = pd.read_csv(DAS_INFO)
    good_ch_idx = das_geo_df["index"].values.astype(int)
    n_good      = len(good_ch_idx)
    logger.info(f"  {'Events':<12}: {len(EVENT_IDS)}  (from {CAT_FILE})")
    logger.info(f"  {'Channels':<12}: {n_good}  (index {good_ch_idx[0]}–{good_ch_idx[-1]})")
    if use_gpu:
        logger.info(f"  {'Device':<12}: {gpu_devices}")

    FREQ_MIN     = bandpass_freq_min_hz
    FREQ_MAX     = bandpass_freq_max_hz
    CUT_HALF_SEC = window_half_duration_sec
    pick_medfilt_window = 100
    pick_smooth_window  = 20

    DAS_WIN_DIR.mkdir(parents=True, exist_ok=True)
    DAS_FFT_DIR.mkdir(parents=True, exist_ok=True)

    dt = n_ch = taper_n = None
    n_processed = 0
    first_eid = None
    p_counts  = []
    s_counts  = []

    # Parallel strategy:
    #   NUM_CPU=1, GPU≤1  → serial (GPU Hilbert if available)
    #   NUM_CPU>1, GPU≤1  → Pool(NUM_CPU), force CPU Hilbert (faster than 1 GPU)
    #   GPU>1             → Thread(NUM_GPU), GPU Hilbert per thread

    def _make_args(eid, dev, force_cpu=False):
        _use_gpu = hilbert and not force_cpu and isinstance(dev, str) and dev.startswith("cuda")
        return (eid, RAW_DATA_DIR, RAW_TIME_DIR, DAS_WIN_DIR, DAS_FFT_DIR,
                good_ch_idx, n_good, FREQ_MIN, FREQ_MAX,
                CUT_HALF_SEC, hilbert, _use_gpu, dev,
                pick_medfilt_window, pick_smooth_window)

    if num_gpu > 1:
        # Multi-GPU: distribute events, one thread per GPU, GPU Hilbert
        from concurrent.futures import ThreadPoolExecutor
        gpu_event_lists = [[] for _ in range(num_gpu)]
        for i, eid in enumerate(EVENT_IDS):
            gpu_event_lists[i % num_gpu].append(eid)
        logger.info(f"  Parallel: {num_gpu} GPUs, "
                    + ", ".join(f"{gpu_devices[g]}={len(gpu_event_lists[g])}ev" for g in range(num_gpu)))

        pbar = tqdm(total=len(EVENT_IDS), desc="step2a_window", unit="ev", leave=True)

        def gpu_thread(gpu_id, gpu_dev, eids):
            thread_results = []
            for eid in eids:
                thread_results.append(process_single_event(_make_args(eid, gpu_dev)))
                pbar.update(1)
            return thread_results

        all_thread_results = []
        with ThreadPoolExecutor(max_workers=num_gpu) as executor:
            futures = [executor.submit(gpu_thread, gi, gpu_devices[gi], gpu_event_lists[gi])
                       for gi in range(num_gpu)]
            for f in futures:
                all_thread_results.extend(f.result())
        pbar.close()
        results = all_thread_results

    elif num_cpu_workers > 1:
        # Multi-CPU: Pool parallel, force CPU Hilbert (ignore single GPU)
        import multiprocessing
        logger.info(f"  Parallel: {num_cpu_workers} CPU workers (CPU Hilbert)")
        worker_args = [_make_args(eid, "cpu", force_cpu=True) for eid in EVENT_IDS]
        with multiprocessing.Pool(num_cpu_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_event, worker_args),
                total=len(worker_args), desc="step2a_window", unit="ev"))

    else:
        # Serial: single GPU or single CPU
        worker_args = [_make_args(eid, primary_device) for eid in EVENT_IDS]
        results = []
        for args in tqdm(worker_args, desc="step2a_window", unit="ev"):
            results.append(process_single_event(args))

    for r in results:
        n_processed += 1
        if dt is None:
            dt = r["dt"]
            n_ch = r["n_ch"]
            taper_n = r["taper_n"]
        if first_eid is None:
            first_eid = r["eid"]
        p_counts.append(r["p_count"])
        s_counts.append(r["s_count"])
        logger.log(f"  {r['eid']}  {r['n_ch']} ch  M{r['magnitude']:.1f}  ok")

    logger.info(f"  Done: {n_processed} ok")
    logger.info(f"  → {DAS_WIN_DIR}")

    # ── QC plots ──────────────────────────────────────────────────────────────
    logger.info("[2/2] QC plots")
    if n_processed > 0 and first_eid is not None:
        import matplotlib
        import matplotlib.pyplot as plt
        from dasfm.io.middle_io import load_win_middle

        fig_dir = root / "cache/figs/stage2a"
        fig_dir.mkdir(parents=True, exist_ok=True)

        mid_data = load_win_middle(DAS_WIN_DIR / f"{first_eid}.h5")
        n_plots = 0

        # P windows
        vmax_p = np.nanpercentile(np.abs(mid_data["p_original"]), 99) or 1.0
        if hilbert:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=150)
            for ax, data, lbl in zip(axes,
                                      [mid_data["p_original"], mid_data["p_right"], mid_data["p_left"]],
                                      ["P original", "P right", "P left"]):
                im = ax.imshow(data.T, aspect="auto", cmap="seismic",
                               vmin=-vmax_p, vmax=vmax_p, origin="upper")
                ax.set_title(lbl); ax.set_xlabel("Channel"); ax.set_ylabel("Sample")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
            im = ax.imshow(mid_data["p_original"].T, aspect="auto", cmap="seismic",
                           vmin=-vmax_p, vmax=vmax_p, origin="upper")
            ax.set_title("P original"); ax.set_xlabel("Channel"); ax.set_ylabel("Sample")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"P windows — event {first_eid}", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / f"hilbert_P_windows_{first_eid}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        if show_plots:
            from IPython.display import display, Image
            display(Image(filename=str(fig_dir / f"hilbert_P_windows_{first_eid}.png")))
        n_plots += 1

        # S windows
        vmax_s = np.nanpercentile(np.abs(mid_data["s_original"]), 99) or 1.0
        if hilbert:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=150)
            for ax, data, lbl in zip(axes,
                                      [mid_data["s_original"], mid_data["s_right"], mid_data["s_left"]],
                                      ["S original", "S right", "S left"]):
                im = ax.imshow(data.T, aspect="auto", cmap="seismic",
                               vmin=-vmax_s, vmax=vmax_s, origin="upper")
                ax.set_title(lbl); ax.set_xlabel("Channel"); ax.set_ylabel("Sample")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
            im = ax.imshow(mid_data["s_original"].T, aspect="auto", cmap="seismic",
                           vmin=-vmax_s, vmax=vmax_s, origin="upper")
            ax.set_title("S original"); ax.set_xlabel("Channel"); ax.set_ylabel("Sample")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"S windows — event {first_eid}", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / f"hilbert_S_windows_{first_eid}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        if show_plots:
            from IPython.display import display, Image
            display(Image(filename=str(fig_dir / f"hilbert_S_windows_{first_eid}.png")))
        n_plots += 1

        # Taper weights (only when hilbert=True)
        if hilbert:
            taper_weights = mid_data["taper_weights_x"]
            if taper_weights is not None:
                ch_axis = np.arange(len(taper_weights))
                fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
                ax.plot(ch_axis, taper_weights, lw=1.2, color="steelblue")
                ax.set_xlabel("Channel index"); ax.set_ylabel("Taper weight")
                ax.set_title("Spatial taper weights (taper_weights_x)")
                ax.set_ylim(0, 1.05)
                fig.tight_layout()
                fig.savefig(fig_dir / "taper_weights.png", dpi=150, bbox_inches="tight")
                plt.close("all")
                if show_plots:
                    from IPython.display import display, Image
                    display(Image(filename=str(fig_dir / "taper_weights.png")))
                n_plots += 1

        logger.info(f"  → {fig_dir}  ({n_plots} plots)")

    logger.info()
    logger.info("=" * 60)
    logger.info()
    logger.info(f"  Done  ({_time.time() - t0:.1f} s)")
    logger.info()
    logger.info("=" * 60)
    logger.close()
