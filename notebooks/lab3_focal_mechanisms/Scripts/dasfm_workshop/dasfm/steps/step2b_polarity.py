"""step2b_polarity — DAS polarity via MCCC + SVD (thin dispatcher).

This file is a routing layer only.  It validates inputs, normalises the
``device`` argument, then dispatches to one of three backends:

* :mod:`dasfm.steps.step2b_polarity_serial` — single CPU or single GPU
* :mod:`dasfm.steps.step2b_polarity_cpus`   — fork Pool, multi-CPU
* :mod:`dasfm.steps.step2b_polarity_gpus`   — ThreadPoolExecutor, multi-GPU

All shared state lives in :class:`dasfm.picking.mccc_context.Step2bContext`,
all shared logic lives in :func:`dasfm.picking.mccc_context.run_with_iteration`
and :func:`dasfm.picking.polarity_postprocess.postprocess`.  This dispatcher
contains **no** MCCC algorithm, sparse iteration, postprocess, or backend
conditionals beyond the single ``if/elif/elif/else`` route.
"""
from __future__ import annotations

from pathlib import Path

from dasfm.utils.step_utils import resolve_path, resolve_device
from dasfm.io.das_fft import validate_das_fft_dir
from dasfm.io.das_io import validate_das_geo
from dasfm.io.event_catalog_io import validate_event_catalog, validate_per_event_files
from dasfm.io.sta_io import validate_sta_polarity, validate_sta_geo


def run(
    project_dir="",
    event_catalog="",
    das_fft="",
    das_geo="",
    sta4mccc_ref=None,
    sta_polarity=None,
    sta_geo=None,
    mccc_max_lag_sec=0.5,
    mccc_maxwin=10,
    mccc_damp=1.0,
    mccc_max_shift=2,
    polarity_smooth_window=10,
    polarity_method="hilbert",
    cal_min_picks=5,
    mccc_cache_path=None,
    pol_out_path=None,
    sparse=False,
    k_neighbors=10,
    top_xcorr_frac=0.1,
    n_remote=3,
    xcorr_subsample=10,
    stability_threshold=0.99,
    device="cpu",
    num_cpu_workers=1,
    show_plots=False,
):
    """DAS polarity via MCCC + SVD.  Reads only from das_fft cache.

    Parameters
    ----------
    project_dir, event_catalog, das_fft, das_geo, pol_out_path : str
        Required.  ``das_fft`` is the per-event FFT cache produced by
        ``step2a_window``; step2b never reads ``das_win``.
    sta4mccc_ref : str or None
        Pre-computed calibration CSV with columns: ``event_id, network,
        station, location, channel, p_polarity, closest_das_ch, latitude,
        longitude``.  If None, ``sta_polarity + sta_geo`` are required so
        calibration can be auto-computed.
    polarity_method : "direct" | "hilbert"
        Hybrid sums P + L + R Hilbert components in MCCC; requires
        ``has_lr=True`` in the das_fft cache.
    sparse : bool
        Sparse pair selection mode (precompute_pair_rankings + iterative
        stability check).
    device : "cpu" | "cuda:N" | list[str]
        Single GPU, multi GPU, or CPU.
    num_cpu_workers : int
        Multi-CPU fork Pool size; ignored when GPU is in use.
    mccc_cache_path : str or None
        If given, the dense Ckij/Skij pairwise cross-correlation matrices
        are saved to this path as an HDF5 file with a single dataset
        ``pol_LR_mccc_shift`` of shape ``(n_ch, n_ev, n_ev, 2)``.
    """
    import pandas as pd

    # ── 1. Required-param check ──────────────────────────────────────────
    _required = {
        "project_dir": project_dir,
        "event_catalog": event_catalog,
        "das_fft": das_fft,
        "das_geo": das_geo,
        "pol_out_path": pol_out_path,
    }
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")

    # ── 2. Normalize device → ("cpu" | "cuda:N" | list[str]) ─────────────
    device, num_gpu, _ = resolve_device(device)

    # ── 3. Pre-flight validation ─────────────────────────────────────────
    root = Path(project_dir).resolve()
    _cat_path     = resolve_path(event_catalog, root)
    _das_geo_path = resolve_path(das_geo, root)
    _das_fft_path = resolve_path(das_fft, root)

    validate_event_catalog(_cat_path)
    validate_das_geo(_das_geo_path)
    validate_das_fft_dir(
        _das_fft_path,
        require_lr=(polarity_method == "hilbert"),
    )
    _eids = pd.read_csv(_cat_path)["event_id"].astype(str).tolist()
    validate_per_event_files(
        _eids, _das_fft_path, ".h5",
        label="DAS FFT cache", upstream_step="step2a_window",
    )

    # Calibration source: sta4mccc_ref OR (sta_polarity + sta_geo)
    if sta4mccc_ref:
        validate_sta_polarity(resolve_path(sta4mccc_ref, root))
    elif sta_polarity and sta_geo:
        validate_sta_polarity(resolve_path(sta_polarity, root))
        validate_sta_geo(resolve_path(sta_geo, root))
    else:
        raise SystemExit(
            "  ERROR — Cannot correct polarity sign. Either pass "
            "sta4mccc_ref=<existing cal CSV> or both sta_polarity and sta_geo "
            "(to auto-compute the calibration)."
        )

    # ── 4. Build common kwargs for backend ───────────────────────────────
    kwargs = dict(
        project_dir=project_dir,
        event_catalog=event_catalog,
        das_fft=das_fft,
        das_geo=das_geo,
        sta4mccc_ref=sta4mccc_ref,
        sta_polarity=sta_polarity,
        sta_geo=sta_geo,
        mccc_max_lag_sec=mccc_max_lag_sec,
        mccc_maxwin=mccc_maxwin,
        mccc_damp=mccc_damp,
        mccc_max_shift=mccc_max_shift,
        polarity_smooth_window=polarity_smooth_window,
        polarity_method=polarity_method,
        cal_min_picks=cal_min_picks,
        mccc_cache_path=mccc_cache_path,
        pol_out_path=pol_out_path,
        sparse=sparse,
        k_neighbors=k_neighbors,
        top_xcorr_frac=top_xcorr_frac,
        n_remote=n_remote,
        xcorr_subsample=xcorr_subsample,
        stability_threshold=stability_threshold,
        device=device,
        num_cpu_workers=num_cpu_workers,
        show_plots=show_plots,
    )

    # ── 5. Route to backend (single if/elif chain — no further branching) ──
    # Priority: multi-GPU → single GPU (serial) → multi-CPU → single CPU (serial)
    # GPU always wins over CPU pool, even when num_cpu_workers > 1.
    if num_gpu > 1:
        from dasfm.steps.step2b_polarity_gpus import run as _run
    elif num_gpu == 1:
        from dasfm.steps.step2b_polarity_serial import run as _run
    elif num_cpu_workers > 1:
        from dasfm.steps.step2b_polarity_cpus import run as _run
    else:
        from dasfm.steps.step2b_polarity_serial import run as _run

    _run(**kwargs)
