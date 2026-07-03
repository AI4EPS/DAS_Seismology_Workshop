"""step3_invert_serial — Focal mechanism inversion, serial mode (CPU/GPU).

Per-event work is delegated to ``dasfm.inversion.pipeline.process_events``.
This file defines the data loading + torch-state building, plus the entry
point ``run()`` that step3_invert.py calls.

Architecture
------------
The legacy monolithic ``setup()`` is split into two phases:

    _load_data(**kwargs) -> data dict     # Phase 1: pure I/O + numpy, NO torch
    _build_torch_state(data, device) -> ctx  # Phase 2: build per-device torch tensors

This split enables:
    1. step3_invert.py main process loads data ONCE via _load_data() (no torch)
    2. cpus.py forks workers — they inherit data via COW (zero copy)
    3. each worker / thread builds its own per-device torch state via _build_torch_state()

Fork-safety contract
--------------------
This file's **module-level imports** must be torch-free (so that
``from dasfm.steps.step3_invert_serial import _load_data`` doesn't trigger
torch import in the main process before fork).

    - dasfm.utils.step_utils       (torch-free)
    - dasfm.io.{ray_params,middle,sta_data,event_catalog}_io (torch-free)
    - dasfm.inversion.modes        (torch-free)
    - dasfm.inversion.grid         (torch-free, after Phase 6.5 fixes __init__.py)

The ``import torch`` only happens inside ``_build_torch_state()`` (function body)
and inside ``run()`` body via the lazy ``from dasfm.inversion.pipeline import process_events``.
"""
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from dasfm.utils.step_utils import (
    Logger, resolve_path, parse_inversion_types, find_result_dir,
)
from dasfm.io.data import RayParamTable
from dasfm.io.middle_io import load_das_polarity, load_das_sp_ratios
from dasfm.io.sta_io import load_sta_polarity, load_sta_sp_ratio
from dasfm.inversion.modes import mode_requires
from dasfm.inversion.grid import make_sdr_grid_skhash_style, sdr_to_mt


# ═══════════════════════════════════════════════════════════════════════════
#  Per-data-type loaders (extracted from legacy setup)
# ═══════════════════════════════════════════════════════════════════════════

def _load_das_polarity(need_das_pol, das_polarity, root, logger):
    """Load DAS polarity (Pkic) if needed.  Pure numpy."""
    Pkic = pol_valid = None
    pol_event_ids = pol_channel_ids = None
    HAS_DAS_POL = False
    if need_das_pol:
        if not das_polarity:
            raise SystemExit(
                "  ERROR — Inversion mode requires DAS polarity but "
                "`das_polarity` was not provided."
            )
        pol_path = resolve_path(das_polarity, root)
        if not pol_path.exists():
            raise SystemExit(f"  ERROR — DAS polarity file not found: {pol_path}")
        _pol = load_das_polarity(pol_path)
        Pkic = _pol["Pkic"]
        pol_event_ids = _pol["event_ids"]
        pol_channel_ids = _pol["channel_ids"]
        pol_valid = _pol["pol_valid"]
        HAS_DAS_POL = True
        logger.info(
            f"  DAS polarity loaded: {Pkic.shape}"
            + (f"  pol_valid: {pol_valid.sum(axis=0).min()}-{pol_valid.sum(axis=0).max()} ch/ev"
               if pol_valid is not None else "")
        )
    return dict(HAS_DAS_POL=HAS_DAS_POL, Pkic=Pkic, pol_valid=pol_valid,
                pol_event_ids=pol_event_ids, pol_channel_ids=pol_channel_ids)


def _load_das_sp(need_das_sp, das_sp_ratio, root, sp_norm, logger):
    """Load DAS S/P ratios + (optionally) compute Cauchy noise scale.  Pure numpy."""
    sp_ratios = sp_valid = None
    sp_event_ids = sp_channel_ids = None
    HAS_DAS_SP = False
    cauchy_c = None
    if need_das_sp:
        if not das_sp_ratio:
            raise SystemExit(
                "  ERROR — Inversion mode requires DAS S/P ratios but "
                "`das_sp_ratio` was not provided."
            )
        sp_path = resolve_path(das_sp_ratio, root)
        if not sp_path.exists():
            raise SystemExit(f"  ERROR — DAS S/P ratio file not found: {sp_path}")
        _sp = load_das_sp_ratios(sp_path)
        sp_ratios = _sp["sp_ratios"]
        sp_valid = _sp["sp_valid"]
        sp_event_ids = _sp["event_ids"]
        sp_channel_ids = _sp["channel_ids"]
        HAS_DAS_SP = True
        logger.info(f"  DAS SP ratios loaded: {sp_ratios.shape}")
        if sp_norm == "cauchy":
            from scipy.ndimage import median_filter
            _sigmas = []
            for _iev in range(sp_ratios.shape[1]):
                _col = sp_ratios[:, _iev].copy()
                _col[np.isnan(_col)] = 0.0
                _smoothed = median_filter(_col, size=200, mode='nearest')
                _res = (_col[sp_valid[:, _iev]] - _smoothed[sp_valid[:, _iev]]
                        if sp_valid is not None else _col - _smoothed)
                if len(_res) > 0:
                    _sigmas.append(1.4826 * np.median(np.abs(_res)))
            if _sigmas:
                cauchy_c = 2.0 * float(np.median(_sigmas))
                logger.info(f"  Cauchy scale: c = 2s = {cauchy_c:.4f}")
    return dict(HAS_DAS_SP=HAS_DAS_SP, sp_ratios=sp_ratios, sp_valid=sp_valid,
                sp_event_ids=sp_event_ids, sp_channel_ids=sp_channel_ids,
                cauchy_c=cauchy_c)


def _load_das_src(need_das, compute_uncertainty, RAY_PARAMS_DIR, forward_method, logger):
    """Load DAS ray params from ``cache/ray_params/table_das_{method}.h5``.

    The Layer-2+3 ``RayParamTable`` carries an optional leading trial axis.
    Nominal tables are shape ``(n_ev, n_rx)``; MC tables are shape
    ``(n_trials, n_ev, n_rx)`` with trial 0 = unperturbed.

    When ``compute_uncertainty=False``, we only expose trial 0 as the
    nominal ray params. When True, all trials are surfaced.
    """
    das_mc_takeoff_deg = []
    das_mc_azimuth_deg = []
    das_mc_distance = []
    NMC_DAS = 0
    HAS_DAS_SRC = False
    if need_das:
        table_path = RAY_PARAMS_DIR / f"table_das_{forward_method}.h5"
        if table_path.exists():
            table = RayParamTable.from_hdf5(table_path)
            if table.is_perturbed:
                if compute_uncertainty:
                    for i in range(table.n_trials):
                        das_mc_takeoff_deg.append(table.takeoff[i])
                        das_mc_azimuth_deg.append(table.azimuth[i])
                        das_mc_distance.append(table.raypath_length[i])
                    NMC_DAS = table.n_trials
                else:
                    # Use trial 0 = unperturbed nominal
                    das_mc_takeoff_deg = [table.takeoff[0]]
                    das_mc_azimuth_deg = [table.azimuth[0]]
                    das_mc_distance = [table.raypath_length[0]]
                    NMC_DAS = 1
            else:
                das_mc_takeoff_deg = [table.takeoff]
                das_mc_azimuth_deg = [table.azimuth]
                das_mc_distance = [table.raypath_length]
                NMC_DAS = 1
            HAS_DAS_SRC = True
            label = f"MC x {NMC_DAS}" if NMC_DAS > 1 else "nominal"
            logger.info(
                f"  DAS ray params loaded ({label}): {das_mc_takeoff_deg[0].shape}"
            )
    return dict(HAS_DAS_SRC=HAS_DAS_SRC,
                das_mc_takeoff_deg=das_mc_takeoff_deg,
                das_mc_azimuth_deg=das_mc_azimuth_deg,
                das_mc_distance=das_mc_distance,
                NMC_DAS=NMC_DAS)


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: _load_data — pure I/O + numpy, NO TORCH
# ═══════════════════════════════════════════════════════════════════════════

def _load_data(
    project_dir="",
    event_catalog="",
    forward_method="",
    inversion_types=None,
    das_geo=None,
    das_polarity=None,
    das_sp_ratio=None,
    sta_geo=None,
    sta_polarity=None,
    sta_sp_ratio=None,
    result_dir=None,
    compute_uncertainty=False,
    vs_vp_ratio=1.7,
    sp_decay_k=None,
    sp_decay_c=None,
    weighted_polarity=False,
    das_pol_weight=0.5,
    sdr_grid_dang_deg=5.0,
    init_pol_error=0.05,
    init_amp_error_perc=0.005,
    joint_pol_size=60,
    sp_norm="L1",
    mode_label="serial",
):
    """Phase 1: Load all data + build SDR grid + cauchy_c.  NO TORCH IMPORT.

    Returns a numpy-only ``data`` dict that step3_invert.py passes to runners.
    The runners then call ``_build_torch_state(data, device)`` to add torch
    tensors per device.
    """
    root = Path(project_dir).resolve()
    CACHE_DIR = root / "cache"
    logger = Logger("step3_invert", log_dir=str(root / "logs"))

    CAT_FILE = resolve_path(event_catalog, root)
    if result_dir is not None:
        RESULT_ROOT = resolve_path(result_dir, root)
        if RESULT_ROOT.exists() and any(RESULT_ROOT.glob("inv_sol/*/*.h5")):
            logger.info(f"  WARNING: result_dir '{RESULT_ROOT}' already contains results, "
                        f"existing files will be overwritten.")
    else:
        result_prefix = "result_uncert" if compute_uncertainty else "result"
        RESULT_ROOT = find_result_dir(root, result_prefix)

    t0 = _time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info(f"  step3_invert — Focal mechanism inversion "
                f"({'MC uncertainty' if compute_uncertainty else 'single pass'}, "
                f"{mode_label})")
    logger.info()
    logger.info("=" * 60)

    inversion_types = parse_inversion_types(inversion_types)
    inv_set = set(inversion_types)
    RAY_PARAMS_DIR = CACHE_DIR / "ray_params"

    req = mode_requires(inversion_types)
    need_das_pol = req["das_pol"]
    need_das_sp  = req["das_sp"]
    need_sta_pol = req["sta_pol"]
    need_sta_sp  = req["sta_sp"]
    need_das = need_das_pol or need_das_sp
    need_sta = need_sta_pol or need_sta_sp

    logger.info(f"[1/5] Configuration")
    logger.info(f"  {'Forward':<12}: {forward_method}")
    logger.info(f"  {'Result dir':<12}: {RESULT_ROOT}")
    logger.info(f"  {'Inv types':<12}: {inversion_types}")
    logger.info(f"  need_das_pol={need_das_pol}  need_das_sp={need_das_sp}  "
                f"need_sta_pol={need_sta_pol}  need_sta_sp={need_sta_sp}")

    # ── [2/5] Load DAS observations ──
    logger.info(f"[2/5] Load data")
    catalog = pd.read_csv(CAT_FILE)
    all_event_ids = [str(eid) for eid in catalog["event_id"].values]

    _pol = _load_das_polarity(need_das_pol, das_polarity, root, logger)
    _sp = _load_das_sp(need_das_sp, das_sp_ratio, root, sp_norm, logger)

    HAS_DAS_POL = _pol["HAS_DAS_POL"]
    Pkic = _pol["Pkic"]
    pol_valid = _pol["pol_valid"]
    pol_event_ids = _pol["pol_event_ids"]
    pol_channel_ids = _pol["pol_channel_ids"]
    HAS_DAS_SP = _sp["HAS_DAS_SP"]
    sp_ratios = _sp["sp_ratios"]
    sp_event_ids = _sp["sp_event_ids"]
    sp_channel_ids = _sp["sp_channel_ids"]
    cauchy_c = _sp["cauchy_c"]

    # ── Determine n_ch, n_ev, das_event_ids ──
    n_ch = n_ev = 0
    if HAS_DAS_POL:
        n_ch, n_ev = Pkic.shape
    elif HAS_DAS_SP:
        n_ch, n_ev = sp_ratios.shape
    if n_ev == 0:
        n_ev = len(all_event_ids)

    if pol_event_ids is not None:
        das_event_ids = [str(e) for e in pol_event_ids]
    elif sp_event_ids is not None:
        das_event_ids = [str(e) for e in sp_event_ids]
    else:
        das_event_ids = all_event_ids[:n_ev]

    # ── DAS data alignment check + event index mapping ──
    das_src_event_indices = None
    if need_das and (HAS_DAS_POL or HAS_DAS_SP):
        errors = []
        if pol_event_ids is not None and sp_event_ids is not None:
            if pol_event_ids != sp_event_ids:
                errors.append(
                    f"event_ids MISMATCH: polarity ({len(pol_event_ids)}) and "
                    f"sp_ratios ({len(sp_event_ids)}) have different event lists.")
        if pol_channel_ids is not None and sp_channel_ids is not None:
            if not np.array_equal(pol_channel_ids, sp_channel_ids):
                errors.append(
                    f"channel_ids MISMATCH: polarity and sp_ratios shapes differ.")
        if das_geo:
            das_geo_df = pd.read_csv(resolve_path(das_geo, root))
            expected_ch = das_geo_df["index"].values.astype(np.int32)
            check_ch = pol_channel_ids if pol_channel_ids is not None else sp_channel_ids
            if check_ch is not None and not np.array_equal(check_ch, expected_ch):
                errors.append(
                    f"channel_ids MISMATCH vs das_geo.")
        if errors:
            for e in errors:
                logger.info(f"  ERROR: {e}")
            raise RuntimeError("DAS data alignment check failed.")
        src_eid_to_row = {str(eid): i for i, eid in enumerate(all_event_ids)}
        das_src_event_indices = np.array(
            [src_eid_to_row[eid] for eid in das_event_ids if eid in src_eid_to_row])
        logger.info(f"  DAS alignment OK: {len(das_event_ids)} events x {n_ch} channels")

    # ── DAS source parameters (single-pass or MC) ──
    _src = _load_das_src(need_das, compute_uncertainty, RAY_PARAMS_DIR,
                          forward_method, logger)
    HAS_DAS_SRC = _src["HAS_DAS_SRC"]
    das_mc_takeoff_deg = _src["das_mc_takeoff_deg"]
    das_mc_azimuth_deg = _src["das_mc_azimuth_deg"]
    das_mc_distance = _src["das_mc_distance"]
    NMC_DAS = _src["NMC_DAS"]

    # Pre-slice + convert to degrees
    if HAS_DAS_SRC and n_ch > 0:
        ev_idx = das_src_event_indices if das_src_event_indices is not None else np.arange(n_ev)
        ch_idx = np.arange(n_ch)
        needs_slice = (das_mc_takeoff_deg[0].shape[0] != n_ev
                       or das_mc_takeoff_deg[0].shape[1] != n_ch)
        if needs_slice:
            for imc in range(NMC_DAS):
                das_mc_takeoff_deg[imc] = das_mc_takeoff_deg[imc][np.ix_(ev_idx, ch_idx)]
                das_mc_azimuth_deg[imc] = das_mc_azimuth_deg[imc][np.ix_(ev_idx, ch_idx)]
                das_mc_distance[imc] = das_mc_distance[imc][np.ix_(ev_idx, ch_idx)]
        das_mc_takeoff_deg = [np.degrees(t).astype(np.float32) for t in das_mc_takeoff_deg]
        das_mc_azimuth_deg = [np.degrees(a).astype(np.float32) for a in das_mc_azimuth_deg]

    if need_das and not HAS_DAS_SRC:
        raise RuntimeError(
            f"Missing DAS ray params file (run step1_das_* first): "
            f"{RAY_PARAMS_DIR / f'table_das_{forward_method}.h5'}"
        )

    # ── Station ray params + STA polarity / sp data ──
    sta_ray_params = None      # nominal-shape RayParamTable (or None)
    sta_table = None           # full table (may be MC-shaped)
    if need_sta:
        sta_src_path = RAY_PARAMS_DIR / f"table_sta_{forward_method}.h5"
        if sta_src_path.exists():
            sta_table = RayParamTable.from_hdf5(sta_src_path)
            sta_ray_params = (
                sta_table.trial(0) if sta_table.is_perturbed else sta_table
            )
            logger.info(f"  Station ray params loaded: {sta_src_path.name}")

    sta_pol_data = None
    if need_sta_pol:
        if not sta_polarity:
            raise SystemExit(
                "  ERROR — Inversion mode requires station polarity but "
                "`sta_polarity` was not provided."
            )
        sta_csv = resolve_path(sta_polarity, root)
        if not sta_csv.exists():
            raise SystemExit(f"  ERROR — Station polarity file not found: {sta_csv}")
        sta_geo_path = resolve_path(sta_geo, root) if sta_geo else None
        sta_pol_data = load_sta_polarity(
            sta_csv, sta_geo_path, das_event_ids, all_event_ids,
            sta_ray_params, logger)

    sta_sp_data = None
    if need_sta_sp:
        if not sta_sp_ratio:
            raise SystemExit(
                "  ERROR — Inversion mode requires station S/P ratios but "
                "`sta_sp_ratio` was not provided."
            )
        sta_sp_csv = resolve_path(sta_sp_ratio, root)
        if not sta_sp_csv.exists():
            raise SystemExit(f"  ERROR — Station S/P ratio file not found: {sta_sp_csv}")
        sta_sp_data = load_sta_sp_ratio(
            sta_sp_csv, das_event_ids, all_event_ids, sta_ray_params, logger)

    # ── STA MC source parameters ──
    HAS_STA_MC = False
    sta_mc_takeoff = []
    sta_mc_azimuth = []
    sta_name_to_col = {}
    NMC_STA = 0
    if compute_uncertainty and need_sta and sta_table is not None:
        if sta_table.is_perturbed and sta_table.station is not None:
            sta_name_to_col = {
                str(name): i for i, name in enumerate(sta_table.station)
            }
            NMC_STA = sta_table.n_trials
            for i in range(NMC_STA):
                sta_mc_takeoff.append(sta_table.takeoff[i])
                sta_mc_azimuth.append(sta_table.azimuth[i])
            HAS_STA_MC = True
            logger.info(f"  STA MC trials: {NMC_STA}")

    if not compute_uncertainty:
        NMC_STA = 1

    NMC = max(NMC_DAS, NMC_STA)
    if compute_uncertainty and NMC == 0:
        raise RuntimeError("No MC perturbation files found. Run step 1 with compute_uncertainty=True.")
    NMC = max(NMC, 1)

    # ── [4/5] Build SDR grid (numpy only) ──
    logger.info(f"[4/5] Build SDR grid")
    stk_g, dip_g, rak_g = make_sdr_grid_skhash_style(dang=sdr_grid_dang_deg)
    num_Mt = stk_g.shape[0]
    M_g = sdr_to_mt(stk_g, dip_g, rak_g)
    logger.info(f"  Grid size: {num_Mt} focal mechanisms")

    # Mean alignment is disabled only when a fixed constant offset (sp_decay_c) is provided.
    do_mean_alignment = (sp_decay_c is None)

    # ── Assemble the data dict ──
    return dict(
        # Bookkeeping
        root=root, t0=t0, RESULT_ROOT=RESULT_ROOT,
        inv_set=inv_set, n_ev=n_ev, n_ch=n_ch,
        das_event_ids=das_event_ids, all_event_ids=all_event_ids,
        # DAS observations
        HAS_DAS_POL=HAS_DAS_POL, HAS_DAS_SP=HAS_DAS_SP, HAS_DAS_SRC=HAS_DAS_SRC,
        Pkic=Pkic, pol_valid=pol_valid, sp_ratios=sp_ratios,
        das_mc_takeoff_deg=das_mc_takeoff_deg,
        das_mc_azimuth_deg=das_mc_azimuth_deg,
        das_mc_distance=das_mc_distance,
        NMC_DAS=NMC_DAS, NMC=NMC,
        # STA observations
        sta_pol_data=sta_pol_data, sta_sp_data=sta_sp_data,
        HAS_STA_MC=HAS_STA_MC,
        sta_mc_takeoff=sta_mc_takeoff, sta_mc_azimuth=sta_mc_azimuth,
        sta_name_to_col=sta_name_to_col, NMC_STA=NMC_STA,
        # SDR grid (numpy)
        stk_g=stk_g, dip_g=dip_g, rak_g=rak_g, M_g=M_g, num_Mt=num_Mt,
        # Algorithm params
        sp_decay_k=sp_decay_k, sp_decay_c=sp_decay_c,
        weighted_polarity=weighted_polarity, das_pol_weight=das_pol_weight,
        init_pol_error=init_pol_error, init_amp_error_perc=init_amp_error_perc,
        joint_pol_size=joint_pol_size,
        sp_norm=sp_norm, cauchy_c=cauchy_c,
        do_mean_alignment=do_mean_alignment,
        vp_vs_ratio=vs_vp_ratio,
        compute_uncertainty=compute_uncertainty,
        # Logger reference (only set in main process — workers will use stdout-only)
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2: _build_torch_state — build per-device torch tensors
# ═══════════════════════════════════════════════════════════════════════════

def _build_torch_state(data: dict, device: str) -> dict:
    """Phase 2: build per-device torch tensors and return the full ctx dict.

    Imports torch INSIDE the function — fork-safe.

    The returned ctx dict is what pipeline.process_event consumes.  It is the
    union of `data` (numpy + parameters) plus the new torch tensors:
        stk_t, dip_t, rak_t, M_g_t   (all on `device`)
    """
    import torch
    from dasfm.inversion.tensor_utils import numpy_to_torch

    ctx = dict(data)
    ctx["device"] = device
    ctx["stk_t"] = numpy_to_torch(data["stk_g"], device=device)
    ctx["dip_t"] = numpy_to_torch(data["dip_g"], device=device)
    ctx["rak_t"] = numpy_to_torch(data["rak_g"], device=device)
    ctx["M_g_t"] = numpy_to_torch(data["M_g"], device=device)
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point — called by step3_invert.py for the serial path
# ═══════════════════════════════════════════════════════════════════════════

def run(data, device="cpu", event_indices=None, _quiet=False):
    """Serial run.  Called by step3_invert.run() with pre-loaded data.

    Parameters
    ----------
    data : dict
        Output of _load_data() — pure numpy.
    device : str
        "cpu" or "cuda:N".
    event_indices : list[int] or None
        Subset of events to process.  None = all events.
    _quiet : bool
        If True, suppress screen output and tqdm.
    """
    # Lazy import — pipeline pulls torch (via forward / representative).
    from dasfm.inversion.pipeline import process_events

    ctx = _build_torch_state(data, device)
    ctx["_quiet"] = _quiet

    logger = ctx.get("logger")
    if logger is not None and not _quiet:
        logger.info("=" * 60)
        logger.info(f"  [serial] device={device}  events={len(event_indices) if event_indices else ctx['n_ev']}")
        logger.info("=" * 60)

    process_events(ctx, event_indices=event_indices)

    if logger is not None and not _quiet:
        logger.info("=" * 60)
        logger.info(f"  Done  ({_time.time() - ctx['t0']:.1f} s)")
        logger.info(f"  -> {ctx['RESULT_ROOT']}")
        logger.info("=" * 60)
        logger.close()
