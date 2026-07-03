"""step3_invert — Focal mechanism inversion entry point.

Top-level dispatcher that:

    1. Validates inputs (paths exist, required files present per mode).
    2. Loads ALL data ONCE in the main process via ``_load_data()`` — no torch.
    3. Resolves the device + dispatches to one of three runners:
        - serial : single CPU or single GPU
        - cpus   : multi-CPU fork pool      (workers inherit data via COW)
        - gpus   : multi-GPU thread pool    (threads share data via closure)

Fork-safety contract
--------------------
This module's **top-level imports** are intentionally torch-free.  The runners
are imported lazily inside the dispatch branches so importing this module
does not pull torch into ``sys.modules`` — required for the multi-CPU fork
pool to work without poisoning workers with parent-process torch state.

A.1 fork-safety smoke test::

    python -c "import sys; import dasfm.steps.step3_invert; \
        assert 'torch' not in sys.modules; print('OK')"
"""
from __future__ import annotations

from pathlib import Path

from dasfm.utils.step_utils import (
    resolve_path, parse_inversion_types, resolve_device,
)
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.das_io import validate_das_geo
from dasfm.io.sta_io import (
    validate_sta_geo, validate_sta_polarity, validate_sta_sp_ratio,
)
from dasfm.io.middle_io import validate_das_polarity, validate_das_sp_ratios
from dasfm.inversion.modes import mode_requires


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-flight validation (mode-driven)
# ═══════════════════════════════════════════════════════════════════════════

def _validate_inputs(kwargs: dict) -> list[str]:
    """Pre-flight: required-arg check + per-mode file existence check.

    Parses inversion_types via ``mode_requires`` (replaces the legacy scattered
    ``_need_*`` booleans).  Returns the canonical list of mode names.
    """
    required = {
        "project_dir":     kwargs.get("project_dir"),
        "event_catalog":   kwargs.get("event_catalog"),
        "forward_method":  kwargs.get("forward_method"),
        "inversion_types": kwargs.get("inversion_types"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Required parameters missing: {', '.join(missing)}")

    root = Path(kwargs["project_dir"]).resolve()
    modes = parse_inversion_types(kwargs["inversion_types"])
    if not modes:
        raise ValueError(
            f"No valid inversion modes parsed from: {kwargs['inversion_types']}")

    req = mode_requires(modes)
    need_das_pol = req["das_pol"]
    need_das_sp  = req["das_sp"]
    need_sta_pol = req["sta_pol"]
    need_sta_sp  = req["sta_sp"]
    need_das = need_das_pol or need_das_sp
    need_sta = need_sta_pol or need_sta_sp

    validate_event_catalog(resolve_path(kwargs["event_catalog"], root))

    if need_das:
        if not kwargs.get("das_geo"):
            raise SystemExit("step3_invert: das_geo is required for DAS modes.")
        validate_das_geo(resolve_path(kwargs["das_geo"], root))
    if need_sta:
        if not kwargs.get("sta_geo"):
            raise SystemExit("step3_invert: sta_geo is required for STA modes.")
        validate_sta_geo(resolve_path(kwargs["sta_geo"], root))

    if need_das_pol:
        if not kwargs.get("das_polarity"):
            raise SystemExit(
                "step3_invert: das_polarity is required for *_pol modes "
                "(run step2b first).")
        validate_das_polarity(resolve_path(kwargs["das_polarity"], root))
    if need_das_sp:
        if not kwargs.get("das_sp_ratio"):
            raise SystemExit(
                "step3_invert: das_sp_ratio is required for *_sp modes "
                "(run step2c first).")
        validate_das_sp_ratios(resolve_path(kwargs["das_sp_ratio"], root))
    if need_sta_pol:
        if not kwargs.get("sta_polarity"):
            raise SystemExit(
                "step3_invert: sta_polarity is required for STA *_pol modes.")
        validate_sta_polarity(resolve_path(kwargs["sta_polarity"], root))
    if need_sta_sp:
        if not kwargs.get("sta_sp_ratio"):
            raise SystemExit(
                "step3_invert: sta_sp_ratio is required for STA_pol_sp mode.")
        validate_sta_sp_ratio(resolve_path(kwargs["sta_sp_ratio"], root))

    # Step1 outputs (ray params from forward modeling)
    ray_dir = root / "cache" / "ray_params"
    method = kwargs["forward_method"]
    if need_das:
        _p = ray_dir / f"table_das_{method}.h5"
        if not _p.exists():
            raise FileNotFoundError(f"Ray parameters table not found: {_p}")
    if need_sta:
        _p = ray_dir / f"table_sta_{method}.h5"
        if not _p.exists():
            raise FileNotFoundError(f"Ray parameters table not found: {_p}")

    return modes


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point — auto-dispatch
# ═══════════════════════════════════════════════════════════════════════════

def run(
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
    device="cpu",
    num_cpu_workers=1,
    event_indices=None,
    _quiet=False,
):
    """Focal mechanism inversion — auto-dispatch.

    Dispatches to serial / cpus / gpus runner based on ``device`` and
    ``num_cpu_workers``.

    Supported inversion_types (use "+" format):
        "sta_pol", "sta_pol + sta_sp", "das_pol", "das_pol + das_sp",
        "sta_pol + das_pol", "sta_pol + das_sp", "sta_pol + das_pol + das_sp"

    DAS S/P bias correction (joint modes):
        sp_decay_k : float or None
            Linear distance decay coefficient (per km).  When set, applies
            obs += k * dist_km before inversion.  None = no correction.
        sp_decay_c : float or None
            Constant offset added to observed S/P.  When set, applies
            obs += c and disables the per-event mean alignment.  None =
            mean alignment is kept (and absorbs any residual constant bias).

        Combinations:
          (None, None)   default: per-event mean alignment auto-absorbs all bias
          (None, c)      fixed offset, no mean alignment
          (k,    None)   distance correction + mean alignment for residual bias
          (k,    c)      distance correction + fixed offset, no mean alignment

    Parallelism:
        device='cpu', num_cpu_workers=1     → serial CPU
        device='cuda:N', num_cpu_workers=*  → serial GPU on cuda:N
        device='cpu', num_cpu_workers>1     → multi-CPU fork pool
        device=['cuda:0','cuda:1',...]      → multi-GPU thread pool
    """
    # Pre-flight validation (paths + per-mode required files)
    kwargs = dict(
        project_dir=project_dir, event_catalog=event_catalog,
        forward_method=forward_method, inversion_types=inversion_types,
        das_geo=das_geo, das_polarity=das_polarity, das_sp_ratio=das_sp_ratio,
        sta_geo=sta_geo, sta_polarity=sta_polarity, sta_sp_ratio=sta_sp_ratio,
        result_dir=result_dir, compute_uncertainty=compute_uncertainty,
        vs_vp_ratio=vs_vp_ratio, sp_decay_k=sp_decay_k, sp_decay_c=sp_decay_c,
        weighted_polarity=weighted_polarity, das_pol_weight=das_pol_weight,
        sdr_grid_dang_deg=sdr_grid_dang_deg, init_pol_error=init_pol_error,
        init_amp_error_perc=init_amp_error_perc,
        joint_pol_size=joint_pol_size,
        sp_norm=sp_norm,
    )
    _validate_inputs(kwargs)

    # Resolve device → canonical form (str for single, list for multi-GPU)
    device_resolved, num_gpu, _ = resolve_device(device)

    # ── Load all data ONCE in the main process (no torch — fork-safe) ──
    from dasfm.steps.step3_invert_serial import _load_data
    if num_gpu > 1:
        mode_label = f"{num_gpu} GPUs"
    elif num_gpu == 0 and num_cpu_workers > 1:
        mode_label = f"{num_cpu_workers} CPUs"
    else:
        mode_label = f"single {device_resolved}"
    data = _load_data(mode_label=mode_label, **kwargs)

    # ── Dispatch ──
    if num_gpu > 1:
        from dasfm.steps.step3_invert_gpus import run as _run
        _run(data, devices=device_resolved, event_indices=event_indices)
    elif num_gpu == 0 and num_cpu_workers > 1:
        from dasfm.steps.step3_invert_cpus import run as _run
        _run(data, num_workers=num_cpu_workers, event_indices=event_indices)
    else:
        # Serial: single GPU or single CPU
        from dasfm.steps.step3_invert_serial import run as _run
        _run(data, device=device_resolved,
             event_indices=event_indices, _quiet=_quiet)
