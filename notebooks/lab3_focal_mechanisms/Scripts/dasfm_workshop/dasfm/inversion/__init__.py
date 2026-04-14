"""
dasfm.inversion — Focal mechanism grid-search inversion for DAS.

Modules:
    moment_tensor   SDR ↔ vectors, DCM, Kagan angle (numpy)
    kagan           Kagan angle & weighted average (torch)
    cluster         Hierarchical clustering (no overlap)
    grid            SKHASH-style SDR grid & sdr_to_mt
    forward_model   Strain-rate radiation pattern (torch)
    pipeline        Per-event inversion orchestration
    quality         STDR, plane uncertainty
    utils           Tensor/array conversion helpers

Fork-safety contract
--------------------
``import dasfm.inversion`` MUST NOT pull torch into ``sys.modules``.  This is
required by the multi-CPU fork-pool runner (step3_invert_cpus.py): the parent
process loads numpy data, then forks workers; if torch were imported in the
parent, every worker would inherit a poisoned torch state (CUDA contexts can
deadlock across fork, OMP thread pools double-spawn, etc).

To preserve the convenient ``from dasfm.inversion import compute_das_forward``
API while keeping the module top-level torch-free, the torch-touching exports
use PEP 562 lazy ``__getattr__`` — they import on first attribute access, not
at module load.
"""

# torch-free exports — safe to import unconditionally
from .grid import make_sdr_grid_skhash_style, sdr_to_mt

# PEP 562 lazy lookup table — name → submodule path.  Each entry defers torch
# import until the attribute is actually used.
_LAZY_TORCH_EXPORTS = {
    # forward_model
    "strain_rate_pattern_das_torch":      ".forward_model",
    "strain_rate_pattern_allsta_torch":   ".forward_model",
    "sp_misfit_norm":                     ".forward_model",
    "sp_misfit_norm_np":                  ".forward_model",
    "compute_das_forward":                ".forward_model",
    "compute_das_polarity_misfit_torch":  ".forward_model",
    # quality
    "cal_stdr":                           ".quality",
    "compute_plane_uncertainty":          ".quality",
    # tensor_utils
    "numpy_to_torch":                     ".tensor_utils",
    "dict_to_torch":                      ".tensor_utils",
    "dict_to_numpy":                      ".tensor_utils",
    # kagan
    "dcm2kagan":                          ".kagan",
    "sdr2kagan_pdist":                    ".kagan",
}


def __getattr__(name):
    """PEP 562 module-level lazy attribute lookup (Python 3.7+).

    Triggers torch import only when a torch-touching name is actually
    accessed (e.g. ``dasfm.inversion.compute_das_forward``).
    """
    if name in _LAZY_TORCH_EXPORTS:
        from importlib import import_module
        mod = import_module(_LAZY_TORCH_EXPORTS[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Make lazy attributes visible to ``dir()`` / IDE autocomplete."""
    return sorted(list(globals().keys()) + list(_LAZY_TORCH_EXPORTS.keys()))


__all__ = [
    "make_sdr_grid_skhash_style",
    "sdr_to_mt",
    *_LAZY_TORCH_EXPORTS.keys(),
]
