"""dasfm.steps — Callable pipeline steps for focal mechanism inversion.

Each module exposes a ``run()`` function whose first argument is ``project_dir``
(all output paths are resolved relative to it).

Usage example::

    from dasfm.steps import step1_das_3d, step3_invert
    step1_das_3d(
        project_dir="my_project",
        event_catalog="input/catalog.csv",
        das_geo="input/das_geo/das_info.csv",
        vel_1d="input/velocity/vel1d.csv",
        topo="input/topo/topo.npy",
    )
    step3_invert(project_dir="my_project", ...)

Fork-safety contract
--------------------
``import dasfm.steps`` MUST NOT pull torch into ``sys.modules``.  This is
required by ``step3_invert_cpus`` (multi-CPU fork pool): the parent process
loads numpy data via ``_load_data()`` (torch-free), then forks workers.  If
torch were already in the parent, every worker would inherit a poisoned
torch state.

``step3_invert.py`` itself is torch-free (audited at Phase 6.5), so it is
imported eagerly to preserve the legacy ``from dasfm.steps import step3_invert``
calling convention.  Every other step (``step1_*``, ``step2*``, ``step4_*``)
pulls torch and is therefore exposed via PEP 562 lazy ``__getattr__`` — the
underlying submodule is imported only when the alias is actually accessed.
"""

# Eager import — step3_invert.py is torch-free, so this stays fork-safe.
# We rebind the name `step3_invert` from the submodule to its `run` function
# so users can do `from dasfm.steps import step3_invert; step3_invert(...)`.
from dasfm.steps.step3_invert import run as step3_invert


# Lazy aliases — every other step pulls torch (DAS / STA forward modeling,
# polarity workflow, plotting), so deferred until first attribute access.
_LAZY_STEP_FUNCTIONS = {
    "step1_sta_1d":        (".step1_sta_1d",        "run"),
    "step1_sta_2d":        (".step1_sta_2d",        "run"),
    "step1_sta_3d":        (".step1_sta_3d",        "run"),
    "step1_das_1d":        (".step1_das_1d",        "run"),
    "step1_das_2d":        (".step1_das_2d",        "run"),
    "step1_das_3d":        (".step1_das_3d",        "run"),
    "step2a_window":       (".step2a_window",       "run"),
    "step2b_polarity":     (".step2b_polarity",     "run"),
    "step2c_spratio":      (".step2c_spratio",      "run"),
    "step3_plot":          (".step3_plot",          "run"),
    "step4_summarize":     (".step4_summarize",     "run"),
    "step4_compare_runs":  (".step4_compare_runs",  "run"),
    "step4_compare_modes": (".step4_compare_modes", "run"),
}


def __getattr__(name):
    """PEP 562 module-level lazy attribute lookup (Python 3.7+).

    Defers submodule import (and its torch dependency) until the alias is
    actually used.  Crucial for fork-safety in step3_invert_cpus.

    The resolved function is **cached in the module globals** so that the
    legacy ``from dasfm.steps import step1_das_3d`` calling convention
    returns the function.  Without the cache, Python's import machinery
    would do a fallback submodule import after ``__getattr__`` returns and
    overwrite the binding with the submodule object — making
    ``step1_das_3d(...)`` fail because modules aren't callable.
    """
    if name in _LAZY_STEP_FUNCTIONS:
        from importlib import import_module
        modpath, attr = _LAZY_STEP_FUNCTIONS[name]
        mod = import_module(modpath, package=__name__)
        fn = getattr(mod, attr)
        globals()[name] = fn          # ← cache so the submodule fallback finds it
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Make lazy aliases visible to ``dir()`` / IDE autocomplete."""
    return sorted(list(globals().keys()) + list(_LAZY_STEP_FUNCTIONS.keys()))


__all__ = [
    "step3_invert",
    *_LAZY_STEP_FUNCTIONS.keys(),
]
