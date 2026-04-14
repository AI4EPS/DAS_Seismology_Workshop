"""modes.py — Declarative spec for the 7 inversion modes.

Replaces the 14 if-blocks in legacy solve_event.py + repr_misfit.py with a
single declarative table that drives both the filter stage (per-mode pol/sp
input assembly) and the representative stage (fm_mean misfit on trial-0 ray).

The 7 modes:
    STA_pol             — STA polarity only
    STA_pol_sp          — STA polarity + STA S/P
    DAS_pol             — DAS polarity only
    DAS_pol_sp          — DAS polarity + DAS S/P
    STA_pol_DAS_pol     — joint STA+DAS polarity, no S/P
    STA_pol_DAS_sp      — STA polarity + DAS S/P (no DAS pol)
    STA_pol_DAS_pol_sp  — joint STA+DAS polarity + DAS S/P

Strict "missing data → skip" semantics: if a mode requires both DAS and STA
polarity (joint pol modes), and STA is unavailable for an event (n_sta == 0),
the mode is SKIPPED for that event — NOT silently degraded to DAS-only.

This module is **torch-free** — safe to import in fork-safety chain.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModeSpec:
    """Declarative spec for one inversion mode.

    Fields
    ------
    name : str
        Mode name (e.g. "STA_pol_DAS_pol_sp"), used as result file stem.
    pol_sources : tuple[str, ...]
        Subset of ("sta", "das"), indicating which sources contribute to
        the polarity misfit.
    sp_source : "sta" / "das" / None
        Which source provides the S/P amplitude misfit, or None if the mode
        is polarity-only.
    """
    name: str
    pol_sources: tuple[str, ...]
    sp_source: Optional[str]

    @property
    def is_joint(self) -> bool:
        return self.sp_source is not None

    @property
    def has_sta_pol(self) -> bool:
        return "sta" in self.pol_sources

    @property
    def has_das_pol(self) -> bool:
        return "das" in self.pol_sources


MODE_REGISTRY: dict[str, ModeSpec] = {
    "STA_pol":             ModeSpec("STA_pol",             ("sta",),       None),
    "STA_pol_sp":          ModeSpec("STA_pol_sp",          ("sta",),       "sta"),
    "DAS_pol":             ModeSpec("DAS_pol",             ("das",),       None),
    "DAS_pol_sp":          ModeSpec("DAS_pol_sp",          ("das",),       "das"),
    "STA_pol_DAS_pol":     ModeSpec("STA_pol_DAS_pol",     ("sta", "das"), None),
    "STA_pol_DAS_sp":      ModeSpec("STA_pol_DAS_sp",      ("sta",),       "das"),
    "STA_pol_DAS_pol_sp":  ModeSpec("STA_pol_DAS_pol_sp",  ("sta", "das"), "das"),
}


def mode_requires(modes: list[str]) -> dict[str, bool]:
    """Compute combined data requirements across a list of modes.

    Used by step3_invert.py for input file validation:
    if any selected mode needs DAS polarity, das_polarity= must be provided, etc.

    Returns 4 atomic flags. The 'das' / 'sta' aggregates are derived by the
    caller via simple OR (e.g. need_das = need_das_pol or need_das_sp).
    """
    specs = [MODE_REGISTRY[m] for m in modes]
    return {
        "das_pol": any(s.has_das_pol for s in specs),
        "das_sp":  any(s.sp_source == "das" for s in specs),
        "sta_pol": any(s.has_sta_pol for s in specs),
        "sta_sp":  any(s.sp_source == "sta" for s in specs),
    }
