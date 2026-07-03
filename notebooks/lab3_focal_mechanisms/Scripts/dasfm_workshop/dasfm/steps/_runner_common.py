"""Shared helpers for the step3_invert parallel runners.

Fork-safety: NO torch import here.  This module is loaded by step3_invert
(via the gpus runner) before the multi-CPU pool fork — it must stay torch-free
to keep the parent process clean.
"""
from __future__ import annotations


def partition_events_round_robin(event_indices: list[int],
                                  num_workers: int) -> list[list[int]]:
    """Round-robin partition of ``event_indices`` across ``num_workers`` lists.

    Accepts an arbitrary list (not just ``range(n_ev)``), so user-supplied
    event subsets are partitioned naturally — e.g. ``event_indices=[5, 10, 15]``
    on 2 GPUs gives ``[[5, 15], [10]]``.  Used by the gpus runner; the cpus
    runner uses ``imap_unordered`` and doesn't need partitioning.
    """
    return [list(event_indices[i::num_workers]) for i in range(num_workers)]
