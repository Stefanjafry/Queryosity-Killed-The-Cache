"""
Deduplicated, optionally parallel exact-simulation selection.

Shared by the sweep consumers, the paper-grid runner, and the schedule
exporter so all three select identically.

Why explicit dedup, given the objective already memoizes
--------------------------------------------------------
``ExactSimObjective`` memoizes on the schedule tuple, so within ONE process a
repeated schedule already costs a dict lookup (~0.008 ms) rather than a full
simulation (~6 s). Explicit dedup therefore does not speed up serial runs.
It matters for two other reasons:

1. **Parallelism requires it.** The memo lives on the objective instance, so
   forked workers cannot share it. Without dedup, two workers handed the same
   schedule would each simulate it in full.
2. **Honest accounting.** The candidate count and the simulation count are
   different numbers (e.g. 117 beam candidates but 66 distinct schedules on
   TPC-H at 262,144 pages). Reporting distinct simulations is the accurate
   measure of selection cost.

Determinism
-----------
Candidates are kept in construction order; duplicates map back to their first
occurrence. Downstream reduction uses strict ``>``, so the selected schedule
is identical for any worker count and for serial runs. ``pool.map`` preserves
input order, so no completion-order effect can leak in.
"""

from __future__ import annotations

import multiprocessing as mp

from src.bayesopt.objective import ExactSimObjective

Sched = tuple[int, ...]

_OBJ: ExactSimObjective | None = None


def _init_worker(page_sets: list[frozenset[int]], cache_pages: int,
                 d_matrix: list[list[int]] | None) -> None:
    global _OBJ
    _OBJ = ExactSimObjective(page_sets, cache_pages, d_matrix=d_matrix)


def _score_one(sched: Sched) -> float:
    assert _OBJ is not None
    metrics, _ = _OBJ.evaluate_schedule(list(sched))
    return metrics.hit_ratio


def dedup_index(cands: list[Sched]) -> tuple[list[Sched], list[int]]:
    """
    Order-preserving dedup.

    Returns ``(distinct, index)`` where ``index[i]`` is the position in
    ``distinct`` of candidate ``i``, so per-candidate scores can be
    reconstructed exactly as a serial run would produce them.
    """
    pos: dict[Sched, int] = {}
    distinct: list[Sched] = []
    index: list[int] = []
    for c in cands:
        p = pos.get(c)
        if p is None:
            p = len(distinct)
            pos[c] = p
            distinct.append(c)
        index.append(p)
    return distinct, index


def score_pool(
    cands: list[Sched],
    page_sets: list[frozenset[int]],
    cache_pages: int,
    d_matrix: list[list[int]] | None = None,
    workers: int = 1,
) -> tuple[list[float], int]:
    """
    Exact-simulate ``cands``, deduplicating first.

    Returns ``(scores, n_distinct)`` with one score per input candidate, in
    input order. ``workers=1`` runs in-process (no fork overhead).
    """
    if not cands:
        return [], 0
    distinct, index = dedup_index(cands)

    if workers <= 1:
        obj = ExactSimObjective(page_sets, cache_pages, d_matrix=d_matrix)
        vals = [obj.evaluate_schedule(list(s))[0].hit_ratio for s in distinct]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(page_sets, cache_pages, d_matrix)) as pool:
            vals = pool.map(_score_one, distinct, chunksize=1)

    return [vals[i] for i in index], len(distinct)


def resolve_workers(requested: int) -> int:
    """``0`` or negative means 'use every available core'; clamp to CPU count."""
    avail = mp.cpu_count()
    if requested <= 0:
        return avail
    return min(requested, avail)
