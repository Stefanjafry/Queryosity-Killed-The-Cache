"""
Deterministic consumers that turn a scorer into candidate schedules.

Two consumers, both fully deterministic given a scorer config and
features:

* **Multistart greedy** — builds one schedule per start query in a
  deterministic, de-duplicated start set, each grown by repeatedly
  appending ``argmax_{j∈U} score(i → j)`` (ties to lower index).  The
  start set always includes the Greedy-D seed (largest pagecount), so
  for a pure-D config the exact Greedy-D schedule is guaranteed to be
  among the candidates.
* **Beam search** — keeps the top-``width`` partial schedules ranked by
  cumulative scorer value (ties broken lexicographically by the partial
  order, then by next-query index), expanding all continuations each
  step.  No prefix exact-simulation, no local search.

Neither consumer scores schedules — they only *construct* candidates.
The exact simulator (in the runner) judges them.  Each candidate is
returned with provenance so logs can attribute it to a start query or
a beam rank.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bayesopt.mode_a.features import FeatureTable
from src.bayesopt.mode_a.scorers import ScorerConfig, score_edge

CONSUMERS = ("multistart_greedy", "beam")
"""Implemented consumers."""

DEFAULT_NUM_STARTS = 5
DEFAULT_BEAM_WIDTHS = (2, 4, 8)


@dataclass(frozen=True)
class Candidate:
    """
    One constructed schedule with provenance.

    Attributes
    ----------
    schedule : tuple[int, ...]
        Permutation of ``range(n)`` (query indices).
    source : str
        Human-readable provenance, e.g. ``"start=q7"`` or
        ``"beam_rank=0"``.
    """

    schedule: tuple[int, ...]
    source: str


def _greedy_from_start(
    cfg: ScorerConfig, feats: FeatureTable, start: int
) -> tuple[int, ...]:
    """Grow a full schedule greedily from *start* under *cfg*."""
    n = feats.n
    visited = [False] * n
    visited[start] = True
    schedule = [start]
    current = start
    for _ in range(n - 1):
        best_j = -1
        best_score = float("-inf")
        for j in range(n):
            if visited[j]:
                continue
            s = score_edge(cfg, feats, current, j)
            if s > best_score:  # strict ⇒ ties keep the lower index
                best_score = s
                best_j = j
        schedule.append(best_j)
        visited[best_j] = True
        current = best_j
    return tuple(schedule)


def _start_set(
    cfg: ScorerConfig, feats: FeatureTable, num_starts: int
) -> list[int]:
    """
    Deterministic, de-duplicated start queries.

    Always includes the Greedy-D seed (largest pagecount, lowest index
    on ties) so pure-D multistart contains the exact Greedy-D schedule.
    Remaining starts are drawn from family-appropriate node features.
    """
    n = feats.n
    ranked: list[int] = []

    def add(idx: int) -> None:
        if idx not in ranked:
            ranked.append(idx)

    # 1. Greedy-D seed: largest pagecount (the original seed rule).
    add(max(range(n), key=lambda i: (feats.pagecount[i], -i)))
    # 2. Highest out-potential for the family.
    out = feats.out_D_topk[cfg.topk] if cfg.family in ("d", "hybrid") \
        else feats.out_M_topk[cfg.topk]
    add(max(range(n), key=lambda i: (out[i], -i)))
    # 3. Highest balance_D (D / hybrid only) else highest in_D.
    if cfg.family in ("d", "hybrid"):
        add(max(range(n), key=lambda i: (feats.balance_D[i], -i)))
    else:
        add(max(range(n), key=lambda i: (feats.in_D[i], -i)))
    # 4. Smallest pagecount (a contrasting seed).
    add(min(range(n), key=lambda i: (feats.pagecount[i], i)))
    # 5+. Fill deterministically by descending pagecount.
    for i in sorted(range(n), key=lambda i: (-feats.pagecount[i], i)):
        if len(ranked) >= num_starts:
            break
        add(i)

    return ranked[:num_starts]


def multistart_greedy(
    cfg: ScorerConfig,
    feats: FeatureTable,
    num_starts: int = DEFAULT_NUM_STARTS,
) -> list[Candidate]:
    """
    Build one greedy schedule per deterministic start query.

    Parameters
    ----------
    cfg : ScorerConfig
        Scoring policy.
    feats : FeatureTable
        Precomputed features.
    num_starts : int
        Number of (de-duplicated) starts; capped at ``n``.

    Returns
    -------
    list[Candidate]
        One candidate per distinct start, in start-set order.
    """
    starts = _start_set(cfg, feats, min(num_starts, feats.n))
    return [
        Candidate(_greedy_from_start(cfg, feats, s), source=f"start={s}")
        for s in starts
    ]


def beam_search(
    cfg: ScorerConfig,
    feats: FeatureTable,
    beam_width: int,
) -> list[Candidate]:
    """
    Deterministic beam search ranked by cumulative scorer value.

    Each partial is ``(cumulative_score, schedule_tuple)``; ties break
    lexicographically on the schedule so ordering is reproducible.  The
    first query is seeded by the same Greedy-D rule (largest pagecount)
    to fix the otherwise-arbitrary root; beam then explores from there.

    Parameters
    ----------
    cfg : ScorerConfig
        Scoring policy.
    feats : FeatureTable
        Precomputed features.
    beam_width : int
        Number of partials retained per step (≥ 1).

    Returns
    -------
    list[Candidate]
        Completed schedules, best cumulative score first (up to
        *beam_width* of them), tagged by final beam rank.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    n = feats.n
    root = max(range(n), key=lambda i: (feats.pagecount[i], -i))
    # Each beam entry: (cum_score, schedule_tuple, visited_frozenset)
    beam: list[tuple[float, tuple[int, ...]]] = [(0.0, (root,))]

    for _ in range(n - 1):
        expansions: list[tuple[float, tuple[int, ...]]] = []
        for cum, sched in beam:
            visited = set(sched)
            current = sched[-1]
            for j in range(n):
                if j in visited:
                    continue
                s = cum + score_edge(cfg, feats, current, j)
                expansions.append((s, sched + (j,)))
        # Rank by descending cumulative score, then ascending schedule
        # tuple for deterministic tie-breaking.
        expansions.sort(key=lambda e: (-e[0], e[1]))
        beam = expansions[:beam_width]

    return [
        Candidate(sched, source=f"beam_rank={rank}")
        for rank, (_, sched) in enumerate(beam)
    ]


def build_candidates(
    cfg: ScorerConfig,
    feats: FeatureTable,
    consumer: str,
    num_starts: int = DEFAULT_NUM_STARTS,
    beam_width: int = 4,
) -> list[Candidate]:
    """
    Dispatch to the requested consumer and return its candidate set.

    Parameters
    ----------
    cfg : ScorerConfig
        Scoring policy.
    feats : FeatureTable
        Precomputed features.
    consumer : str
        One of :data:`CONSUMERS`.
    num_starts : int
        Multistart-greedy start count.
    beam_width : int
        Beam width (beam consumer only).

    Returns
    -------
    list[Candidate]
        Candidate schedules with provenance.
    """
    if consumer == "multistart_greedy":
        return multistart_greedy(cfg, feats, num_starts)
    if consumer == "beam":
        return beam_search(cfg, feats, beam_width)
    raise ValueError(f"unknown consumer {consumer!r}")


__all__ = [
    "CONSUMERS",
    "DEFAULT_NUM_STARTS",
    "DEFAULT_BEAM_WIDTHS",
    "Candidate",
    "multistart_greedy",
    "beam_search",
    "build_candidates",
]
