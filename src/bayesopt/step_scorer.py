"""
Single-step edge scorer for greedy/beam consumers, paired M vs D.

The scorer is parameterised by the ``immediate`` matrix, so M, D, and a
Hybrid mix run through the *same* formula and the *same* weight space —
every D experiment has a fair M twin by construction::

    score(i -> j, U, C) = w_immediate * immediate[i][j]
                        + w_future    * future(j, U)
                        - w_regret    * regret(i, j, U)
                        - w_cache     * cache_feature(j, C)

``immediate`` is row-normalised so that ``w_immediate``-only reproduces
the existing greedy on that matrix exactly (argmax over a row is
unchanged by a per-row constant), which means Greedy-M and Greedy-D are
literally the ``M_only`` / ``D_only`` points of this scorer.

This is single-step edge-local on purpose: the windowed/triple-
intersection variant collapses in the overflow regime (documented
negative), so the look-ahead lives in the ``future``/``regret`` terms,
not in a cache-budget window.

``exact_state_greedy`` is the matrix-free reference both M and D
approximate: at each step it reads the *true* resident set from the
clock-sweep simulator and picks the candidate with the most resident
pages. D should track it more closely than M, since D models eviction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.simulator.cache_simulator import PageClockSweepCache

Matrix = list[list[float]]
ScoreFn = Callable[[int, int, Sequence[int]], float]


def row_normalize(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Row-normalise off-diagonal entries; preserves per-row argmax."""
    n = len(matrix)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = sum(matrix[i][j] for j in range(n) if j != i)
        if s > 0.0:
            for j in range(n):
                if i != j:
                    out[i][j] = matrix[i][j] / s
    return out


def hybrid_immediate(
    m_norm: Sequence[Sequence[float]],
    d_norm: Sequence[Sequence[float]],
    w_m: float,
    w_d: float,
) -> Matrix:
    """Tunable mix of the two normalised matrices (the Hybrid arm)."""
    n = len(m_norm)
    return [[w_m * m_norm[i][j] + w_d * d_norm[i][j] for j in range(n)]
            for i in range(n)]


@dataclass
class ScorerWeights:
    """Weights + knobs for one scorer formula (the BO search point)."""

    w_immediate: float = 1.0
    w_future: float = 0.0
    w_regret: float = 0.0
    w_cache: float = 0.0
    w_connector: float = 0.0
    topk: int = 5
    connector_rho: float = 0.25  # connector threshold = rho * per-query max
    cache_mode: str = "pressure"  # "pressure" = pagecount/C, "fit" = min(1, C/pc)


def future_value(
    j: int, unscheduled: Sequence[int],
    immediate: Sequence[Sequence[float]], topk: int,
) -> float:
    """Top-k mean of how much j feeds the remaining queries."""
    vals = sorted((immediate[j][k] for k in unscheduled if k != j), reverse=True)
    top = vals[:topk]
    return sum(top) / len(top) if top else 0.0


def regret(
    i: int, j: int, unscheduled: Sequence[int],
    immediate: Sequence[Sequence[float]],
) -> float:
    """Missed opportunity: best remaining predecessor for j beyond i, >= 0."""
    best = max((immediate[k][j] for k in unscheduled if k != j and k != i),
               default=0.0)
    return max(0.0, best - immediate[i][j])


def cache_feature(j: int, cache_pages: int, pagecounts: Sequence[int],
                  mode: str) -> float:
    """pagecount/C (pressure) or min(1, C/pagecount) (saturating fit)."""
    pc = pagecounts[j]
    if mode == "fit":
        return min(1.0, cache_pages / pc) if pc > 0 else 1.0
    return pc / cache_pages


def connector_out(j: int, unscheduled: Sequence[int],
                  immediate: Sequence[Sequence[float]], rho: float) -> float:
    """
    Degree (not weighted sum): fraction of remaining queries that j *feeds*
    within rho of j's strongest connection.  With predecessor-as-row
    (D[j][k] = j feeds k), this asks 'if I run j now, how many future
    queries can reuse what j leaves behind'.  Count, not sum, so it is
    structurally distinct from future/reuse (a weighted sum).

    The threshold is RELATIVE (rho * row-max), which makes the result
    invariant to row-normalisation — so the scorer (normalised immediate)
    and the screen (raw matrix) agree.
    """
    others = [k for k in unscheduled if k != j]
    if not others:
        return 0.0
    vals = [immediate[j][k] for k in others]
    hi = max(vals)
    if hi <= 0.0:
        return 0.0
    thr = rho * hi
    return sum(1 for v in vals if v >= thr) / len(others)


def connector_in(j: int, unscheduled: Sequence[int],
                 immediate: Sequence[Sequence[float]], rho: float) -> float:
    """
    Mirror of connector_out: fraction of remaining queries that would *feed*
    j (j as successor, column), within rho of j's strongest incoming edge.
    On RAW symmetric M this equals connector_out (plumbing sanity check);
    on D it diverges — the out-vs-in gap is a directional signal about
    whether j's value is in what it feeds vs consumes.  Screen-only unless
    it shows a pulse.
    """
    others = [k for k in unscheduled if k != j]
    if not others:
        return 0.0
    vals = [immediate[k][j] for k in others]
    hi = max(vals)
    if hi <= 0.0:
        return 0.0
    thr = rho * hi
    return sum(1 for v in vals if v >= thr) / len(others)


def net_reuse_gap(j: int, unscheduled: Sequence[int],
                  d_raw: Sequence[Sequence[float]],
                  m_raw: Sequence[Sequence[float]]) -> float:
    """
    DIAGNOSTIC, not a paired feature: reads both D and M, so it has no
    symmetric twin and must stay OUT of the paired BO grid.  Signed reuse
    minus survival gap = sum_k D[j][k] - (M[j][k] - D[j][k]).  Must use the
    RAW matrices: on row-normalised matrices both rows sum to 1 and this
    collapses to the constant 1.  Reported in the collinearity screen only;
    if it earns weight it is a cross-matrix hybrid feature, described as
    such — never part of the M-vs-D comparison.
    """
    total = 0.0
    for k in unscheduled:
        if k == j:
            continue
        total += d_raw[j][k] - (m_raw[j][k] - d_raw[j][k])
    return total


def make_step_scorer(
    immediate: Sequence[Sequence[float]],
    weights: ScorerWeights,
    pagecounts: Sequence[int],
    cache_pages: int,
) -> ScoreFn:
    """Build ``score(i, j, unscheduled)`` from a matrix and weights."""
    def score(i: int, j: int, unscheduled: Sequence[int]) -> float:
        s = weights.w_immediate * immediate[i][j]
        if weights.w_future:
            s += weights.w_future * future_value(j, unscheduled, immediate,
                                                 weights.topk)
        if weights.w_regret:
            s -= weights.w_regret * regret(i, j, unscheduled, immediate)
        if weights.w_cache:
            s -= weights.w_cache * cache_feature(j, cache_pages, pagecounts,
                                                 weights.cache_mode)
        if weights.w_connector:
            s += weights.w_connector * connector_out(j, unscheduled, immediate,
                                                     weights.connector_rho)
        return s
    return score


def greedy_schedule(
    score: ScoreFn, n: int, pagecounts: Sequence[int],
    start: int | None = None,
) -> list[int]:
    """Greedy-1: append argmax score(last, j, unscheduled). Ties -> lower index."""
    if n == 0:
        return []
    if start is None:
        start = max(range(n), key=lambda i: pagecounts[i])
    visited = [False] * n
    schedule = [start]
    visited[start] = True
    for _ in range(n - 1):
        i = schedule[-1]
        unscheduled = [k for k in range(n) if not visited[k]]
        best_j = -1
        best = float("-inf")
        for j in unscheduled:
            sc = score(i, j, unscheduled)
            if sc > best:
                best = sc
                best_j = j
        schedule.append(best_j)
        visited[best_j] = True
    return schedule


def exact_state_greedy(
    page_sets: Sequence[frozenset[int]], cache_pages: int,
    start: int | None = None,
) -> list[int]:
    """
    Matrix-free reference: at each step pick the candidate with the most
    pages already resident under the *real* clock-sweep state.  This is
    the 'perfect immediate' signal that M and D approximate.
    """
    n = len(page_sets)
    pagecounts = [len(ps) for ps in page_sets]
    if n == 0:
        return []
    if start is None:
        start = max(range(n), key=lambda i: pagecounts[i])
    cache = PageClockSweepCache(cache_pages)
    visited = [False] * n
    schedule = [start]
    visited[start] = True
    cache.batch_access(page_sets[start])
    for _ in range(n - 1):
        resident = cache.resident_pages()
        best_j = -1
        best = -1
        for j in range(n):
            if visited[j]:
                continue
            hits = len(resident & page_sets[j])
            if hits > best:
                best = hits
                best_j = j
        schedule.append(best_j)
        visited[best_j] = True
        cache.batch_access(page_sets[best_j])
    return schedule


def beam_search_schedule(
    score: ScoreFn, n: int, pagecounts: Sequence[int],
    beam_width: int, start: int | None = None,
) -> list[list[int]]:
    """
    Beam search over the step scorer: keep the top ``beam_width`` partial
    schedules by cumulative score, expand each by every unscheduled query,
    re-prune to the top ``beam_width``.  Returns the completed beams
    (the caller exact-sims all of them — that is the eval cost).

    beam_width == 1 reduces to Greedy-1.  Ties break deterministically on
    the schedule prefix so runs are reproducible under PYTHONHASHSEED=0.
    """
    if n == 0:
        return []
    if start is None:
        start = max(range(n), key=lambda i: pagecounts[i])
    # (cumulative_score, schedule, visited)
    beams: list[tuple[float, list[int], frozenset[int]]] = [
        (0.0, [start], frozenset({start}))]
    for _ in range(n - 1):
        cand: list[tuple[float, list[int], frozenset[int]]] = []
        for cum, sched, visited in beams:
            i = sched[-1]
            unscheduled = [k for k in range(n) if k not in visited]
            for j in unscheduled:
                cand.append((cum + score(i, j, unscheduled),
                             sched + [j], visited | {j}))
        cand.sort(key=lambda b: (-b[0], b[1]))
        beams = cand[:beam_width]
    return [sched for _, sched, _ in beams]


def good_start_set(
    immediate: Sequence[Sequence[float]], pagecounts: Sequence[int], k: int,
) -> list[int]:
    """
    Deterministic good-start set for multistart greedy (Meeting 5):
    largest pagecount, highest outgoing potential (row sum), highest incoming
    potential (col sum), and highest balance (out - in).  Deduplicated,
    truncated to k, in a stable order.
    """
    n = len(pagecounts)
    if n == 0:
        return []
    out_pot = [sum(immediate[i][j] for j in range(n) if j != i) for i in range(n)]
    in_pot = [sum(immediate[k2][i] for k2 in range(n) if k2 != i) for i in range(n)]
    cand = [
        max(range(n), key=lambda i: pagecounts[i]),
        max(range(n), key=lambda i: out_pot[i]),
        max(range(n), key=lambda i: in_pot[i]),
        max(range(n), key=lambda i: out_pot[i] - in_pot[i]),
    ]
    seen: list[int] = []
    for c in cand:
        if c not in seen:
            seen.append(c)
    # pad deterministically with largest-pagecount order if k > distinct cands
    if len(seen) < k:
        for i in sorted(range(n), key=lambda i: -pagecounts[i]):
            if i not in seen:
                seen.append(i)
            if len(seen) >= k:
                break
    return seen[:k]


def multistart_greedy_schedules(
    score: ScoreFn, n: int, pagecounts: Sequence[int], starts: Sequence[int],
) -> list[list[int]]:
    """
    Run Greedy-1 from each start in *starts*; return one schedule per start.
    The caller exact-sims all of them (that is the eval cost = len(starts)).
    """
    return [greedy_schedule(score, n, pagecounts, start=s) for s in starts]


__all__ = [
    "row_normalize", "hybrid_immediate", "ScorerWeights", "future_value",
    "regret", "cache_feature", "connector_out", "connector_in",
    "net_reuse_gap", "make_step_scorer", "greedy_schedule",
    "beam_search_schedule", "good_start_set", "multistart_greedy_schedules",
    "exact_state_greedy",
]
