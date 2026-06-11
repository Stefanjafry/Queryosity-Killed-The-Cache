"""
Backend-agnostic search harness for Mode B.

Every optimizer backend receives the same :class:`TrialRecorder`
callable.  The recorder owns the entire trial lifecycle — budget
enforcement, priority decoding, permutation validation, exact-sim
evaluation, incumbent tracking and JSONL logging — so that random
search, SMAC3 and BoTorch are compared on identical terms and produce
identical log schemas.  A backend's only job is to propose priority
vectors and call the recorder.

The public entry point is :func:`run_search`, the
``run(space, objective, budget) → SearchRun`` abstraction from the
project briefing.  Backends are imported lazily so that, e.g., running
SMAC never imports torch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Sequence

from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.priority import decode_priority_vector, is_valid_permutation

MODE = "priority_vector"
"""Mode tag written into every trial record (Mode B)."""

BACKENDS = ("random", "smac", "botorch_gp")
"""Implemented Mode B optimizer backends."""


class BudgetExhausted(RuntimeError):
    """Raised if a backend requests more evaluations than its budget."""


@dataclass(frozen=True)
class PrioritySpace:
    """
    Mode B search space: the unit hypercube ``[0, 1]^n``.

    Attributes
    ----------
    n : int
        Number of queries, i.e. the dimensionality of the space.
    """

    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("search space needs at least one query")


@dataclass(frozen=True)
class SearchRun:
    """
    Outcome of one optimizer run at one (workload, cache, seed, budget).

    Attributes
    ----------
    backend : str
        Optimizer backend name.
    budget : int
        Evaluation budget the run was granted.
    n_trials : int
        Evaluations actually consumed (== budget on a healthy run).
    seed : int
        Random seed used by the backend.
    best_trial : int
        1-based trial index at which the incumbent was found.
    best_cost : float
        Incumbent cost (1 − F_hit).
    best_hit_ratio : float
        Incumbent simulated F_hit.
    best_page_reads : int
        Incumbent simulated page reads.
    best_schedule : list[int]
        Incumbent permutation (query indices).
    best_schedule_qids : list[str]
        Incumbent permutation as query ids, directly consumable by the
        wall-clock executor.
    best_priorities : list[float]
        Priority vector that decoded to the incumbent.
    best_so_far : list[float]
        Best cost after each trial (length ``n_trials``); the
        convergence curve required by the evaluation plan.
    wall_seconds : float
        Total optimizer wall time including simulator evaluations.
    """

    backend: str
    budget: int
    n_trials: int
    seed: int
    best_trial: int
    best_cost: float
    best_hit_ratio: float
    best_page_reads: int
    best_schedule: list[int]
    best_schedule_qids: list[str]
    best_priorities: list[float]
    best_so_far: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0


class TrialRecorder:
    """
    Shared per-trial bookkeeping for every backend.

    Calling the recorder with a priority vector decodes it, validates
    the permutation, evaluates the exact simulator, appends one JSONL
    record, updates the incumbent (strict improvement only, so the
    first achiever of a cost keeps the incumbency deterministically)
    and returns the cost for the backend to minimize.

    Parameters
    ----------
    objective : ExactSimObjective
        Memoized exact-simulator objective.
    query_ids : list[str]
        Query ids aligned with the objective's page sets.
    budget : int
        Maximum number of trials; exceeding it raises
        :class:`BudgetExhausted`.
    static_fields : dict
        Run-level fields (workload, cache_pages, backend, seed, …)
        copied into every trial record.
    trials_stream : IO[str] | None
        Open text stream for JSONL trial records; None disables
        logging (used by unit tests that only check return values).
    """

    def __init__(
        self,
        objective: ExactSimObjective,
        query_ids: list[str],
        budget: int,
        static_fields: dict[str, object],
        trials_stream: IO[str] | None = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if len(query_ids) != objective.n:
            raise ValueError("query_ids length does not match objective")
        self.objective = objective
        self.query_ids = query_ids
        self.budget = budget
        self.static_fields = dict(static_fields)
        self._stream = trials_stream
        self.n_trials = 0
        self.best_cost = float("inf")
        self.best_trial = 0
        self.best_metrics = None
        self.best_schedule: list[int] = []
        self.best_priorities: list[float] = []
        self.best_so_far: list[float] = []
        self._t_start = time.perf_counter()

    def __call__(self, priorities: Sequence[float]) -> float:
        """Evaluate one priority vector; returns the cost to minimize."""
        if self.n_trials >= self.budget:
            raise BudgetExhausted(
                f"backend requested trial {self.n_trials + 1} "
                f"with budget {self.budget}"
            )
        vec = [float(p) for p in priorities]
        if len(vec) != self.objective.n:
            raise ValueError(
                f"priority vector has length {len(vec)}, "
                f"expected {self.objective.n}"
            )
        self.n_trials += 1

        t0 = time.perf_counter()
        schedule = decode_priority_vector(vec)
        valid = is_valid_permutation(schedule, self.objective.n)
        if not valid:
            self._write_record(vec, schedule, None, False, False, 0.0)
            raise ValueError(f"decoded schedule is not a permutation: {schedule}")
        metrics, memo_hit = self.objective.evaluate_schedule(schedule)
        eval_seconds = time.perf_counter() - t0

        if metrics.cost < self.best_cost:
            self.best_cost = metrics.cost
            self.best_trial = self.n_trials
            self.best_metrics = metrics
            self.best_schedule = schedule
            self.best_priorities = vec
        self.best_so_far.append(self.best_cost)

        self._write_record(vec, schedule, metrics, True, memo_hit, eval_seconds)
        return metrics.cost

    def _write_record(
        self,
        priorities: list[float],
        schedule: list[int],
        metrics,
        valid: bool,
        memo_hit: bool,
        eval_seconds: float,
    ) -> None:
        if self._stream is None:
            return
        record: dict[str, object] = dict(self.static_fields)
        record.update(
            mode=MODE,
            trial=self.n_trials,
            priorities=priorities,
            schedule=schedule,
            schedule_qids=[self.query_ids[i] for i in schedule],
            valid=valid,
            memo_hit=memo_hit,
            eval_seconds=round(eval_seconds, 6),
            cum_seconds=round(time.perf_counter() - self._t_start, 6),
            best_cost_so_far=None if self.best_trial == 0 else self.best_cost,
        )
        if metrics is not None:
            record.update(
                cost=metrics.cost,
                sim_hit_ratio=metrics.hit_ratio,
                sim_hits=metrics.total_hits,
                sim_requests=metrics.total_requests,
                sim_page_reads=metrics.page_reads,
                edge_sum_d=metrics.edge_sum_d,
            )
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()

    def finalize(self, backend: str, seed: int) -> SearchRun:
        """Package the run outcome after the backend returns."""
        if self.best_metrics is None:
            raise RuntimeError("finalize() called before any trial ran")
        return SearchRun(
            backend=backend,
            budget=self.budget,
            n_trials=self.n_trials,
            seed=seed,
            best_trial=self.best_trial,
            best_cost=self.best_cost,
            best_hit_ratio=self.best_metrics.hit_ratio,
            best_page_reads=self.best_metrics.page_reads,
            best_schedule=self.best_schedule,
            best_schedule_qids=[self.query_ids[i] for i in self.best_schedule],
            best_priorities=self.best_priorities,
            best_so_far=self.best_so_far,
            wall_seconds=time.perf_counter() - self._t_start,
        )


def default_n_init(budget: int) -> int:
    """
    Initial-design size shared by every backend that has one.

    ``max(5, min(25, budget // 4))`` — roughly a quarter of the budget,
    floored at 5 and capped at 25 so large budgets spend most trials on
    surrogate-guided proposals.
    """
    return max(5, min(25, budget // 4))


def run_search(
    backend: str,
    space: PrioritySpace,
    objective: ExactSimObjective,
    query_ids: list[str],
    budget: int,
    seed: int,
    n_init: int | None = None,
    static_fields: dict[str, object] | None = None,
    trials_stream: IO[str] | None = None,
    workdir: Path | None = None,
) -> SearchRun:
    """
    Run one optimizer backend against the exact-simulator objective.

    Parameters
    ----------
    backend : str
        One of :data:`BACKENDS`.
    space : PrioritySpace
        Search space; must match the objective's query count.
    objective : ExactSimObjective
        Exact-simulator objective.
    query_ids : list[str]
        Query ids aligned with the objective.
    budget : int
        Total evaluation budget (initial design included).
    seed : int
        Backend random seed.
    n_init : int | None
        Initial-design size; None applies :func:`default_n_init`.
        Ignored by the random backend.
    static_fields : dict | None
        Run-level fields copied into every trial record.
    trials_stream : IO[str] | None
        Open JSONL stream for trial records.
    workdir : Path | None
        Scratch directory for backends that write files (SMAC).

    Returns
    -------
    SearchRun
        Final incumbent and convergence history.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected {BACKENDS}")
    if space.n != objective.n:
        raise ValueError("space and objective disagree on query count")
    resolved_init = default_n_init(budget) if n_init is None else n_init
    if not 1 <= resolved_init <= budget:
        raise ValueError(
            f"n_init={resolved_init} must be in [1, budget={budget}]"
        )

    fields = dict(static_fields or {})
    fields.setdefault("backend", backend)
    fields.setdefault("seed", seed)
    recorder = TrialRecorder(
        objective, query_ids, budget, fields, trials_stream=trials_stream
    )

    if backend == "random":
        from src.bayesopt.backends.random_search import run as _run
    elif backend == "smac":
        from src.bayesopt.backends.smac_backend import run as _run
    else:
        from src.bayesopt.backends.botorch_gp import run as _run

    _run(
        recorder,
        n=space.n,
        budget=budget,
        seed=seed,
        n_init=resolved_init,
        workdir=workdir,
    )
    return recorder.finalize(backend, seed)


__all__ = [
    "BACKENDS",
    "MODE",
    "BudgetExhausted",
    "PrioritySpace",
    "SearchRun",
    "TrialRecorder",
    "default_n_init",
    "run_search",
]
