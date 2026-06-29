"""
BO scorer-tuning harness (Mode A).

Tunes the paired step-scorer weights for a fixed consumer, judged by the
exact simulator.  ``w_immediate`` is anchored at 1.0 (removing the scale
degeneracy and making the greedy-equivalent config — all aux weights 0 —
a literal point in the space, so a directional run can never lose to
greedy-D).  The tuned parameters are::

    w_regret, w_cache, w_connector, connector_rho

The fairness currency is exact-sim evaluations, not trials: a consumer
that completes K schedules per weight vector (multistart-K, beam-B) costs
K evals per trial, and every backend is compared at equal *evals*.

Backends: ``random`` (the floor — is BO learning at all?), ``neighbor``
(local refinement around the best), ``tpe`` (Optuna, real BO).  SMAC-RF
is a planned add.  All share warm-up seeding, best-so-far tracking,
eval-counting, and patience early stopping with a logged reason.
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from dataclasses import dataclass, field

from src.bayesopt.step_scorer import ScorerWeights

# objective(weights) -> (best_F_hit, n_exact_evals_consumed)
Objective = Callable[[ScorerWeights], tuple[float, int]]


@dataclass
class Param:
    name: str
    low: float
    high: float


DEFAULT_SPACE: list[Param] = [
    Param("w_regret", 0.0, 2.0),
    Param("w_cache", 0.0, 2.0),
    Param("w_connector", 0.0, 2.0),
    Param("connector_rho", 0.1, 0.5),
]


def weights_from_vec(vec: dict[str, float], cache_mode: str) -> ScorerWeights:
    return ScorerWeights(
        w_immediate=1.0,
        w_regret=vec["w_regret"],
        w_cache=vec["w_cache"],
        w_connector=vec["w_connector"],
        connector_rho=vec["connector_rho"],
        cache_mode=cache_mode,
    )


def warmup_vectors() -> list[dict[str, float]]:
    """Structured seeds: greedy-equiv, +each term, full."""
    base = {"w_regret": 0.0, "w_cache": 0.0, "w_connector": 0.0,
            "connector_rho": 0.25}
    outs = [dict(base)]  # greedy-equivalent (aux weights 0)
    for k in ("w_regret", "w_cache", "w_connector"):
        v = dict(base)
        v[k] = 0.5
        outs.append(v)
    full = dict(base)
    full["w_regret"] = full["w_cache"] = full["w_connector"] = 0.5
    outs.append(full)
    return outs


@dataclass
class Trial:
    vec: dict[str, float]
    fhit: float
    evals: int
    is_warmup: bool


@dataclass
class BOResult:
    backend: str
    best_fhit: float
    best_vec: dict[str, float]
    n_trials: int
    n_evals: int
    early_stop_reason: str
    history: list[Trial] = field(default_factory=list)


@dataclass
class Convergence:
    """Trajectory diagnostics for a single BO run."""

    best_trial_index: int        # 0-based trial that first hit the best F_hit
    best_from_warmup: bool       # was the best a warm-up seed, not search?
    n_warmup: int
    best_so_far: list[float]     # running-best F_hit per trial
    dominant_term: str           # aux weight with the largest magnitude at best


def convergence_of(result: BOResult) -> Convergence:
    """Derive the best-so-far trajectory and where/when the best was found."""
    best_so_far: list[float] = []
    run_best = -1.0
    best_idx = -1
    best_warm = False
    for i, t in enumerate(result.history):
        if t.fhit > run_best + 1e-12:
            run_best = t.fhit
            best_idx = i
            best_warm = t.is_warmup
        best_so_far.append(run_best)
    n_warmup = sum(1 for t in result.history if t.is_warmup)
    aux = {k: abs(v) for k, v in result.best_vec.items() if k != "connector_rho"}
    dominant = max(aux, key=lambda k: aux[k]) if aux and max(aux.values()) > 0 \
        else "none(greedy-equiv)"
    return Convergence(best_idx, best_warm, n_warmup, best_so_far, dominant)


def history_records(result: BOResult) -> list[dict[str, float | int | bool | str]]:
    """Flat per-trial records (vec unpacked) for JSON/CSV serialization."""
    recs: list[dict[str, float | int | bool | str]] = []
    run_best = -1.0
    for i, t in enumerate(result.history):
        run_best = max(run_best, t.fhit)
        rec: dict[str, float | int | bool | str] = {
            "trial": i, "is_warmup": t.is_warmup, "evals": t.evals,
            "fhit": t.fhit, "best_so_far": run_best}
        rec.update(t.vec)
        recs.append(rec)
    return recs


def run_search(
    objective: Objective,
    backend: str,
    eval_budget: int,
    *,
    space: list[Param] | None = None,
    patience: int = 15,
    seed: int = 42,
    cache_mode: str = "fit",
) -> BOResult:
    """Run one backend to an exact-eval budget; warm-up first, then search."""
    space = space or DEFAULT_SPACE
    rng = _random.Random(seed)
    history: list[Trial] = []
    n_evals = 0
    best_fhit = -1.0
    best_vec: dict[str, float] = {}
    no_improve = 0
    stop_reason = "budget"

    def record(vec: dict[str, float], is_warmup: bool) -> None:
        nonlocal n_evals, best_fhit, best_vec, no_improve
        fhit, evals = objective(weights_from_vec(vec, cache_mode))
        n_evals += evals
        history.append(Trial(dict(vec), fhit, evals, is_warmup))
        if fhit > best_fhit + 1e-12:
            best_fhit = fhit
            best_vec = dict(vec)
            no_improve = 0
        else:
            no_improve += 1

    def sample_random() -> dict[str, float]:
        return {p.name: rng.uniform(p.low, p.high) for p in space}

    def perturb(center: dict[str, float], frac: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in space:
            span = (p.high - p.low) * frac
            out[p.name] = min(p.high, max(p.low,
                              center[p.name] + rng.uniform(-span, span)))
        return out

    # --- warm-up (counts against the budget, never early-stopped) ---
    for vec in warmup_vectors():
        if n_evals >= eval_budget:
            break
        record(vec, is_warmup=True)

    # --- backend search ---
    if backend == "random":
        while n_evals < eval_budget:
            record(sample_random(), is_warmup=False)

    elif backend == "neighbor":
        while n_evals < eval_budget:
            vec = perturb(best_vec, 0.15) if best_vec else sample_random()
            record(vec, is_warmup=False)
            if no_improve >= patience:
                stop_reason = "patience"
                break

    elif backend == "tpe":
        try:
            import optuna  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - VM-only
            raise RuntimeError(
                "tpe backend requires optuna (install on the VM)") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        # enqueue warm-up so TPE starts from the structured seeds
        for vec in warmup_vectors():
            study.enqueue_trial(vec)

        def opt_objective(trial: "optuna.Trial") -> float:
            vec = {p.name: trial.suggest_float(p.name, p.low, p.high)
                   for p in space}
            fhit, evals = objective(weights_from_vec(vec, cache_mode))
            nonlocal n_evals
            n_evals += evals
            history.append(Trial(dict(vec), fhit, evals, False))
            return 1.0 - fhit

        while n_evals < eval_budget:
            study.optimize(opt_objective, n_trials=1)
        bp = study.best_params
        best_vec = {p.name: float(bp[p.name]) for p in space}
        best_fhit = 1.0 - study.best_value

    else:
        raise ValueError(f"unknown backend {backend!r}")

    return BOResult(backend, best_fhit, best_vec, len(history), n_evals,
                    stop_reason, history)


__all__ = ["Param", "DEFAULT_SPACE", "weights_from_vec", "warmup_vectors",
           "Trial", "BOResult", "Convergence", "convergence_of",
           "history_records", "run_search"]
