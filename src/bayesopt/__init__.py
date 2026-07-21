"""
Directional scheduling pipeline: profile loading, exact objective,
baselines, and the regret-sweep scheduler.

This package provides the database-free profile loader
(:mod:`src.bayesopt.data`), the memoized exact clock-sweep objective
(:mod:`src.bayesopt.objective`), fresh regeneration of the incumbent
baselines (:mod:`src.bayesopt.run_baselines`) — greedy_d, ga_m, ga_d —
and the production scheduler (:mod:`src.bayesopt.run_regret_sweep`):
a 1-D exhaustive sweep of the regret weight over the directional step
scorer, consumed by multistart-greedy and beam search, with every
candidate scored by the exact simulator.

Bayesian-optimization experiments previously developed here (SMAC /
BoTorch scorer tuning, Mode A structured search) were retired after
the tuned auxiliary weights collapsed to zero in 8/9 configurations,
leaving w_regret as the only active dimension; the exhaustive sweep
matches BO within measurement noise while being deterministic. That
code is preserved on the archive branch (see the README).
"""
