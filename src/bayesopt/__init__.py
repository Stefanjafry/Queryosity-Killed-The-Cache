"""
Bayesian-optimization support for the directional scheduling pipeline.

This package provides the database-free profile loader
(:mod:`src.bayesopt.data`), the memoized exact clock-sweep objective
(:mod:`src.bayesopt.objective`), and fresh-regeneration of the
incumbent baselines (:mod:`src.bayesopt.run_baselines`) — greedy_d,
ga_m, ga_d — all scored on the exact simulator in a single consistent
cost unit (1 - F_hit) so that any BO consumer can be compared against
them directly.

The direct-scheduler (priority-vector) BO experiment that previously
lived here was evaluated, found non-competitive with the D-specialized
consumers, and removed; only the shared infrastructure above remains.
"""
