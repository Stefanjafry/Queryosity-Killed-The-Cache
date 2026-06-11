"""
Bayesian-optimization consumers for the directional scheduling pipeline.

Mode B (this package, first deliverable): BO as a *direct scheduler*.
A continuous priority vector p ∈ [0, 1]^n is decoded into a permutation
by descending argsort (ties broken by query index), and every candidate
schedule is scored by the **exact clock-sweep simulator** — never by the
edge-sum surrogate.  Backends share one trial recorder so that random
search, SMAC3 and BoTorch are compared at identical evaluation budgets
with identical logging.

Mode A (BO-tuned D scorer) is intentionally absent; it is built only
after Mode B has been implemented, audited and interpreted.
"""
