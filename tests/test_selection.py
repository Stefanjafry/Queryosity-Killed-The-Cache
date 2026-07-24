"""Selection helpers: dedup correctness and worker-count invariance."""

from __future__ import annotations

from src.bayesopt.selection import dedup_index, resolve_workers, score_pool

PAGE_SETS: list[frozenset[int]] = [
    frozenset({1, 2, 3}),
    frozenset({3, 4, 5}),
    frozenset({5, 6, 7}),
    frozenset({1, 7, 8}),
]


def test_dedup_index_preserves_order_and_maps_back() -> None:
    a, b, c = (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 2, 3)
    distinct, index = dedup_index([a, b, c, a])
    assert distinct == [a, b]
    assert index == [0, 1, 0, 0]


def test_dedup_index_empty() -> None:
    assert dedup_index([]) == ([], [])


def test_scores_map_back_to_every_input_candidate() -> None:
    cands = [(0, 1, 2, 3), (3, 2, 1, 0), (0, 1, 2, 3)]
    scores, n_distinct = score_pool(cands, PAGE_SETS, 4, workers=1)
    assert n_distinct == 2
    assert len(scores) == 3
    assert scores[0] == scores[2]


def test_parallel_matches_serial_exactly() -> None:
    cands = [
        (0, 1, 2, 3), (1, 0, 2, 3), (3, 2, 1, 0),
        (0, 2, 1, 3), (0, 1, 2, 3), (2, 3, 0, 1),
    ]
    serial, n1 = score_pool(cands, PAGE_SETS, 3, workers=1)
    par, n2 = score_pool(cands, PAGE_SETS, 3, workers=2)
    assert n1 == n2 == 5
    assert serial == par


def test_argmax_is_worker_count_invariant() -> None:
    cands = [
        (0, 1, 2, 3), (1, 0, 2, 3), (3, 2, 1, 0),
        (0, 2, 1, 3), (2, 3, 0, 1), (1, 3, 0, 2),
    ]

    def winner(w: int) -> tuple[int, ...]:
        scores, _ = score_pool(cands, PAGE_SETS, 3, workers=w)
        bi, bf = 0, -1.0
        for i, f in enumerate(scores):
            if f > bf:      # strict: earliest construction order wins ties
                bf, bi = f, i
        return cands[bi]

    assert winner(1) == winner(2) == winner(3)


def test_empty_pool() -> None:
    assert score_pool([], PAGE_SETS, 4, workers=2) == ([], 0)


def test_resolve_workers_clamps() -> None:
    assert resolve_workers(1) == 1
    assert resolve_workers(0) >= 1
    assert resolve_workers(9999) >= 1
    assert resolve_workers(-3) == resolve_workers(0)
