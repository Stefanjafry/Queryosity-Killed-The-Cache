from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    ClockSweepCache,
    MAX_USAGE_COUNT,
    PageClockSweepCache,
    SimulationResult,
    approximate_schedule_fitness,
    approximate_schedule_fitness_directional,
    compute_directional_matrix,
    compute_directional_reusable_sets,
    compute_overlap_matrix,
    compute_residual,
    encode_page_sets,
    simulate_schedule,
    simulate_schedule_page_level,
)


class TestClockSweepCache:
    def test_hit_on_second_access(self):
        cache = ClockSweepCache(capacity_pages=100)
        assert cache.access("t1", 10) is False  # miss
        assert cache.access("t1", 10) is True  # hit

    def test_eviction_when_full(self):
        cache = ClockSweepCache(capacity_pages=20)
        cache.access("t1", 10)  # miss, used=10
        cache.access("t2", 10)  # miss, used=20
        # t3 needs space; clock sweeps t1 (usage 1->0) then t2 (1->0),
        # then wraps and evicts t1 (usage==0).
        cache.access("t3", 10)  # miss, evicts one table
        # At least one of the earlier tables must have been evicted.
        evicted = (
            cache.access("t1", 10) is False
            or cache.access("t2", 10) is False
        )
        assert evicted

    def test_usage_count_protects_hot_entries(self):
        """Repeatedly accessed tables survive eviction sweeps."""
        cache = ClockSweepCache(capacity_pages=20)
        cache.access("hot", 10)
        # Bump hot's usage count several times
        for _ in range(4):
            cache.access("hot", 10)

        cache.access("cold", 10)  # fills cache

        # Insert a new table — clock must sweep through hot (high usage)
        # and cold (usage=1) before evicting.  cold should be evicted
        # because its usage count drops to 0 first.
        cache.access("new", 10)
        assert cache.access("hot", 10) is True  # survived
        assert cache.access("cold", 10) is False  # was evicted

    def test_usage_count_capped(self):
        """Usage count should never exceed MAX_USAGE_COUNT."""
        cache = ClockSweepCache(capacity_pages=100)
        cache.access("t1", 10)
        for _ in range(MAX_USAGE_COUNT + 5):
            cache.access("t1", 10)
        idx = cache._lookup["t1"]
        assert cache._usage[idx] == MAX_USAGE_COUNT

    def test_oversized_entry(self):
        cache = ClockSweepCache(capacity_pages=5)
        cache.access("t1", 3)
        # t_big exceeds capacity — not inserted, t1 survives
        assert cache.access("t_big", 10) is False
        assert cache.access("t1", 3) is True  # still cached

    def test_reset(self):
        cache = ClockSweepCache(capacity_pages=100)
        cache.access("t1", 10)
        cache.reset()
        assert cache.used == 0
        assert cache.access("t1", 10) is False  # miss after reset

    def test_zero_capacity(self):
        cache = ClockSweepCache(capacity_pages=0)
        assert cache.access("t1", 5) is False
        assert cache.access("t1", 5) is False  # never cached


class TestPageClockSweepBatchAccess:
    def test_batch_hits_and_misses(self):
        cache = PageClockSweepCache(capacity_pages=10)
        # First access — all misses
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0

        # Second access — all hits
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 3

    def test_batch_partial_overlap(self):
        cache = PageClockSweepCache(capacity_pages=10)
        cache.batch_access(frozenset({0, 1, 2}))
        # Partial overlap: 0 and 1 are hits, 3 is a miss
        hits = cache.batch_access(frozenset({0, 1, 3}))
        assert hits == 2

    def test_batch_eviction(self):
        cache = PageClockSweepCache(capacity_pages=3)
        cache.batch_access(frozenset({0, 1, 2}))  # fills cache
        cache.batch_access(frozenset({3, 4, 5}))  # evicts all old pages
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0  # all evicted

    def test_batch_consistent_with_individual(self):
        """batch_access should produce the same hit count as individual access calls."""
        pages = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]

        # Batch path
        cache_batch = PageClockSweepCache(capacity_pages=5)
        batch_hits = sum(cache_batch.batch_access(ps) for ps in pages)

        # Individual path
        cache_individual = PageClockSweepCache(capacity_pages=5)
        individual_hits = 0
        for ps in pages:
            for page in ps:
                if cache_individual.access(page):
                    individual_hits += 1

        assert batch_hits == individual_hits

    def test_batch_zero_capacity(self):
        cache = PageClockSweepCache(capacity_pages=0)
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0


class TestOverlapMatrix:
    def test_symmetric(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]
        matrix = compute_overlap_matrix(page_sets)
        n = len(page_sets)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]

    def test_overlap_values(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]
        matrix = compute_overlap_matrix(page_sets)
        # {0,1,2} & {2,3,4} = {2} → 1
        assert matrix[0][1] == 1
        # {0,1,2} & {0,4,5} = {0} → 1
        assert matrix[0][2] == 1
        # {2,3,4} & {0,4,5} = {4} → 1
        assert matrix[1][2] == 1
        # Diagonal (self-overlap) is 0 by construction
        assert matrix[0][0] == 0

    def test_empty(self):
        assert compute_overlap_matrix([]) == []


class TestApproximateScheduleFitness:
    def test_basic_ordering(self):
        """Schedule that groups overlapping queries should score higher."""
        page_sets = [
            frozenset({0, 1, 2, 3, 4}),       # q0: 5 pages
            frozenset({0, 1, 2, 3, 4, 5, 6}), # q1: 7 pages, shares 5 with q0
            frozenset({10, 11, 12}),            # q2: 3 pages, no overlap
        ]
        page_counts = [len(ps) for ps in page_sets]
        matrix = compute_overlap_matrix(page_sets)

        # cache=6: only the immediate predecessor fits
        #   good [0,1,2]: k=1 (q1): overlap(q0,q1)=5, no discount → 5 hits
        #                  k=2 (q2): budget=6-7=-1 < 0 → 0 hits. total=5/15
        #   bad  [0,2,1]: k=1 (q2): overlap(q0,q2)=0 → 0 hits
        #                  k=2 (q1): budget=6-3=3 ≥ 0 → overlap(q2,q1)=0
        #                            budget=3-5=-2 < 0 → break → 0 hits. total=0/15
        good = approximate_schedule_fitness(
            matrix, page_counts, [0, 1, 2], cache_capacity_pages=6,
        )
        bad = approximate_schedule_fitness(
            matrix, page_counts, [0, 2, 1], cache_capacity_pages=6,
        )
        assert good > bad

    def test_empty_schedule(self):
        assert approximate_schedule_fitness([], [], [], 100) == 0.0

    def test_single_query(self):
        matrix = [[0]]
        page_counts = [10]
        # Single query — no predecessor, so 0 hits
        f = approximate_schedule_fitness(matrix, page_counts, [0], 100)
        assert f == 0.0


class TestSimulationResult:
    def test_hit_ratio(self):
        r = SimulationResult(total_requests=100, total_hits=25)
        assert r.hit_ratio == 0.25

    def test_hit_ratio_zero_requests(self):
        r = SimulationResult(total_requests=0, total_hits=0)
        assert r.hit_ratio == 0.0


class TestSimulateSchedule:
    def _make_profiles(self) -> list[AccessProfile]:
        """Three queries: q0 and q2 share table A, q1 uses table B."""
        return [
            AccessProfile(query_id="q0", table_pages={"A": 10}),
            AccessProfile(query_id="q1", table_pages={"B": 10}),
            AccessProfile(query_id="q2", table_pages={"A": 10}),
        ]

    def test_adjacent_same_table_gives_hit(self):
        profiles = self._make_profiles()
        # q0 then q2: q2 hits on table A
        result = simulate_schedule(profiles, [0, 2, 1], cache_capacity_pages=100)
        assert result.total_hits == 10  # q2's access to A
        assert result.total_requests == 30

    def test_separated_same_table_may_miss(self):
        profiles = self._make_profiles()
        # q0, q1, q2 with tiny cache: table A evicted by B
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=10)
        assert result.total_hits == 0

    def test_large_cache_all_hit(self):
        profiles = self._make_profiles()
        # large cache: both A and B fit, q2 hits A
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        assert result.total_hits == 10
        assert result.hit_ratio == 10 / 30

    def test_identity_schedule(self):
        profiles = self._make_profiles()
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        result2 = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        assert result.total_hits == result2.total_hits  # deterministic

    def test_empty_schedule(self):
        result = simulate_schedule([], [], cache_capacity_pages=100)
        assert result.total_requests == 0
        assert result.total_hits == 0


class TestSimulateSchedulePageLevel:
    def test_basic_hits(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
        ]
        result = simulate_schedule_page_level(page_sets, [0, 1], cache_capacity_pages=10)
        # Page 2 is a hit for q1
        assert result.total_hits == 1
        assert result.total_requests == 6

    def test_empty(self):
        result = simulate_schedule_page_level([], [], cache_capacity_pages=10)
        assert result.total_requests == 0
        assert result.total_hits == 0

class TestComputeResidual:
    def test_empty_pages(self):
        assert compute_residual(frozenset(), cache_capacity_pages=10) == frozenset()

    def test_zero_capacity(self):
        assert compute_residual(frozenset({0, 1, 2}), cache_capacity_pages=0) == frozenset()

    def test_fits_in_cache(self):
        # |P| <= C => everything survives, R == P
        pages = frozenset({0, 1, 2, 3})
        residual = compute_residual(pages, cache_capacity_pages=10)
        assert residual == pages

    def test_exact_fit(self):
        # |P| == C => still everything fits, R == P
        pages = frozenset({0, 1, 2, 3})
        residual = compute_residual(pages, cache_capacity_pages=4)
        assert residual == pages

    def test_subset_when_oversized(self):
        # |P| > C => R is a non-empty subset of P bounded by capacity
        pages = frozenset(range(20))
        residual = compute_residual(pages, cache_capacity_pages=5)
        assert residual.issubset(pages)
        assert len(residual) <= 5
        # Under clock-sweep on a cold cache with capacity 5 and 20 inserts,
        # the cache fills to capacity and stays full.
        assert len(residual) == 5


class TestComputeDirectionalMatrix:
    def test_empty(self):
        assert compute_directional_matrix([], cache_capacity_pages=10) == []

    def test_diagonal_is_zero(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({4, 5, 6}),
        ]
        D = compute_directional_matrix(page_sets, cache_capacity_pages=100)
        for i in range(len(D)):
            assert D[i][i] == 0

    def test_directional_equals_symmetric_when_no_eviction(self):
        # Invariant: when every query fits entirely in cache,
        # D[i][j] == M[i][j] for all i != j.
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({4, 5, 6, 0}),
        ]
        big_capacity = 100
        D = compute_directional_matrix(page_sets, cache_capacity_pages=big_capacity)
        M = compute_overlap_matrix(page_sets)
        n = len(page_sets)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert D[i][j] == M[i][j], (
                    f"D[{i}][{j}]={D[i][j]} vs M[{i}][{j}]={M[i][j]}"
                )

    def test_directional_bounded_by_symmetric(self):
        # Invariant: residual is always a subset of P(Qi),
        # so D[i][j] <= M[i][j] for all i, j.
        # Use a workload with real eviction pressure so the bound is
        # strict for at least one pair.
        page_sets = [
            frozenset(range(0, 30)),    # Lq overlapping with q1 on {10..19}
            frozenset(range(10, 20)),   # small
            frozenset(range(15, 45)),   # another Lq
        ]
        capacity = 12  # forces eviction for the large queries
        D = compute_directional_matrix(page_sets, cache_capacity_pages=capacity)
        M = compute_overlap_matrix(page_sets)
        n = len(page_sets)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert D[i][j] <= M[i][j], (
                    f"D[{i}][{j}]={D[i][j]} exceeds M[{i}][{j}]={M[i][j]}"
                )

    def test_directional_asymmetry_under_pressure(self):
        # Construct a clear large -> small scenario.
        # Lq has 30 pages, capacity is 10, so 20 of them evict before
        # any subsequent query starts.  Small Sq has 5 pages that all
        # appear in Lq.  M[L][S] = 5, but D[L][S] can be strictly less.
        Lq = frozenset(range(30))
        Sq = frozenset({0, 1, 2, 3, 4})
        page_sets = [Lq, Sq]
        capacity = 10

        D = compute_directional_matrix(page_sets, cache_capacity_pages=capacity)
        M = compute_overlap_matrix(page_sets)

        # All of Sq's pages overlap with Lq, so M is symmetric and equal to |Sq|.
        assert M[0][1] == 5
        assert M[1][0] == 5
        # Sq fits entirely, so its residual is Sq itself; D[Sq][Lq] = 5.
        assert D[1][0] == 5
        # Lq does not fit; pages 0..19 are evicted under clock-sweep.
        # Therefore D[Lq][Sq] < 5 (most of Sq's pages were among the evicted).
        assert D[0][1] < M[0][1]


class TestApproximateScheduleFitnessDirectional:
    def test_empty_schedule(self):
        f = approximate_schedule_fitness_directional(
            [], [], [], cache_capacity_pages=10,
        )
        assert f == 0.0

    def test_single_query(self):
        # No predecessor => no hits estimated.
        page_sets = [frozenset({0, 1, 2})]
        reusable = compute_directional_reusable_sets(
            page_sets, cache_capacity_pages=10,
        )
        f = approximate_schedule_fitness_directional(
            reusable, [3], [0], cache_capacity_pages=10,
        )
        assert f == 0.0

    def test_edge_sum(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({1, 2, 3}),
        ]
        # No eviction pressure, so reusable[0][1] = {1, 2}, |.| = 2.
        reusable = compute_directional_reusable_sets(
            page_sets, cache_capacity_pages=100,
        )
        page_counts = [3, 3]
        # Schedule q0 then q1: 2 estimated hits out of 6 page requests.
        f = approximate_schedule_fitness_directional(
            reusable, page_counts, [0, 1], cache_capacity_pages=100,
        )
        assert abs(f - 2 / 6) < 1e-9

    def test_directional_distinguishes_orderings(self):
        # Lq has 30 pages, Sq's 5 pages are all in Lq.  Capacity 10:
        # only ~10 of Lq survive (and probably not Sq's 5 pages).
        # Schedule Sq -> Lq should score better than Lq -> Sq under the
        # directional approximation because all of Sq survives.
        Lq = frozenset(range(30))
        Sq = frozenset({0, 1, 2, 3, 4})
        page_sets = [Lq, Sq]
        capacity = 10
        reusable = compute_directional_reusable_sets(
            page_sets, cache_capacity_pages=capacity,
        )
        page_counts = [len(ps) for ps in page_sets]
        fit_L_first = approximate_schedule_fitness_directional(
            reusable, page_counts, [0, 1], cache_capacity_pages=capacity,
        )
        fit_S_first = approximate_schedule_fitness_directional(
            reusable, page_counts, [1, 0], cache_capacity_pages=capacity,
        )
        assert fit_S_first > fit_L_first

    def test_windowed_credits_non_immediate_predecessor(self):
        # Q0 brings pages {0, 1, 2, 3} fully into cache.
        # Q1 brings disjoint pages {10, 11} — small, doesn't evict Q0.
        # Q2 needs pages {0, 1, 2, 3} — none in Q1's residue, all in
        # Q0's residue.
        # A single-step fitness would credit Q2 with 0 hits (since
        # |residue(Q1) ∩ pages(Q2)| = 0).  The windowed fitness must
        # walk past Q1 and credit all 4 from Q0, because the cache
        # budget at C=10 easily holds both Q0 and Q1.
        page_sets = [
            frozenset({0, 1, 2, 3}),
            frozenset({10, 11}),
            frozenset({0, 1, 2, 3}),
        ]
        capacity = 10
        reusable = compute_directional_reusable_sets(
            page_sets, cache_capacity_pages=capacity,
        )
        page_counts = [len(ps) for ps in page_sets]
        f = approximate_schedule_fitness_directional(
            reusable, page_counts, [0, 1, 2], cache_capacity_pages=capacity,
        )
        # Total page requests = 4 + 2 + 4 = 10.
        # Hits: Q1 has none from Q0 ({10,11} ∩ {0,1,2,3} = empty);
        #       Q2 picks up 4 from Q0 via the window.
        # So expected fitness = 4 / 10.
        assert abs(f - 4 / 10) < 1e-9

    def test_windowed_does_not_double_count_pages(self):
        # Q0 and Q1 both hold pages {0, 1, 2, 3} fully.  Q2 needs the
        # same pages.  reusable[0][2] = reusable[1][2] = {0,1,2,3}.
        # The exact set discount must count those four pages exactly
        # once for Q2 — not eight.
        page_sets = [
            frozenset({0, 1, 2, 3}),
            frozenset({0, 1, 2, 3}),
            frozenset({0, 1, 2, 3}),
        ]
        capacity = 100  # everything fits
        reusable = compute_directional_reusable_sets(
            page_sets, cache_capacity_pages=capacity,
        )
        page_counts = [len(ps) for ps in page_sets]
        f = approximate_schedule_fitness_directional(
            reusable, page_counts, [0, 1, 2], cache_capacity_pages=capacity,
        )
        # Q1 picks up 4 hits from Q0; Q2 picks up 4 hits (not 8) from
        # the window {Q1, Q0}.  Total hits = 8.  Total requests = 12.
        assert abs(f - 8 / 12) < 1e-9


class TestEncodePageSetsDeterminism:
    """
    Regression tests against ``encode_page_sets`` returning different
    integer encodings across Python invocations due to PYTHONHASHSEED
    salting string hashes.  The simulator's downstream behaviour (clock
    sweep eviction order) depends on the integer values, so
    non-determinism here propagated into ``H_total`` drift across
    otherwise-identical runs.
    """

    def test_encoding_is_deterministic_within_process(self):
        # Identical inputs must produce identical encodings.
        ps1 = [{("lineitem", b) for b in range(0, 50)},
               {("orders", b) for b in range(0, 30)}]
        ps2 = [{("lineitem", b) for b in range(0, 50)},
               {("orders", b) for b in range(0, 30)}]
        e1, _ = encode_page_sets(ps1)
        e2, _ = encode_page_sets(ps2)
        assert e1 == e2

    def test_id_assignment_is_sorted_order(self):
        # With sorted iteration, ('a', 0) gets ID 0 because it sorts
        # first.  This pins the encoding so reviewers can reason about
        # the integer values without needing to know PYTHONHASHSEED.
        ps = [{("z_table", 5), ("a_table", 0), ("a_table", 1)}]
        encoded, page_to_id = encode_page_sets(ps)
        assert page_to_id[("a_table", 0)] == 0
        assert page_to_id[("a_table", 1)] == 1
        assert page_to_id[("z_table", 5)] == 2
        assert encoded[0] == frozenset({0, 1, 2})
