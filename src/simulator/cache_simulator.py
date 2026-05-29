"""
Cache simulation using PostgreSQL's clock-sweep eviction algorithm.

PostgreSQL does not use pure LRU for its shared buffer pool.  Instead it
employs a clock-sweep (also called second-chance) algorithm:

* Each buffer frame carries a **usage count** (0 to ``MAX_USAGE_COUNT``).
* On a **hit** the usage count is incremented (capped at the maximum).
* On a **miss** a clock hand sweeps through the buffer pool, decrementing
  each frame's usage count.  The first frame whose count reaches 0 is
  evicted and replaced with the new page.

This module also provides:

* ``PageClockSweepCache.batch_access`` — bulk hit detection via set
  intersection (C-optimised in CPython) followed by clock-sweep
  insertion of misses only, avoiding per-page dict lookups for hits.
* ``compute_overlap_matrix`` / ``approximate_schedule_fitness`` — a
  precomputed pairwise page-overlap matrix with statistical correction
  for double-counting that estimates cache hits without full clock-sweep
  simulation, framing cache-aware scheduling as an Asymmetric Travelling
  Salesman Problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulator.access_profile import AccessProfile

# PostgreSQL caps the per-buffer usage count at 5 (see bufmgr.c).
MAX_USAGE_COUNT = 5


def encode_page_sets(
    page_sets: list[set[tuple[str, int]]],
) -> tuple[list[frozenset[int]], dict[tuple[str, int], int]]:
    """
    Map (table, block) tuples to contiguous integers for faster hashing.

    Iterates the input sets in **sorted** order so that the integer
    assignment is reproducible across Python invocations.  Set iteration
    order over tuples otherwise depends on ``PYTHONHASHSEED`` (because
    string hashing is salted per-process), which would propagate into
    the clock-sweep simulator and produce different ``H_total`` values
    on identical inputs across runs.

    Returns frozensets so they can be used directly with
    ``PageClockSweepCache.batch_access`` and set-intersection operations.

    Parameters
    ----------
    page_sets : list[set[tuple[str, int]]]
        Per-query page sets using (table, block) tuples.

    Returns
    -------
    tuple[list[frozenset[int]], dict[tuple[str, int], int]]
        Integer-encoded page frozensets and the mapping used.
    """
    page_to_id: dict[tuple[str, int], int] = {}
    encoded: list[frozenset[int]] = []
    for ps in page_sets:
        int_pages: list[int] = []
        for page in sorted(ps):
            if page not in page_to_id:
                page_to_id[page] = len(page_to_id)
            int_pages.append(page_to_id[page])
        encoded.append(frozenset(int_pages))
    return encoded, page_to_id


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """
    Result of simulating a query schedule through the cache.

    Attributes
    ----------
    total_requests : int
        Total page requests across all queries in the schedule.
    total_hits : int
        Number of page requests served from cache.
    """

    total_requests: int
    total_hits: int

    @property
    def hit_ratio(self) -> float:
        """
        Cache hit ratio: total_hits / total_requests.

        Returns 0.0 when there are no requests.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests


# ---------------------------------------------------------------------------
# Clock-sweep caches
# ---------------------------------------------------------------------------


class ClockSweepCache:
    """
    Fixed-capacity clock-sweep cache operating at table granularity.

    Models PostgreSQL's shared-buffer eviction strategy.  Each cached
    table carries a usage count that is incremented on hits (up to
    ``MAX_USAGE_COUNT``) and decremented by the clock hand during
    eviction sweeps.  Tables whose page count exceeds the total cache
    capacity are never inserted.

    Parameters
    ----------
    capacity_pages : int
        Maximum number of pages the cache can hold.

    Raises
    ------
    ValueError
        If *capacity_pages* is negative.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        self.capacity = capacity_pages
        self._tables: list[str] = []
        self._pages: list[int] = []
        self._usage: list[int] = []
        self._lookup: dict[str, int] = {}
        self._used: int = 0
        self._hand: int = 0

    @property
    def used(self) -> int:
        """Number of pages currently held in the cache."""
        return self._used

    def pages_held(self, table: str) -> int:
        """
        Return the number of pages of *table* currently cached.

        Parameters
        ----------
        table : str
            Name of the table to look up.

        Returns
        -------
        int
            Page count held for *table*, or 0 if the table is not cached.
        """
        idx = self._lookup.get(table)
        if idx is None:
            return 0
        return self._pages[idx]

    def _advance_hand(self) -> None:
        """Advance the clock hand, wrapping around the buffer pool."""
        if self._tables:
            self._hand = (self._hand + 1) % len(self._tables)

    def _evict_one(self) -> int:
        """
        Sweep until a frame with usage_count == 0 is found and evict it.

        Returns
        -------
        int
            Number of pages freed by the eviction.
        """
        while True:
            if self._usage[self._hand] == 0:
                freed = self._pages[self._hand]
                evicted_table = self._tables[self._hand]

                # Replace with last element to keep arrays compact
                last = len(self._tables) - 1
                if self._hand != last:
                    self._tables[self._hand] = self._tables[last]
                    self._pages[self._hand] = self._pages[last]
                    self._usage[self._hand] = self._usage[last]
                    self._lookup[self._tables[self._hand]] = self._hand

                del self._lookup[evicted_table]
                self._tables.pop()
                self._pages.pop()
                self._usage.pop()

                if self._tables:
                    self._hand %= len(self._tables)
                else:
                    self._hand = 0
                return freed

            self._usage[self._hand] -= 1
            self._advance_hand()

    def access(self, table: str, pages: int) -> bool:
        """
        Access a table in the cache.

        On a hit the table's usage count is incremented (capped at
        ``MAX_USAGE_COUNT``).  On a miss the clock hand sweeps to free
        enough space, then the table is inserted with a usage count of 1.

        Parameters
        ----------
        table : str
            Name of the table being accessed.
        pages : int
            Number of 8 KB pages the table occupies.

        Returns
        -------
        bool
            True if the table was already in the cache (hit).
        """
        if table in self._lookup:
            idx = self._lookup[table]
            if self._usage[idx] < MAX_USAGE_COUNT:
                self._usage[idx] += 1
            return True

        if pages > self.capacity:
            return False

        while self._used + pages > self.capacity and self._tables:
            self._used -= self._evict_one()

        slot = len(self._tables)
        self._tables.append(table)
        self._pages.append(pages)
        self._usage.append(1)
        self._lookup[table] = slot
        self._used += pages
        return False

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self._tables.clear()
        self._pages.clear()
        self._usage.clear()
        self._lookup.clear()
        self._used = 0
        self._hand = 0


class PageClockSweepCache:
    """
    Fixed-capacity clock-sweep cache operating at individual page
    granularity.

    Models PostgreSQL's shared-buffer eviction strategy at the page
    level.  The buffer pool is a fixed-size array of slots.  Each slot
    stores an integer page ID and a usage count.  A clock hand sweeps
    to find eviction victims.

    Parameters
    ----------
    capacity_pages : int
        Maximum number of pages the cache can hold.

    Raises
    ------
    ValueError
        If *capacity_pages* is negative.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        self.capacity = capacity_pages
        self._page_ids: list[int] = []
        self._usage: list[int] = []
        self._lookup: dict[int, int] = {}
        self._hand: int = 0

    def _insert_page(self, page: int) -> None:
        """
        Insert a single page into the cache, evicting if necessary.

        Parameters
        ----------
        page : int
            Integer page ID to insert (must not already be cached).
        """
        if len(self._page_ids) < self.capacity:
            self._lookup[page] = len(self._page_ids)
            self._page_ids.append(page)
            self._usage.append(1)
            return

        while self._usage[self._hand] > 0:
            self._usage[self._hand] -= 1
            self._hand = (self._hand + 1) % self.capacity

        old_page = self._page_ids[self._hand]
        del self._lookup[old_page]

        self._page_ids[self._hand] = page
        self._usage[self._hand] = 1
        self._lookup[page] = self._hand
        self._hand = (self._hand + 1) % self.capacity

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        On a hit the page's usage count is incremented (capped at
        ``MAX_USAGE_COUNT``).  On a miss the clock hand sweeps until it
        finds a frame with usage_count == 0, evicts it, and inserts the
        new page with a usage count of 1.

        Parameters
        ----------
        page : int
            Integer page ID (from ``encode_page_sets``).

        Returns
        -------
        bool
            True if the page was already cached (hit).
        """
        if page in self._lookup:
            idx = self._lookup[page]
            if self._usage[idx] < MAX_USAGE_COUNT:
                self._usage[idx] += 1
            return True

        if self.capacity == 0:
            return False

        self._insert_page(page)
        return False

    def batch_access(self, pages: frozenset[int]) -> int:
        """
        Access a batch of pages, returning the number of cache hits.

        Hit detection uses a single set intersection against the
        internal lookup dict (C-optimised in CPython), avoiding
        per-page branching for hits.  Only misses are processed
        individually through the clock-sweep eviction path.

        This models PostgreSQL's behaviour more accurately than
        element-by-element access: a query reads all its pages during
        plan execution, so the buffer manager processes them as a batch.

        Parameters
        ----------
        pages : frozenset[int]
            Set of integer page IDs accessed by a single query.

        Returns
        -------
        int
            Number of pages that were already cached (hits).
        """
        cached_keys = self._lookup.keys()
        hits = pages & cached_keys
        hit_count = len(hits)

        # Bump usage counts for hits
        for page in hits:
            idx = self._lookup[page]
            if self._usage[idx] < MAX_USAGE_COUNT:
                self._usage[idx] += 1

        # Insert misses through clock-sweep eviction
        if self.capacity > 0:
            for page in pages - hits:
                self._insert_page(page)

        return hit_count

    def resident_pages(self) -> frozenset[int]:
        """
        Return the set of pages currently resident in the cache.

        This is a snapshot of the cache state, used to compute per-query
        residuals R(Qᵢ; C) for the directional utility matrix.

        Returns
        -------
        frozenset[int]
            Integer page IDs currently held in the buffer pool.
        """
        return frozenset(self._lookup.keys())

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self._page_ids.clear()
        self._usage.clear()
        self._lookup.clear()
        self._hand = 0


# ---------------------------------------------------------------------------
# Per-query residuals and directional utility matrix
# ---------------------------------------------------------------------------


def compute_residual(
    pages: frozenset[int],
    cache_capacity_pages: int,
) -> frozenset[int]:
    """
    Pages from a query that survive a cold-cache run under clock-sweep.

    Runs *pages* through a fresh ``PageClockSweepCache`` of the given
    capacity and snapshots the resident set afterwards.  This defines
    the residual ``R(Q; C)`` used by the directional matrix:

        R(Q; C) = pages of Q still resident after Q runs alone
                  starting from a cold cache of capacity C

    Cases collapse cleanly:

    * ``|pages| ≤ C`` → ``R = pages`` (everything fits, nothing evicts)
    * ``|pages| > C`` → policy-dependent; matches what the simulator
      actually leaves behind under clock-sweep

    Because the residual is computed by the same simulator that scores
    final schedules, the directional matrix is a faithful projection of
    simulator behaviour rather than an analytical guess.

    Parameters
    ----------
    pages : frozenset[int]
        Integer-encoded page set for a single query.
    cache_capacity_pages : int
        Cache capacity in pages.

    Returns
    -------
    frozenset[int]
        Subset of *pages* still resident after a cold-cache batch access.
    """
    if cache_capacity_pages <= 0 or not pages:
        return frozenset()
    cache = PageClockSweepCache(cache_capacity_pages)
    cache.batch_access(pages)
    return cache.resident_pages()


def compute_directional_reusable_sets(
    page_sets: list[frozenset[int]],
    cache_capacity_pages: int,
) -> list[list[frozenset[int]]]:
    """
    Per-edge directional reusable page sets ``R_set[i][j] = R(Qᵢ; C) ∩ P(Qⱼ)``.

    For every ordered pair (i, j), records the *actual page IDs* that
    survive a cold cache after Qᵢ runs alone *and* are needed by Qⱼ.
    These are the pages Qⱼ could plausibly hit if it ran immediately
    after Qᵢ from a cold cache of capacity *C*.

    Storing the sets — not just their sizes — lets the windowed
    directional fitness apply exact, page-level duplicate-page
    discounting when summing contributions across multiple predecessors,
    instead of the independence estimate the symmetric variant is forced
    to use because it only has scalar overlap counts.

    Complexity
    ----------
    O(n) cold-cache runs to build residuals, plus O(n²) set
    intersections.  Memory is O(n² · |R|) where |R| ≤ C.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Per-query page sets (integer-encoded).
    cache_capacity_pages : int
        Cache capacity in pages; defines which pages survive eviction.

    Returns
    -------
    list[list[frozenset[int]]]
        Asymmetric *n* x *n* list of frozensets.  ``R_set[i][i]`` is
        always empty by convention.  ``R_set[i][j]`` and ``R_set[j][i]``
        may differ in size and contents.
    """
    n = len(page_sets)
    residuals = [
        compute_residual(ps, cache_capacity_pages) for ps in page_sets
    ]
    empty: frozenset[int] = frozenset()
    sets: list[list[frozenset[int]]] = [[empty] * n for _ in range(n)]
    for i in range(n):
        res_i = residuals[i]
        if not res_i:
            continue
        for j in range(n):
            if i == j:
                continue
            sets[i][j] = res_i & page_sets[j]
    return sets


def compute_directional_matrix(
    page_sets: list[frozenset[int]],
    cache_capacity_pages: int,
) -> list[list[int]]:
    """
    Directional pairwise utility matrix ``D[i][j] = |R(Qᵢ; C) ∩ P(Qⱼ)|``.

    Unlike the symmetric overlap matrix from ``compute_overlap_matrix``,
    this matrix is direction-aware: ``D[i][j]`` is the count of pages
    that survive running Qᵢ alone *and* are needed by Qⱼ — i.e. the
    cache hits Qⱼ would get if it ran immediately after Qᵢ from a cold
    cache.

    For a small query S and a large query L with ``|P(L)| > C``, ``D``
    correctly captures the asymmetry that ``M`` misses:

    * ``D[S][L]`` ≈ ``|P(S) ∩ P(L)|`` (all of S survives, so its
      overlap with L is available as hits)
    * ``D[L][S]`` can be much smaller (most of L evicts, only the tail
      that survived clock-sweep is available for S)

    The diagonal is set to 0 by convention to keep the matrix usable as
    edge weights in an Asymmetric TSP formulation of the scheduling
    problem.

    Computes residuals once per query, then takes set-intersection
    cardinalities pairwise.  Does *not* materialise the per-edge
    intersection sets; ``compute_directional_reusable_sets`` exists
    separately for callers that need the page IDs themselves (e.g.
    page-level analyses or alternative fitness functions).

    Complexity
    ----------
    O(n) cold-cache runs to build residuals, plus O(n²) set
    intersections.  Each residual run is one ``batch_access`` call.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Per-query page sets (integer-encoded).
    cache_capacity_pages : int
        Cache capacity in pages; defines which pages survive eviction.

    Returns
    -------
    list[list[int]]
        Asymmetric *n* x *n* directional utility matrix.  ``D[i][j]``
        and ``D[j][i]`` may differ.
    """
    n = len(page_sets)
    residuals = [
        compute_residual(ps, cache_capacity_pages) for ps in page_sets
    ]
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        res_i = residuals[i]
        if not res_i:
            continue
        for j in range(n):
            if i == j:
                continue
            matrix[i][j] = len(res_i & page_sets[j])
    return matrix


# ---------------------------------------------------------------------------
# Pairwise overlap matrix (ATSP formulation)
# ---------------------------------------------------------------------------


def compute_overlap_matrix(
    page_sets: list[frozenset[int]],
) -> list[list[int]]:
    """
    Precompute pairwise page-overlap counts between all query pairs.

    ``M[i][j]`` is the number of pages shared between query *i* and
    query *j*.  The matrix is symmetric.

    Together with ``approximate_schedule_fitness``, this reframes
    cache-aware query scheduling as a variant of the **Asymmetric
    Travelling Salesman Problem**: maximise the sum of edge weights
    (page overlaps) along the tour (schedule).

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Per-query page sets (integer-encoded).

    Returns
    -------
    list[list[int]]
        Symmetric *n* x *n* overlap matrix.
    """
    n = len(page_sets)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(page_sets[i] & page_sets[j])
            matrix[i][j] = overlap
            matrix[j][i] = overlap
    return matrix


def approximate_schedule_fitness(
    overlap_matrix: list[list[int]],
    page_counts: list[int],
    schedule: list[int],
    cache_capacity_pages: int,
) -> float:
    """
    Fast approximate fitness using pairwise overlaps with correction.

    For each query at position *k*, sums pairwise overlaps with
    preceding queries within a sliding window bounded by
    *cache_capacity_pages*.  To reduce double-counting from pages
    shared by three or more queries, each predecessor's contribution
    is discounted by its estimated overlap with already-counted
    predecessors within the target query's page set.

    The discount for predecessor *p_i* given already-counted
    predecessors *{p_1, ..., p_{i-1}}* is::

        discount_i = Σ_j  overlap[p_i][p_j] * overlap[p_j][q] / page_counts[p_j]

    This estimates the triple intersection ``|p_i ∩ p_j ∩ q|`` via
    an independence assumption, providing a much tighter bound than
    capping at ``page_counts[q]`` alone.

    Complexity
    ----------
    O(n · w²) per call, where *w* is the average window depth.  Only
    marginally slower than the uncorrected O(n · w) approach, and
    significantly faster than full clock-sweep simulation.

    Parameters
    ----------
    overlap_matrix : list[list[int]]
        Precomputed symmetric overlap matrix from ``compute_overlap_matrix``.
    page_counts : list[int]
        Number of distinct pages per query (same indexing as *overlap_matrix*).
    schedule : list[int]
        Permutation of query indices representing the execution order.
    cache_capacity_pages : int
        Cache capacity in pages, used to bound the look-back window.

    Returns
    -------
    float
        Approximate cache hit ratio in [0.0, 1.0].
    """
    n = len(schedule)
    if n == 0:
        return 0.0

    total_requests = 0
    for idx in schedule:
        total_requests += page_counts[idx]
    if total_requests == 0:
        return 0.0

    total_hits = 0
    for k in range(1, n):
        q = schedule[k]
        budget = cache_capacity_pages
        hits_for_q = 0.0
        counted_prevs: list[int] = []

        for w in range(k - 1, -1, -1):
            prev = schedule[w]
            budget -= page_counts[prev]
            if budget < 0:
                break

            incremental = overlap_matrix[prev][q]

            # Discount by estimated triple-intersection with
            # already-counted predecessors
            discount = 0.0
            for cp in counted_prevs:
                if page_counts[cp] > 0:
                    discount += (
                        overlap_matrix[prev][cp]
                        * overlap_matrix[cp][q]
                        / page_counts[cp]
                    )

            hits_for_q += max(0.0, incremental - discount)
            counted_prevs.append(prev)

        # Cap at the query's own page count
        if hits_for_q > page_counts[q]:
            hits_for_q = page_counts[q]
        total_hits += hits_for_q

    return total_hits / total_requests


def approximate_schedule_fitness_directional(
    directional_matrix: list[list[int]],
    page_counts: list[int],
    schedule: list[int],
    cache_capacity_pages: int,
) -> float:
    """
    Windowed approximate fitness using the directional utility matrix.

    Mirrors the symmetric windowed fitness exactly in shape — same
    budget-driven sliding window, same independence-estimate triple-
    intersection discount — but with the directional matrix as the
    asymmetric overlap source.  For each query *q* at position *k*,
    walks backward through predecessors while the budget holds and
    accumulates expected hits as ``D[prev][q]`` discounted by the
    estimated overlap with already-counted predecessors.

    Discount derivation
    -------------------
    The symmetric variant estimates ``|P(prev) ∩ P(cp) ∩ P(q)|`` by
    conditioning on ``P(cp)`` because it has only the symmetric
    overlap matrix to work with.  The directional matrix already
    encodes ``|R(·) ∩ P(q)|``, so the natural analogue is to
    condition on ``P(q)``:

        |R(prev) ∩ R(cp) ∩ P(q)| ≈ |R(prev) ∩ P(q)| · |R(cp) ∩ P(q)| / |P(q)|
                                = D[prev][q] · D[cp][q] / page_counts[q]

    This becomes exact when both residues are subsets of ``P(q)`` and
    conservative otherwise — the same kind of independence-estimate
    bound the symmetric variant is built on.

    Because ``D[prev][q]`` and ``page_counts[q]`` are constant in the
    inner discount loop, the per-predecessor discount factors as

        discount = D[prev][q] / page_counts[q] · Σ_cp D[cp][q]

    and the running sum ``Σ_cp D[cp][q]`` is maintained in O(1) per
    predecessor.  This collapses what would be an O(w²) discount step
    to O(w) total per query.

    Complexity
    ----------
    O(n · w) per call, where *w* is the average window depth.  Strictly
    scalar arithmetic on precomputed integers — no set operations.
    Comparable in speed to ``approximate_schedule_fitness``.

    Parameters
    ----------
    directional_matrix : list[list[int]]
        Precomputed directional utility matrix from
        ``compute_directional_matrix``.  Must be built with the same
        ``cache_capacity_pages`` the schedule will be simulated under.
    page_counts : list[int]
        Number of distinct pages per query (same indexing as
        *directional_matrix*).
    schedule : list[int]
        Permutation of query indices representing execution order.
    cache_capacity_pages : int
        Cache capacity in pages, used to bound the look-back window —
        the same role it plays in the symmetric variant.

    Returns
    -------
    float
        Approximate cache hit ratio in [0.0, 1.0].
    """
    n = len(schedule)
    if n == 0:
        return 0.0

    total_requests = 0
    for idx in schedule:
        total_requests += page_counts[idx]
    if total_requests == 0:
        return 0.0

    total_hits = 0.0
    for k in range(1, n):
        q = schedule[k]
        cap = page_counts[q]
        if cap == 0:
            continue

        budget = cache_capacity_pages
        hits_for_q = 0.0
        # Running sum Σ D[cp][q] over already-counted predecessors cp.
        # Lets us compute the discount for the next predecessor in O(1)
        # by factoring out the cp-independent terms D[prev][q] and
        # 1/cap.
        counted_d_sum = 0

        for w in range(k - 1, -1, -1):
            prev = schedule[w]
            budget -= page_counts[prev]
            if budget < 0:
                break

            d_prev = directional_matrix[prev][q]
            # discount = d_prev * counted_d_sum / cap
            #         = d_prev / cap * Σ_cp D[cp][q]
            discount = d_prev * counted_d_sum / cap
            hits_for_q += max(0.0, d_prev - discount)
            counted_d_sum += d_prev

        # Cap at the query's own page count
        if hits_for_q > cap:
            hits_for_q = cap
        total_hits += hits_for_q

    return total_hits / total_requests


# ---------------------------------------------------------------------------
# Schedule simulation (exact)
# ---------------------------------------------------------------------------


def simulate_schedule(
    profiles: list[AccessProfile],
    schedule: list[int],
    cache_capacity_pages: int,
) -> SimulationResult:
    """
    Simulate executing queries in the given order through a clock-sweep cache.

    Each query's table accesses are replayed through the cache.  A table
    access counts as a hit if the table is already cached, and as a miss
    otherwise.  The number of pages requested and served from cache are
    accumulated across all queries.

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles indexed by position.  Profile at index *i*
        corresponds to query index *i*.
    schedule : list[int]
        Permutation of ``range(len(profiles))`` specifying the execution
        order.
    cache_capacity_pages : int
        Cache capacity in 8 KB pages.

    Returns
    -------
    SimulationResult
        Aggregate page request and cache hit counts for the schedule.
    """
    cache = ClockSweepCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0

    for idx in schedule:
        profile = profiles[idx]
        for table, pages in profile.table_pages.items():
            total_requests += pages
            if cache.access(table, pages):
                total_hits += pages

    return SimulationResult(total_requests=total_requests, total_hits=total_hits)


def simulate_schedule_page_level(
    page_sets: list[frozenset[int]],
    schedule: list[int],
    cache_capacity_pages: int,
) -> SimulationResult:
    """
    Simulate executing queries through a page-level clock-sweep cache.

    Uses ``batch_access`` to detect hits via set intersection at C level,
    then processes only misses through the clock-sweep eviction path.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Per-query page sets (integer-encoded via ``encode_page_sets``).
    schedule : list[int]
        Permutation of ``range(len(page_sets))`` specifying execution order.
    cache_capacity_pages : int
        Cache capacity in pages.

    Returns
    -------
    SimulationResult
        Aggregate page request and cache hit counts.
    """
    cache = PageClockSweepCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0

    for idx in schedule:
        pages = page_sets[idx]
        total_requests += len(pages)
        total_hits += cache.batch_access(pages)

    return SimulationResult(total_requests=total_requests, total_hits=total_hits)
