"""
Run a query scheduler against a TPC-H or TPC-DS workload.

Usage
-----
    python -m src.scheduler.run_scheduler --workload tpch
    python -m src.scheduler.run_scheduler --workload tpcds --cache-pages 2000
    python -m src.scheduler.run_scheduler --workload tpch --generations 300 --pop 150 --seed 42
    python -m src.scheduler.run_scheduler --workload tpcds --approximate

The script connects to PostgreSQL, collects EXPLAIN plans for every query in
the chosen workload, builds access profiles, runs the selected scheduling
algorithm, and prints the resulting schedule with its cache hit ratio vs. the
default (input) order.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import time

logging.basicConfig(level=logging.WARNING)

from src.postgres.connection import close_connection, create_connection
from src.scheduler.genetic_algorithm import GAScheduler
from src.scheduler.genetic_config import ApproxMode, FitnessType, GAConfig
from src.simulator.access_profile import build_access_profiles_from_db
from src.profiler.page_profiler import load_all_page_access
from src.simulator.cache_simulator import (
    encode_page_sets,
    simulate_schedule,
    simulate_schedule_page_level,
)
from src.utilities.configurations import (
    BASELINE_SEED,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, PROJECT_ROOT, WORKLOAD_DIRS
from src.utilities.workload import load_queries


def _print_schedule(
    profiles,
    schedule: list[int],
    cache_pages: int,
    label: str,
    page_sets: list[frozenset[int]] | None = None,
) -> float:
    """
    Simulate a schedule and print its fitness summary.

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles for each query.
    schedule : list[int]
        Permutation of query indices.
    cache_pages : int
        Cache capacity in pages.
    label : str
        Header label for the printed output.
    page_sets : list[frozenset[int]] | None
        Integer-encoded page sets for page-level simulation.

    Returns
    -------
    float
        Cache hit ratio for the schedule.
    """
    if page_sets is not None:
        sim = simulate_schedule_page_level(page_sets, schedule, cache_pages)
    else:
        sim = simulate_schedule(profiles, schedule, cache_pages)
    ids = [profiles[i].query_id for i in schedule]
    print(f"\n{label}")
    print(f"  Order : {' → '.join(ids)}")
    print(f"  H_total / R_total : {sim.total_hits:,} / {sim.total_requests:,}")
    print(f"  F_hit : {sim.hit_ratio:.4f}  ({sim.hit_ratio * 100:.2f}%)")
    return sim.hit_ratio


class Args(argparse.Namespace):
    workload: str
    cache_pages: int
    algorithm: str
    generations: int
    fitness: FitnessType
    pop: int
    seed: int | None
    host: str
    port: int
    user: str
    password: str
    schema: str
    timeout_ms: int
    onnx_path: Path
    approximate: bool
    approx_mode: ApproxMode
    greedy_baseline: bool


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, build access profiles, and run the scheduler.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Uses sys.argv when None.
    """
    parser = argparse.ArgumentParser(description="Cache-aware query scheduler")
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        default="tpch",
        help="Query workload to schedule (default: tpch)",
    )
    parser.add_argument(
        "--cache-pages",
        type=int,
        default=1000,
        help="Cache capacity in 8 KB pages (default: 1000)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ga"],
        default="ga",
        help="Scheduling algorithm to use (default: ga)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of GA generations (default: 200)",
    )
    parser.add_argument(
        "--fitness",
        choices=["lru", "dqn"],
        default="lru",
        help="Fitness evaluation method (default: lru)",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path(__file__).parent.parent.parent / "dqn.onnx",
        help="Fitness evaluation method (default: ./dqn.onnx)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=100,
        help="GA population size (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--approximate",
        action="store_true",
        help="Use overlap-matrix approximate fitness during GA evolution "
             "(faster, final result still uses exact simulation)",
    )
    parser.add_argument(
        "--approx-mode",
        choices=["symmetric", "directional"],
        default="symmetric",
        help="Variant of approximate fitness when --approximate is set. "
             "'symmetric' uses |P(Qi) ∩ P(Qj)|; 'directional' uses "
             "|R(Qi;C) ∩ P(Qj)|, which captures large→small vs small→large "
             "asymmetry under eviction pressure. (default: symmetric)",
    )
    parser.add_argument(
        "--greedy-baseline",
        action="store_true",
        help="Also report a greedy nearest-neighbour schedule on the "
             "directional matrix as an extra baseline. Requires "
             "page-level data; computes the directional matrix once.",
    )
    parser.add_argument("--host", default=PG_HOST)
    parser.add_argument("--port", type=int, default=PG_PORT)
    parser.add_argument("--user", default=PG_USER)
    parser.add_argument("--password", default=PG_PASSWORD)
    parser.add_argument("--schema", default=PG_SCHEMA)
    parser.add_argument("--timeout-ms", type=int, default=PG_STATEMENT_TIMEOUT_MS)
    args = parser.parse_args(argv, namespace=Args())

    db_name = DB_DEFAULTS[args.workload]

    print(f"Loading {args.workload} queries…")
    queries = load_queries(args.workload)
    print(f"  Found {len(queries)} queries: {', '.join(queries)}")

    print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
    conn = create_connection(
        db_name=db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        schema=args.schema,
        statement_timeout_ms=args.timeout_ms,
    )

    try:
        print("Building access profiles via EXPLAIN…")
        t0 = time.perf_counter()
        profiles = build_access_profiles_from_db(queries, conn, analyze=False)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s  ({len(profiles)} profiles)")

        total_pages = sum(p.total_pages for p in profiles)
        print(f"  Total estimated pages across all queries: {total_pages:,}")
        print(f"  Cache capacity: {args.cache_pages:,} pages")
        print(
            f"  Cache-to-workload ratio: "
            f"{args.cache_pages / max(total_pages, 1):.2%}"
        )

    finally:
        close_connection(conn)

    # Load page-level access data if available
    page_access_dir = PROJECT_ROOT / "page_access" / args.workload
    page_sets: list[frozenset[int]] | None = None

    if page_access_dir.is_dir() and any(page_access_dir.glob("*.csv")):
        print(f"\nLoading page access data from {page_access_dir}…")
        all_pages = load_all_page_access(page_access_dir)
        # Build page_sets aligned with profiles order
        raw_page_sets: list[set[tuple[str, int]]] = []
        for profile in profiles:
            if profile.query_id in all_pages:
                raw_page_sets.append(all_pages[profile.query_id])
            else:
                print(f"  WARNING: no page access data for {profile.query_id}, using empty set")
                raw_page_sets.append(set())
        page_sets, page_to_id = encode_page_sets(raw_page_sets)
        print(f"  Loaded page data for {sum(1 for p in page_sets if p):,} / {len(profiles)} queries")
        print(f"  Total unique pages: {len(page_to_id):,}")
        print("  Using PAGE-LEVEL simulation (integer-encoded)")
    else:
        print("\n  No page access data found — using TABLE-LEVEL simulation")
        print(f"  (Run 'python -m src.profiler.run_profiler --workload {args.workload}' to generate)")

    rng = random.Random(BASELINE_SEED)
    random_schedule = list(range(len(profiles)))
    rng.shuffle(random_schedule)
    baseline_fitness = _print_schedule(
        profiles, random_schedule, args.cache_pages, "Baseline (random order)",
        page_sets=page_sets,
    )

    # Optional: greedy nearest-neighbour on the directional matrix.
    # This is a free extra baseline; the matrix is independent of the GA
    # so success here would argue the directional signal is useful on
    # its own, regardless of the search algorithm consuming it.
    if args.greedy_baseline:
        if page_sets is None:
            print(
                "\n  --greedy-baseline requires page-level data; "
                "skipping (no page_access/ files found)."
            )
        else:
            from src.scheduler.greedy_directional import (
                greedy_directional_schedule,
            )
            from src.simulator.cache_simulator import (
                compute_directional_matrix,
            )
            print("\nBuilding directional matrix for greedy baseline…")
            t0 = time.perf_counter()
            d_matrix = compute_directional_matrix(page_sets, args.cache_pages)
            page_counts_for_greedy = [len(ps) for ps in page_sets]
            greedy_perm = greedy_directional_schedule(
                d_matrix, page_counts=page_counts_for_greedy,
            )
            print(f"  Done in {time.perf_counter() - t0:.2f}s")
            _print_schedule(
                profiles, greedy_perm, args.cache_pages,
                "Baseline (greedy directional)", page_sets=page_sets,
            )

    all_tables = sorted(list(set(t for p in profiles for t in p.table_pages)))
    max_pages = {
        t: max(p.table_pages.get(t, 0) for p in profiles)
        for t in all_tables
    }

    dqn = None
    if args.fitness == "dqn":
        # Imported lazily so that the LRU path does not require
        # onnxruntime to be installed.
        from src.simulator.dqn_simulator import DQN
        dqn = DQN(
            onnx_path=args.onnx_path,
            all_tables=all_tables,
            max_pages=max_pages,
        )

    # Build the scheduler based on --algorithm
    if args.algorithm == "ga":
        ga_config = GAConfig(
            population_size=args.pop,
            num_generations=args.generations,
            cache_capacity_pages=args.cache_pages,
            all_tables=all_tables,
            max_pages=max_pages,
            fitness_type=args.fitness,
            dqn=dqn,
            seed=args.seed,
            use_approximate_fitness=args.approximate,
            approx_mode=args.approx_mode,
        )

        if ga_config.use_approximate_fitness:
            mode = f"approximate ({ga_config.approx_mode})"
        else:
            mode = "exact"

        def on_gen(gen: int, best: float) -> None:
            if gen % 50 == 0 or gen == ga_config.num_generations - 1:
                print(f"  gen {gen:4d}  best F_hit = {best:.4f}")

        scheduler = GAScheduler(config=ga_config, on_generation=on_gen)

        print(
            f"\nRunning GA  "
            f"(pop={ga_config.population_size}, gens={ga_config.num_generations}, "
            f"fitness={mode})…"
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    t0 = time.perf_counter()
    result = scheduler.schedule(profiles, page_sets=page_sets)
    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s")

    ga_fitness = _print_schedule(
        profiles, result.best_schedule, args.cache_pages, "Best schedule",
        page_sets=page_sets,
    )

    improvement = ga_fitness - baseline_fitness
    print(f"\n{'─' * 48}")
    print(
        f"  Improvement over baseline : "
        f"{improvement:+.4f}  ({improvement * 100:+.2f}pp)"
    )
    if improvement > 0:
        print("  Scheduler found a better order.")
    elif improvement == 0:
        print("  Matched the baseline (already optimal or cache too large).")
    else:
        print(
            "  Baseline order was not beaten — consider tuning parameters "
            "or a smaller cache."
        )
    from src.visualization.serializers import dump_scheduler_data
    dump_scheduler_data(profiles, result, random_schedule, workload=args.workload)
    
    print(f"{'─' * 48}\n")


if __name__ == "__main__":
    main()
