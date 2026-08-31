# Queryosity Killed The Cache, Directional Query Scheduling

Buffer-aware query scheduling for PostgreSQL 16, extending the
"Queryosity Killed The Cache" scheduler (Dolores, Ruparelia, Di Giovanni;
EECS 6414, York University). The original system reorders a batch of OLAP
queries with a genetic algorithm over a **symmetric** pairwise page-overlap
matrix `M[i][j] = |P(Qi) ∩ P(Qj)|` so that queries sharing pages run
back-to-back and reuse each other's buffer contents.

This branch adds and evaluates a **directional utility matrix**

```
D[i][j] = |R(Qi; C) ∩ P(Qj)|
```

where `R(Qi; C)` is the set of Qi's pages that *survive* clock-sweep
eviction at cache capacity `C`. `D` is asymmetric and cache-size-aware:
it measures what a predecessor actually leaves behind for a successor,
not what the two queries share on paper. Two invariants hold by
construction: `D[i][j] ≤ M[i][j]` pointwise, with equality exactly when
no eviction occurs, so `D` collapses to `M` when the working set fits
in cache, and diverges from it under eviction pressure, which is
precisely where scheduling decisions matter.

The schedule *consumer* changes too. Instead of a GA, the directional
matrix is consumed by a deterministic **1-D regret sweep**: a single
weight `w_regret` is swept over a fixed 13-point grid
`{0.0, 0.1, …, 1.0, 1.5, 2.0}`; each grid point parameterizes a greedy
step scorer, candidate schedules are built by multistart-greedy (K=4
starts) and beam search (widths 2/3/4), and every candidate is scored by
the exact clock-sweep simulator. The best candidate wins. There is no
training, no tuning of the search itself, and no seed dependence, the
same inputs always produce the same schedule.

**Methods compared in the paper** (all re-scored by the exact simulator):

| Method | Matrix | Consumer |
|---|---|---|
| `GA_M` | symmetric `M`, windowed fitness | genetic algorithm (Queryosity baseline) |
| `GA_D` | directional `D`, single-step edge-sum fitness | genetic algorithm |
| `sweep_D` | directional `D` | regret sweep → multistart-greedy (K=4) |
| `sweep_beam_D` | directional `D` | regret sweep → beam search (widths 2/3/4) |

`GA_M → GA_D` isolates the matrix (same consumer); `GA_D → sweep_*`
isolates the consumer (same matrix).

> **Archive branch.** Earlier experimental code, Bayesian-optimization
> scorer tuning (SMAC / BoTorch), the Mode A structured search, residual
> and windowing ablations, and their runners and tests, is preserved
> unchanged on the branch this one was cut from
> (`archive/experiments`). BO was retired after the tuned auxiliary
> weights collapsed to zero in 8/9 configurations, leaving `w_regret` as
> the only active dimension; the exhaustive sweep matches BO within
> measurement noise while being deterministic. This branch contains the research implementation and interactive conference demo used by the demo paper.

---

## Pipeline Overview

```
            ┌───────────────────────┐
            │ PostgreSQL 16         │  Docker (§2) or native (§2.4)
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │ Benchmark loaders     │  TPC-H · TPC-DS · JOB/IMDB (§3)
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │ Page profiler         │  pg_buffercache → CSV (§4)
            │ (src.profiler)        │  ── profiles for all three workloads
            └──────────┬────────────┘     are ALREADY SHIPPED in page_access/
                       │
            ┌──────────▼────────────┐
            │ M and D matrices +    │  simulator-only; no database needed
            │ schedulers (§5)       │  run_regret_sweep · run_baselines
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │ export_schedules (§6) │  frozen orderings → JSON
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │ Wall-clock runs (§7)  │  run_sweep on real Postgres
            └──────────┬────────────┘
                       │
            ┌──────────▼────────────┐
            │ Plots (§9)            │
            └───────────────────────┘
```

The top half (database, loaders, profiler) is only needed for wall-clock
measurements or re-profiling. **Everything in the simulated-results path
runs from the shipped profiles with no database at all**.

## Interactive Conference Demo

The `queryosity-whatif-demo/` directory contains the interactive conference
demo for Queryosity. It exposes the directional scheduling pipeline through
three paper-facing views while keeping scheduling and simulation in the
research backend.

### Capacity-Aware Workload Planning

Users select TPC-H, TPC-DS, or JOB/IMDB and a prepared buffer capacity
(102,400 / 262,144 / 524,288 8-KB pages, approximately 800 MB / 2 GB / 4 GB).
The interface presents the generated Sweep-D and Sweep+Beam-D schedules with
their modeled cache hits, modeled misses, and complete-order cache-hit fraction
`F_hit`.

### Interactive Schedule Exploration

A generated schedule can be loaded into an editable workspace and reordered,
extended, shortened, randomized, imported, or exported. Each edited order is
sent directly to the deterministic page-level clock-sweep simulator for one
complete-order rescore.

Editing does not rerun Sweep-D or Sweep+Beam-D. The interface therefore
supports interactive what-if exploration without invoking scheduler search.

### Explaining Generated Schedules

For Sweep-D and Sweep+Beam-D, the explanation view exposes the
capacity-conditioned directional reuse signal `D[i,j;C]`, row-normalized reuse
`D_hat[i,j]`, regret, construction score, selected regret weight, candidate
successors, and beam width where applicable.

Sweep-D greedily extends one partial schedule using the regret-aware
construction score. Sweep+Beam-D instead keeps multiple partial schedules,
ranks them by cumulative construction score, and prunes back to the selected
beam width at each depth. Completed candidate schedules are then scored by the
page-level cache simulator.

### Running the Demo

From the repository root:

```bash
cd queryosity-whatif-demo
python -m pip install -r backend/requirements.txt

export PYTHONHASHSEED=0

# Generate paper-valid schedules and explanation artifacts when required.
python scripts/generate_demo_artifacts.py --all

# Run against the real Queryosity implementation.
export QKC_SIM_BACKEND=real
python backend/app.py

---

## 1. Quickstart, no database required

The per-query page-access profiles for all three workloads are committed
under `page_access/` (TPC-H SF10, TPC-DS SF10, JOB/IMDB, profiling
provenance per workload in §4). That means the full simulated-F_hit
comparison is reproducible on a laptop in minutes:

```bash
git clone -b demo-paper https://github.com/Stefanjafry/Queryosity-Killed-The-Cache.git
cd Queryosity-Killed-The-Cache
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export PYTHONHASHSEED=0
python -m src.bayesopt.run_regret_sweep \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch
```

`--cache-pages 102400` is 800 MB of 8 KB pages: one of the nine
(workload × cache) configurations in the paper grid. The run sweeps
`w_regret` over the 13-point grid for both consumers, prints the best
F_hit per (matrix, consumer) arm with greedy references, and writes a
JSON summary to `experiment_logs/regret_sweep/`. Expect a few minutes;
larger workloads (TPC-DS: 93 queries, JOB: 113) take proportionally
longer.

The GA baselines at the same configuration (this is the slow part , 
population 100 × 200 generations, exact-sim final scoring):

```bash
python -m src.bayesopt.run_baselines \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch
```

Always set `PYTHONHASHSEED=0`. The sweep is seed-invariant by
construction, but the GA is seeded, and hash-order stability keeps every
run byte-reproducible. To reproduce the full nine-configuration table
(every method at every workload × cache) in one command, see §5.4.

---

## 2. PostgreSQL setup (wall-clock measurements only)

Two supported paths. The **native** path is what produced the paper's
wall-clock numbers; Docker is the low-friction way to get started.

### 2.1 Docker

```bash
docker compose up -d
docker ps   # container: query_scheduler_pg
```

Defaults: `host=localhost`, `port=5432`, `user=postgres`,
`password=postgres`. Cache flushes are performed with
`docker restart query_scheduler_pg` (the tools' default).

### 2.2 Python environment

Tested on Python 3.12 (3.10+ required).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Required Postgres settings

Two settings are load-bearing for every measurement in this project:

```sql
ALTER SYSTEM SET enable_seqscan = off;
ALTER SYSTEM SET max_parallel_workers_per_gather = 0;
SELECT pg_reload_conf();
```

`enable_seqscan = off` prevents PostgreSQL's 256 KB ring-buffer
optimization from routing large sequential scans around the shared
buffer pool, with the ring buffer active, pages loaded by one query are
invisible to the next and the premise of cross-query reuse collapses.
Disabling parallel gather keeps page-access profiles deterministic.
`shared_buffers` is varied per experiment (see §7).

### 2.4 Native PostgreSQL (paper configuration)

The wall-clock results were produced on a native PostgreSQL 16 install
(RHEL), not Docker. Every tool that flushes the buffer cache accepts
`--flush-cmd`; for native installs use:

```
--flush-cmd "sudo systemctl restart postgresql-16"
```

(`src.experiment.run_sweep` already defaults to this command; the
profiler and executor default to Docker restart and need the flag.)
Two operational notes that will save you failed multi-hour runs:

- Add a NOPASSWD sudoers rule for exactly that restart command (e.g. in
  `/etc/sudoers.d/pg_restart`). sudo's cached password expires mid-run
  otherwise, and the sweep dies hours in.
- Never run `psql`, `sudo`, or `systemctl` against the instance from
  another terminal while a sweep is running, the restart kills the
  sweep's connection and contaminates the run.

---

## 3. Installing benchmarks

Three workloads: **TPC-H SF10** (22 queries), **TPC-DS SF10** (93
queries after exclusions; the excluded ten are kept in
`workloads/tpcds_excluded/`), and **JOB** on IMDB (113 queries). Load
only what you intend to measure, the shipped profiles already cover
all three for simulation.

> Setup scripts were tested on RHEL. Data directories are expected
> **outside** the repo (e.g. `../tpch-data-sf10/`).

### 3.1 TPC-H

```bash
git clone https://github.com/gregrahn/tpch-kit
cd tpch-kit/dbgen && make && ./dbgen -s 10
mkdir ../../tpch-data-sf10 && mv *.tbl ../../tpch-data-sf10
```

```bash
cd tpch_scripts
chmod +x *.sh
export CONTAINER_NAME=query_scheduler_pg POSTGRES_USER=postgres \
       DB_NAME=tpch DATA_DIR=../../tpch-data-sf10
./setup_tpch.sh
```

Verify: `SELECT COUNT(*) FROM lineitem;` ≈ 60 M rows at SF10.
Queries live in `workloads/tpch/`.

### 3.2 TPC-DS

Linux `dsdgen`/`dsqgen` binaries are pre-built in
`tpcds_scripts/LINUX/`; other platforms build from
[tpcds-kit](https://github.com/gregrahn/tpcds-kit) (on RHEL/GCC 11 add
`-fcommon` to CFLAGS).

```bash
cd tpcds_scripts
./run_dsdgen_parallel.sh    # edit scale factor / output dir inside
./setup_tpcds.sh
```

### 3.3 JOB / IMDB

Download the JOB authors' `imdb.tgz`, extract outside the repo, then:

```bash
cd job_scripts
chmod +x *.sh
export CONTAINER_NAME=query_scheduler_pg POSTGRES_USER=postgres \
       DB_NAME=imdb DATA_DIR=/absolute/path/to/imdb-data
./setup_job.sh
```

---

## 4. Page-level profiling

Profiles are the `(table, block)` page sets each query touches,
captured from `pg_buffercache` after running the query against a cold
buffer. **You do not need to run this** unless the data, schema, or
Postgres configuration changes, `page_access/{tpch,tpcds,job}/` ship
in the repo.

```bash
python -m src.profiler.run_profiler --workload tpch
# native install:
python -m src.profiler.run_profiler --workload tpch \
    --flush-cmd "sudo systemctl restart postgresql-16"
```

Flags: `--workload`, `--output-dir` (default
`page_access/<workload>/`; use a distinct dir when re-profiling at a
non-default `shared_buffers`), `--container`, `--flush-cmd`, plus the
standard connection overrides (`--host --port --user --password
--schema --timeout-ms`). Profiling restarts Postgres before every query
,  TPC-DS takes hours. Do it once.

**Known measurement bound.** `pg_buffercache` only reports pages
resident in `shared_buffers`, so a profile taken at buffer size S caps
every query's recorded page set at S pages. A profile capped at or
below a simulated cache size distorts results at that cache: clamped
queries cannot fill the simulated buffer. Provenance of the shipped
profiles:

- **TPC-H**: profiled at 6 GB (cap ≈786 K pages). The four largest
  `lineitem` queries (`q1`, `q3`, `q6`, `q7`) reach that cap, so their
  recorded footprints are floors, not exact values. The cap sits above
  every simulated cache in the paper grid, so no configuration is
  distorted; relative method comparisons consume identical profiles
  either way.
- **TPC-DS**: profiled at 6 GB (cap ≈786 K pages), above every
  simulated cache in the paper grid, so no configuration is distorted;
  any query reaching the cap is recorded as a floor.
- **JOB**, no cap detected; the largest recorded footprints are
  genuine query sizes.

`src.bayesopt.check_truncation` audits any profile directory for cap
pile-up and reports a per-cache verdict, run it after any
re-profiling, and treat a `CORRUPT_IN_RANGE` verdict as disqualifying
for every cache size at or above the detected cap:

```bash
python -m src.bayesopt.check_truncation \
    --workload tpch --page-access-dir page_access/tpch
```

---

## 5. Schedulers (simulator-only)

### 5.1 Regret sweep, the production scheduler

```bash
python -m src.bayesopt.run_regret_sweep \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch
```

| Flag | Default | Description |
|---|---|---|
| `--workload` | required | `tpch`, `tpcds`, `job` |
| `--cache-pages` | required | simulated buffer size in 8 KB pages |
| `--page-access-dir` | required | profile directory |
| `--exclude` | `""` | comma-separated query IDs to drop |
| `--regret-grid` | 13-point default | override the `w_regret` grid |
| `--beam-widths` | `2,3,4` | beam widths (width 1 = greedy; excluded) |
| `--seed` | `42` | logging parity only; results are seed-invariant |
| `--workers` | `1` | parallel processes for exact-sim selection (1 = serial, 0 = all cores); selection only. The schedule is identical at any worker count |
| `--out-dir` | `experiment_logs/regret_sweep` | JSON summary destination |

The runner sweeps both the M and D matrices under both consumers and
prints greedy references, the M arms and greedy rows are diagnostics;
the paper methods are the D arms.

### 5.2 GA baselines

```bash
python -m src.bayesopt.run_baselines \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch
```

Regenerates `greedy_d`, `ga_m`, `ga_d` on the exact simulator. GA
configuration (the reference implementation's defaults, used for every
GA number in the paper): population 100, 200 generations, tournament
size 3, swap-mutation rate 0.3, crossover rate 0.9, elitism 2. `GA_M`
uses the windowed triple-intersection fitness from the original paper;
`GA_D` uses single-step edge-sum fitness on `D` (the windowed form is
homogeneous in matrix entries and cancels `D`'s per-row scaling, which
is why the directional GA does not use it).

### 5.3 Scheduling-cost benchmark

```bash
python -m src.bayesopt.bench_schedule_time \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch
```

Times schedule *construction* for the sweep's consumers against the
GA's evolutionary search, database-free and confound-free
(`--reps`, `--ga-pop`, `--ga-gen`, `--num-starts` to vary). Scope
caveat, stated plainly: this measures search cost only. The sweep's
production selection exact-simulates all ~52 candidates, so the sweep
does not win on total time including selection, both methods also pay
one shared exact-sim validation at the end, matching the original
paper's Figure 15 framing. The end-to-end version of this comparison,
including selection cost, is §5.5.

### 5.4 Reproduce the paper grid (one command)

```bash
python -m src.bayesopt.sim_hit_grid
```

Runs every method at all nine (workload × cache) configurations , 
caches 102400 / 262144 / 524288 pages (800 MB / 2 GB / 4 GB), prints
the F_hit table with a per-cell directional-vs-GA_M verdict, and writes
`experiment_logs/sim_hit_grid.csv`. All parameters are pinned in the
script (GA 100×200 at seed 42; 13-point regret grid; beam widths
2/3/4). Per-config query exclusions mirror the wall-clock study so the
two results axes share query sets; the sets differ across cache sizes
within a workload, so cross-cache trends inherit that caveat, while
per-cell method comparisons are unaffected (all methods share the
set). Set the `QKC_WORKERS` environment variable to parallelize
exact-simulation selection (`QKC_WORKERS=4`; 0 = all cores); the
selected schedules and F_hit values are identical at any worker count.
The CSV additionally records `sims_sweep_D` and `sims_sweep_beam_D`,
the distinct simulations per consumer after deduplication. The 18 GA
runs dominate the cost, roughly an hour serial, or ~20 minutes at
`QKC_WORKERS=4`.

### 5.5 End-to-end scheduling cost

```bash
python -m src.bayesopt.bench_end_to_end \
    --workload tpch --cache-pages 262144 \
    --page-access-dir page_access/tpch --reps 3
```

The companion to §5.3's search-cost benchmark, answering the
total-cost question: Q1 times GA_M (search + one exact-sim validation)
against the sweep as implemented (construction + exact-sim scoring of
every candidate), the sweep loses this comparison, which is the basis
for the scope caveat in §5.3. Q2 checks whether ranking candidates by
the cheap step-score and exact-simming only the winner ("Framing B")
selects the same schedule as exact-sim-all, it does not in general,
which is why that shortcut was retired. Q3 prices in GA hyperparameter
tuning (N trial searches) against the sweep's zero tuning knobs.

### 5.6 Upstream GA scheduler CLI

`src.scheduler.run_scheduler` is the original Queryosity entry point
(GA over the clock-sweep simulator, `--approx-mode
{symmetric,directional}`, optional DQN surrogate fitness). It is kept
working for continuity with the upstream README; the paper pipeline
uses §5.1–5.3.

---

## 6. Exporting schedules for wall-clock runs

`export_schedules` regenerates the chosen methods' orderings for one
(workload, cache), deterministic under the pinned seed, and writes
the JSON that `run_sweep` consumes:

```bash
python -m src.bayesopt.export_schedules \
    --workload tpch --cache-pages 102400 \
    --page-access-dir page_access/tpch \
    --out schedules/tpch_102400.json
```

| Flag | Default | Description |
|---|---|---|
| `--methods` | `GA_M,GA_D,sweep_D,sweep_beam_D` | any subset of `random,GA_M,GA_D,sweep_D,sweep_beam_D` |
| `--num-starts` | `4` | multistart K for `sweep_D` |
| `--regret-grid` / `--sweep-beam-widths` | 13-point / `2,3,4` | sweep parameters |
| `--ga-pop` / `--ga-gen` | `100` / `200` | GA size |
| `--seed` | `42` | GA seed (sweep methods are seed-invariant) |
| `--exclude` | `""` | drop query IDs before scheduling |

Output format: `{"cache_label": "<cache>", "schedules": {"<method>":
"q1,q3,…"}}`.

---

## 7. Wall-clock measurement

```bash
python -m src.experiment.run_sweep \
    --workload tpch \
    --schedules-file schedules/tpch_102400.json \
    --reps 3 \
    --out results/tpch_102400.csv
```

Before each rep, `shared_buffers` is flushed via `--flush-cmd`
(default: `sudo systemctl restart postgresql-16`). Set
`shared_buffers` in `postgresql.conf` to match the simulated cache of
the schedules file (800 MB ↔ 102400 pages), restart, then run.

Protocol notes required to interpret (or reproduce) the numbers
honestly:

- **Buffer-cold, OS-warm.** The default flush restarts Postgres, which
  empties `shared_buffers` but leaves the OS page cache warm. Reads
  and hit ratios are directly comparable to the original paper's
  protocol; absolute wall-clock times are optimistic versus true cold
  disk. `--drop-os-cache` (default cmd `sudo sysctl -w
  vm.drop_caches=3`) gives true-cold numbers, but it changes *every*
  number, so it is all-or-nothing across a comparison. Expect rep 1 to
  run slower than reps 2–3 under the OS-warm protocol as the page
  cache warms; that is noise, not a method effect.
- **Exclusions must be uniform.** If a query is excluded (timeout or
  pathology), exclude it for *every* method at that configuration, or
  the query sets differ and the comparison is contaminated. For
  cache-trend comparisons, keep exclusions uniform per workload across
  cache sizes as well.
- **Known pathological queries.** TPC-H `q5` is computation-bound
  (17–21 min at every cache size; scheduling cannot help it) and is
  excluded. `q13` is I/O-sensitive (9.5 min at 800 MB, ~1 min at
  larger caches) and is included under a 15-minute statement timeout
  (`--timeout-ms 900000`). TPC-DS `query17/25/29/23a/23b` are known
  runaways.

Plot results with:

```bash
python -m src.experiment.plot_sweep --results results/tpch_102400.csv \
    --out-prefix plots/tpch_102400
```

For single-order runs and side-by-side comparisons against file order,
`src.executor.run_executor` (`--order q5,q12,…`, `--compare-baseline`,
`--flush-cmd`) measures real shared-buffer hits and reads via
`EXPLAIN (ANALYZE, BUFFERS)`.

---

## 8. Tests and type checking

```bash
python -m pytest tests/ -v
./venv/bin/pyright --pythonpath ./venv/bin/python src/ tests/
```

The suite (86 tests) covers the simulator, both matrices and their
invariants, the step scorer and consumers, the GA, profile loading, and
the truncation checker. Pyright (pinned 1.1.408 in requirements) is
clean on `src/` and `tests/`; run it before every commit.

---

## 9. Visualizations

```bash
python -m src.visualization.run_visualizations --workload tpch --scheduler
python -m src.visualization.run_visualizations --workload tpch --executor
```

Consumes the JSON artifacts in `viz_data/` written by the scheduler and
executor runs; writes PNGs to `plots/`. `src/visualization/
directional_matrix.py` renders the M-vs-D comparison heatmap.

---

## 10. SmartQueue / DQN baseline (upstream, unchanged)

`ml/` contains the Deep-Q-Network baseline inherited from the original
project (a reimplementation of SmartQueue), kept as-is for comparison
and continuity: `dqn.ipynb` (jupytext-paired with `dqn.notebook.py`)
and `dqntrainer.py`. It requires PyTorch and the page profiles; see the
notebook's prompts. The DQN hook in `run_scheduler` (`--fitness dqn`)
remains experimental scaffolding, as in the upstream README.

---

## 11. Repository layout

```
src/
├── bayesopt/         profile loader, exact objective, step scorer +
│                     consumers, regret sweep, sim_hit_grid (paper
│                     table), GA baselines, export_schedules,
│                     bench_schedule_time, bench_end_to_end,
│                     check_truncation
├── scheduler/        GA (genetic_algorithm, genetic_config, greedy_directional)
├── simulator/        clock-sweep cache simulator, M and D matrices
├── experiment/       wall-clock sweep runner + plotting
├── executor/         real-DB executor (EXPLAIN ANALYZE, BUFFERS)
├── profiler/         pg_buffercache page-access profiler
├── utilities/        constants, configuration, workload loader
└── visualization/    plotting modules
page_access/          shipped page profiles (tpch, tpcds, job); see §4
                      for per-workload profiling provenance
workloads/            SQL query sets (+ tpcds_excluded/)
tpch_scripts/ tpcds_scripts/ job_scripts/   database setup
ml/                   upstream DQN baseline
queryosity-whatif-demo/  interactive conference demo frontend/backend,
                        artifact generator, schedule exploration, and
                        explanation interface
tests/                pytest suite (86 tests)
```

The directional matrix implementation lives in
`src/simulator/cache_simulator.py` (`compute_residual`,
`compute_directional_matrix`); the step scorer and both consumers in
`src/bayesopt/step_scorer.py`; the sweep in
`src/bayesopt/run_regret_sweep.py`.

---

## Credits

Original system and paper: Rafael Dolores, Mahnsi Ruparelia, Daniel
Di Giovanni, *Queryosity Killed the Cache: Scheduling Queries in
Relational DBMS* (EECS 6414, York University). Directional extension:
Stefan Jafry (BSc, York University; advisor Rafael Dolores).
