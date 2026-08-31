# Queryosity conference demo

Interactive frontend/backend for the Queryosity ICDE demonstration. The live interface mirrors the paper's three objectives:

1. **Capacity-Aware Workload Planning** — select TPC-H, TPC-DS, or JOB/IMDB and a prepared buffer capacity, then compare the actual selected **Sweep-D** and **Sweep+Beam-D** schedules and their modeled hits, misses, and `F_hit`.
2. **Interactive Schedule Exploration** — load a generated schedule or a custom order, drag/add/remove/randomize/import/export it, and rescore the edited order directly with the shared Queryosity cache simulator.
3. **Explaining Generated Schedules** — inspect capacity-specific transition metadata (`D[i,j;C]`, normalized reuse `D̂[i,j]`, regret, construction score, same-state successor scores, selected `w_r`, and beam width when applicable). Sweep+Beam-D transitions are labeled as retained beam transitions rather than assigned inferred local greedy ranks.

The conference UI contains **no live PostgreSQL schedule-execution stage** and does **not** predict runtime. PostgreSQL is used beforehand to create stored profiles. Editing a schedule never reruns Sweep-D or Sweep+Beam-D.

## Recommended location on your VM

Your VS Code screenshot shows the main repository open over SSH. Put/extract this folder directly inside it:

```text
~/Queryosity-Killed-The-Cache/
├── src/
├── page_access/
├── ...
└── queryosity-whatif-demo/
    ├── backend/
    ├── frontend/
    ├── scripts/
    └── data/
```

In that layout the backend and artifact generator automatically discover the parent Queryosity repository. If you place the demo elsewhere, set `QKC_PROJECT_ROOT=/path/to/Queryosity-Killed-The-Cache`.

## 1. Install the demo dependency

Use the same Python environment that already runs the Queryosity research code:

```bash
cd ~/Queryosity-Killed-The-Cache/queryosity-whatif-demo
python -m pip install -r backend/requirements.txt
```

The demo itself adds only Flask/pytest; real-mode scoring imports the research repository's existing Python modules and therefore relies on that environment's existing scientific dependencies.

## 2. Generate the paper-valid schedules and explanation artifacts

From the demo folder:

```bash
export PYTHONHASHSEED=0
python scripts/generate_demo_artifacts.py --all
```

This creates:

```text
data/generated/
├── tpch_102400.json
├── tpch_262144.json
├── tpch_524288.json
├── tpcds_102400.json
├── tpcds_262144.json
├── tpcds_524288.json
├── job_102400.json
├── job_262144.json
└── job_524288.json
```

The generator is an adapter around the **existing research implementation**. It imports the repository's `compute_directional_matrix`, `row_normalize`, regret scorer, deterministic start set, Sweep-D sweep, Sweep+Beam-D sweep/beam search, candidate simulation, and `ExactSimObjective`; it does not maintain a second scheduler implementation in the frontend.

To regenerate just one configuration:

```bash
python scripts/generate_demo_artifacts.py --workload tpch --capacity 262144
```

If your stored profiles are in a non-default location, either point `--project-root` at the correct repository or adjust the profile layout before generation. The default assumes `page_access/{tpch,tpcds,job}` as used by the current demo adapter.

## 3. Run the demo

Force real mode for the conference so a missing research dependency fails loudly rather than silently becoming mock data:

```bash
export QKC_SIM_BACKEND=real
python backend/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

When using VS Code Remote SSH, forward port `8000` in the **Ports** panel if VS Code does not do so automatically.

No live PostgreSQL connection is required for ordinary conference interaction.

The top bar also includes an optional **Paper light** toggle. It preserves the same layout while switching to a print-friendly light palette for tightly cropped IEEE-paper screenshots.

## API

The paper-facing endpoints are:

```text
GET  /api/workloads
GET  /api/generated-schedules?workload=tpch&cap=262144
GET  /api/explanation?workload=tpch&cap=262144&method=sweep_D
POST /api/score
POST /api/compare
POST /api/optimize
```

`POST /api/score` accepts one arbitrary edited order and performs one cache-simulator evaluation. It has no path to scheduler search.

`POST /api/optimize` is deliberately separate: it enumerates all permutations of a **small selected subset** (default maximum five queries) and explicitly reports `scope: "selected_subset_only"` and `global_optimum_claimed: false`. It is not Sweep-D, not Sweep+Beam-D, and not a full-workload optimum.

## Mock mode

For UI development without the research repository:

```bash
export QKC_SIM_BACKEND=mock
python backend/app.py
```

Mock artifacts live in `data/generated_mock/`. They are visibly labelled and contain **no fabricated transition explanations**. Objective 3 tells the user to generate real research artifacts instead.

## Tests / static checks

```bash
pytest -q backend/tests
python -m compileall backend scripts
node --check frontend/js/app.js
node --check frontend/js/whatif.js
node --check frontend/js/explain_view.js
```

## Important invariants

- Primary generated methods are only **Sweep-D** and **Sweep+Beam-D**.
- Changing capacity loads that capacity's generated schedules and explanation artifact.
- Construction score and complete-order `F_hit` are displayed as different quantities.
- Drag/reorder/add/remove goes **Frontend → `/api/score` → cache simulator → Frontend**.
- Ordinary edits never run the scheduler.
- Live PostgreSQL schedule execution is absent from the conference interface.
- The exact subset reference is explicitly demoted and scoped to the selected small subset.
