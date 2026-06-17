#!/usr/bin/env bash
# Stage 9 — Mode A scorer search sweep on TPC-H (seed 42 first).
#
# Comparison the sweep produces, per cache:
#   families   : m, d, hybrid
#   consumers  : multistart_greedy (primary), beam width 4 (spot-check)
#   search modes: warmup_random_neighbor (full) vs random_only (honesty floor)
#
# The full-vs-random_only pair at equal budget is the key internal-validity
# test: does warm-up + neighbour refinement beat random weights over the
# same scorer space?  Multistart is the workhorse; beam-4 is a spot-check
# only (beam is ~5x the exact-eval load and did not win in smoke).
#
# Beam widths: 4 only.  To run a beam-width ablation later, invoke the CLI
# directly with --beam-widths; this sweep does not.
#
# Resumable: a run whose summary.json exists (matched by run-id) is skipped.
# Per-run console output -> $OUT/logs/<stem>.log
#
# Usage:
#   tmux new -s stage9
#   bash scripts/stage9_mode_a_tpch.sh 2>&1 | tee -a experiment_logs/mode_a/stage9_console.log
#
# Override via env: PA_DIR, OUT_DIR, CACHES, SEEDS, MAX_TRIALS.

set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

PY=./venv/bin/python
PA=${PA_DIR:-page_access/tpch_6gb}
OUT=${OUT_DIR:-experiment_logs/mode_a}
CACHES=${CACHES:-"102400 262144 524288"}
SEEDS=${SEEDS:-"42"}
MAX_TRIALS=${MAX_TRIALS:-120}
LOGS=$OUT/logs
mkdir -p "$LOGS"
FAIL=0

run() {  # run <run-id> <command...>
    local rid=$1
    shift
    if [ -f "$OUT/$rid/summary.json" ]; then
        echo "[skip] $rid"
        return 0
    fi
    echo "[run ] $rid  ($(date +%H:%M:%S))"
    if ! "$@" --run-id "$rid" > "$LOGS/$rid.log" 2>&1; then
        echo "[FAIL] $rid  -- see $LOGS/$rid.log"
        FAIL=1
    fi
}

modea() {  # modea <run-id> <family> <consumer> <mode> <seed> <cache> [extra...]
    local rid=$1 fam=$2 cons=$3 mode=$4 seed=$5 cache=$6
    shift 6
    run "$rid" \
        $PY -m src.bayesopt.run_mode_a \
        --workload tpch --cache-pages "$cache" --page-access-dir "$PA" \
        --model-family "$fam" --consumer "$cons" --search-mode "$mode" \
        --seed "$seed" --max-trials "$MAX_TRIALS" --output-dir "$OUT" "$@"
}

for SEED in $SEEDS; do
  for C in $CACHES; do
    for FAM in m d hybrid; do
      # Primary: multistart greedy, full search and random-only floor.
      modea "tpch_c${C}_${FAM}_msg_full_s${SEED}"   "$FAM" multistart_greedy warmup_random_neighbor "$SEED" "$C"
      modea "tpch_c${C}_${FAM}_msg_rand_s${SEED}"   "$FAM" multistart_greedy random_only            "$SEED" "$C"
      # Spot-check: beam width 4, full search and random-only floor.
      modea "tpch_c${C}_${FAM}_beam4_full_s${SEED}" "$FAM" beam warmup_random_neighbor "$SEED" "$C" --beam-widths 4
      modea "tpch_c${C}_${FAM}_beam4_rand_s${SEED}" "$FAM" beam random_only            "$SEED" "$C" --beam-widths 4
    done
  done
done

if [ "$FAIL" -eq 0 ]; then
    echo "ALL DONE"
else
    echo "DONE WITH FAILURES -- re-run after checking $LOGS"
    exit 1
fi
