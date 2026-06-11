#!/usr/bin/env bash
# Stage 4a — TPC-H Mode B simulator sweep + fresh incumbent baselines.
#
# Comparison set (demo paper): greedy_d, ga_m, ga_d baselines, plus the
# BO runner backends random / smac / botorch_gp at equal budgets.
#
# Matrix:
#   Baselines: caches {102400, 262144, 524288} x {greedy_d, ga_m, ga_d},
#              seed 42
#   BO:        same caches x backends {random, smac, botorch_gp} x
#              budgets {25, 50} @ seed 42 and {100, 200} @ seeds {42, 43, 44}
#
# Resumable: a run whose .summary.json already exists is skipped, so the
# script can be re-run after any interruption until it prints ALL DONE.
# Per-run console output goes to $OUT/logs/<stem>.log.
#
# Usage (from anywhere):
#   tmux new -s stage4a
#   bash scripts/stage4a_tpch.sh 2>&1 | tee -a experiment_logs/bo/stage4a_console.log
#
# PA_DIR / OUT_DIR / CACHES can be overridden via environment for reuse.

set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

PY=./venv/bin/python
PA=${PA_DIR:-page_access/tpch_6gb}
OUT=${OUT_DIR:-experiment_logs/bo}
CACHES=${CACHES:-"102400 262144 524288"}
LOGS=$OUT/logs
mkdir -p "$LOGS"
FAIL=0

run() {  # run <stem> <command...>
    local stem=$1
    shift
    if [ -f "$OUT/$stem.summary.json" ]; then
        echo "[skip] $stem"
        return 0
    fi
    echo "[run ] $stem  ($(date +%H:%M:%S))"
    if ! "$@" > "$LOGS/$stem.log" 2>&1; then
        echo "[FAIL] $stem  -- see $LOGS/$stem.log"
        FAIL=1
    fi
}

echo "== Stage 4a: incumbent baselines =="
for C in $CACHES; do
    for M in greedy_d ga_m ga_d; do
        run "tpch_c${C}_baseline_${M}_s42" \
            $PY -m src.bayesopt.run_baselines \
            --workload tpch --cache-pages "$C" --page-access-dir "$PA" \
            --methods "$M" --seed 42 --out-dir "$OUT"
    done
done

bo() {  # bo <cache> <backend> <budget> <seed>
    run "tpch_c${1}_${2}_b${3}_s${4}" \
        $PY -m src.bayesopt.run_bayesopt \
        --workload tpch --cache-pages "$1" --page-access-dir "$PA" \
        --backend "$2" --budget "$3" --seed "$4" --out-dir "$OUT"
}

echo "== Stage 4a: Mode B budget ladder =="
for C in $CACHES; do
    for B in random smac botorch_gp; do
        bo "$C" "$B" 25 42
        bo "$C" "$B" 50 42
        for S in 42 43 44; do
            bo "$C" "$B" 100 "$S"
            bo "$C" "$B" 200 "$S"
        done
    done
done

if [ "$FAIL" -eq 0 ]; then
    echo "ALL DONE"
else
    echo "DONE WITH FAILURES -- re-run this script after checking $LOGS"
    exit 1
fi
