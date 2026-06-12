#!/usr/bin/env bash
# Stage 4b — TPC-DS + JOB Mode B simulator sweep + fresh incumbent baselines.
#
# Comparison set: greedy_d, ga_m, ga_d baselines + BO backends random/smac
# (botorch_gp is the small-n arm and is TPC-H-only; SMAC RF is the
# high-dimensional workhorse per the Stage 4 decision).
#
# Matrix per workload (tpcds, job):
#   Baselines: caches {102400, 262144, 524288} x {greedy_d, ga_m, ga_d},
#              seed 42, one invocation per cache (shared D build);
#              the ga_d summary (written last) is the resume marker.
#   BO:        same caches x backends {random, smac} x
#              budgets {25, 50} @ seed 42 and {100, 200} @ seeds {42, 43, 44},
#              with --no-edge-sum (diagnostic-only D build skipped for speed).
#
# Resumable: completed runs are skipped; re-run until ALL DONE.
# Per-run output in $OUT/logs/<stem>.log.
#
# Usage:
#   tmux new -s stage4b
#   bash scripts/stage4b_tpcds_job.sh 2>&1 | tee -a experiment_logs/bo/stage4b_console.log

set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

PY=./venv/bin/python
OUT=${OUT_DIR:-experiment_logs/bo}
CACHES=${CACHES:-"102400 262144 524288"}
WORKLOADS=${WORKLOADS:-"tpcds job"}
LOGS=$OUT/logs
mkdir -p "$LOGS"
FAIL=0

run() {  # run <resume-marker-stem> <log-stem> <command...>
    local marker=$1 logstem=$2
    shift 2
    if [ -f "$OUT/$marker.summary.json" ]; then
        echo "[skip] $logstem"
        return 0
    fi
    echo "[run ] $logstem  ($(date +%H:%M:%S))"
    if ! "$@" > "$LOGS/$logstem.log" 2>&1; then
        echo "[FAIL] $logstem  -- see $LOGS/$logstem.log"
        FAIL=1
    fi
}

echo "== Stage 4b: incumbent baselines =="
for WL in $WORKLOADS; do
    PA="page_access/${WL}_6gb"
    for C in $CACHES; do
        run "${WL}_c${C}_baseline_ga_d_s42" "${WL}_c${C}_baselines_s42" \
            $PY -m src.bayesopt.run_baselines \
            --workload "$WL" --cache-pages "$C" --page-access-dir "$PA" \
            --seed 42 --out-dir "$OUT"
    done
done

bo() {  # bo <workload> <cache> <backend> <budget> <seed>
    local stem="${1}_c${2}_${3}_b${4}_s${5}"
    run "$stem" "$stem" \
        $PY -m src.bayesopt.run_bayesopt \
        --workload "$1" --cache-pages "$2" \
        --page-access-dir "page_access/${1}_6gb" \
        --backend "$3" --budget "$4" --seed "$5" \
        --no-edge-sum --out-dir "$OUT"
    }

echo "== Stage 4b: Mode B budget ladder =="
for WL in $WORKLOADS; do
    for C in $CACHES; do
        for B in random smac; do
            bo "$WL" "$C" "$B" 25 42
            bo "$WL" "$C" "$B" 50 42
            for S in 42 43 44; do
                bo "$WL" "$C" "$B" 100 "$S"
                bo "$WL" "$C" "$B" 200 "$S"
            done
        done
    done
done

if [ "$FAIL" -eq 0 ]; then
    echo "ALL DONE"
else
    echo "DONE WITH FAILURES -- re-run this script after checking $LOGS"
    exit 1
fi
