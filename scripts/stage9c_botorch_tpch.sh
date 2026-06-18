#!/usr/bin/env bash
# Stage 9c — add the BoTorch GP arm to the four targeted cells (seed 42).
#
# Equal-budget honesty-floor comparison.  For each cell, the full arm set
# across this and Stage 9b is:
#   random_only            : random weights only, no warm-up
#   warmup_random          : warm-up + random, no refinement
#   warmup_random_neighbor : warm-up + random + forced neighbour refinement
#   botorch_gp             : warm-up + Sobol init + GP/LogEI acquisition  <-- this script
#
# Same four cells as Stage 9b so the BO arm is directly comparable:
#   c102400 d  msg   | c262144 m  beam4 | c524288 m  msg | c524288 hybrid beam4
#
# botorch_gp requires the BO stack (torch CPU + botorch + gpytorch) per
# requirements-bo.txt.  Resumable; logs in $OUT/logs/.

set -u
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

PY=./venv/bin/python
PA=${PA_DIR:-page_access/tpch_6gb}
OUT=${OUT_DIR:-experiment_logs/mode_a_arms}
MAX_TRIALS=${MAX_TRIALS:-120}
LOGS=$OUT/logs
mkdir -p "$LOGS"
FAIL=0

run() {
    local rid=$1; shift
    if [ -f "$OUT/$rid/summary.json" ]; then echo "[skip] $rid"; return 0; fi
    echo "[run ] $rid  ($(date +%H:%M:%S))"
    if ! "$@" --run-id "$rid" > "$LOGS/$rid.log" 2>&1; then
        echo "[FAIL] $rid  -- see $LOGS/$rid.log"; FAIL=1
    fi
}

cell() {  # cell <cache> <family> <consumer> <tag> [extra...]
    local C=$1 FAM=$2 CONS=$3 TAG=$4; shift 4
    run "c${C}_${FAM}_${TAG}_botorch_gp_s42" \
        $PY -m src.bayesopt.run_mode_a \
        --workload tpch --cache-pages "$C" --page-access-dir "$PA" \
        --model-family "$FAM" --consumer "$CONS" --search-mode botorch_gp \
        --seed 42 --max-trials "$MAX_TRIALS" --output-dir "$OUT" "$@"
}

#    cache  family consumer            tag
cell 102400 d      multistart_greedy   msg
cell 262144 m      beam                beam4 --beam-widths 4
cell 524288 m      multistart_greedy   msg
cell 524288 hybrid beam                beam4 --beam-widths 4

if [ "$FAIL" -eq 0 ]; then echo "ALL DONE"; else echo "DONE WITH FAILURES"; exit 1; fi
