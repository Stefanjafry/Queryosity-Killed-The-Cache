#!/usr/bin/env bash
# Stage 9b — targeted three-arm comparison on 4 TPC-H cells (seed 42).
#
# Tests whether warm-up + neighbour refinement beats random weights over
# the same scorer space, now that the arms are genuinely distinct:
#   random_only            : random weights only, NO warm-up (true floor)
#   warmup_random          : warm-up grid + random, NO refinement
#   warmup_random_neighbor : warm-up + random + forced refinement pass
#
# Four cells chosen for signal, not coverage:
#   c102400 d  msg   - Mode A beat both baselines (D-family wins small cache)
#   c262144 m  beam4 - M-family beam at mid cache (best Mode A at 2GB was m_beam4)
#   c524288 m  msg   - Mode A beat both baselines (M-family wins large cache)
#   c524288 hybrid beam4 - hybrid beam at large cache (cross-check)
#
# Resumable; per-run logs in $OUT/logs/.  Each run uses a fixed --run-id
# so reruns overwrite cleanly and the arms are addressable by name.

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
        echo "[FAIL] $rid"; FAIL=1
    fi
}

cell() {  # cell <cache> <family> <consumer> <tag> [extra...]
    local C=$1 FAM=$2 CONS=$3 TAG=$4; shift 4
    for MODE in random_only warmup_random warmup_random_neighbor; do
        run "c${C}_${FAM}_${TAG}_${MODE}_s42" \
            $PY -m src.bayesopt.run_mode_a \
            --workload tpch --cache-pages "$C" --page-access-dir "$PA" \
            --model-family "$FAM" --consumer "$CONS" --search-mode "$MODE" \
            --seed 42 --max-trials "$MAX_TRIALS" --output-dir "$OUT" "$@"
    done
}

#    cache  family consumer            tag
cell 102400 d      multistart_greedy   msg
cell 262144 m      beam                beam4 --beam-widths 4
cell 524288 m      multistart_greedy   msg
cell 524288 hybrid beam                beam4 --beam-widths 4

if [ "$FAIL" -eq 0 ]; then echo "ALL DONE"; else echo "DONE WITH FAILURES"; exit 1; fi
