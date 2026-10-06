#!/usr/bin/env bash
# Reproduce every results table and figure in the paper.
#
#   bash scripts/reproduce_paper.sh                      # Tables 1-2, Figures 3-4 (~10 min on one GPU)
#   RUN_CLOSED_LOOP=1 bash scripts/reproduce_paper.sh    # also re-run the closed-loop sweep (needs Hydra)
#
# By default no LLM API calls are made: every LLM row is served from the shipped
# response cache (LANGSENSOR_OFFLINE=1 turns a cache miss into an error). Set
# ALLOW_API=1 to permit calls, e.g. after changing a prompt or the test set.
#
# Stages whose outputs exist are skipped, so an interrupted run resumes.
# Output: $OUT/{table1,fig3,closed_loop,paper}/  (default outputs/)
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${OUT:-outputs}"
PY="${PY:-python}"
RUN_CLOSED_LOOP="${RUN_CLOSED_LOOP:-0}"
ARCHIVE="${LANGSENSOR_ARTIFACTS:-artifacts}/results"
[[ "${ALLOW_API:-0}" == "1" ]] || export LANGSENSOR_OFFLINE=1

stage() {  # stage <name> <output that marks it done> <command...>
  local name="$1" done_marker="$2"; shift 2
  if [[ -e "$done_marker" ]]; then echo "== $name: done ($done_marker)"; return; fi
  echo "== $name"; "$@"
}

# Static benchmark: Table 1 (ground-truth grounding) and Figure 3 (ambiguity sweep).
stage "Table 1" "$OUT/table1/table1_unseen.csv" $PY -m langsensor.eval.table1 --out-dir "$OUT/table1"
stage "Figure 3" "$OUT/fig3/fig3_by_ambiguity.csv" $PY -m langsensor.eval.fig3 --out-dir "$OUT/fig3"

# Closed loop: Figure 4 and Table 2. By default rendered from the archived sweep.
SWEEPS=(--out-root "$ARCHIVE/closed_loop_2026-05-14")
if [[ "$RUN_CLOSED_LOOP" == "1" ]]; then
  for predictor in none lsm llm llm_e2e; do
    stage "closed loop: $predictor" "$OUT/closed_loop/$predictor/metrics_timeseries.json" \
      $PY -m langsensor.vlmap.run configs/vlmap/closed_loop.yaml predictor=$predictor out_dir="$OUT/closed_loop"
  done
  SWEEPS+=(--out-root "$OUT/closed_loop")      # newer results win per method
fi

# Render. Archived CSVs come first so fresh results override them row by row;
# rows that are not re-run here (3D-ViSTA, Scaffolded-VLM) come from the archive.
P="$OUT/paper"
$PY -m langsensor.paper.table1 \
  --seen "$ARCHIVE/static/table1_seen.csv" --seen "$OUT/table1/table1_seen.csv" \
  --unseen "$ARCHIVE/static/table1_unseen.csv" --unseen "$OUT/table1/table1_unseen.csv" \
  --out "$P/table1.tex"
$PY -m langsensor.paper.fig3 --csv "$ARCHIVE/static/fig3_by_ambiguity.csv" --csv "$OUT/fig3/fig3_by_ambiguity.csv" --out-dir "$P"
$PY -m langsensor.paper.fig4 "${SWEEPS[@]}" --fig-dir "$P"
$PY -m langsensor.paper.table2 "${SWEEPS[@]}" --unobserved-from "$ARCHIVE/closed_loop_2026-05-14/_summary.json" \
  --out "$P/table2.tex" --csv "$P/table2.csv"

echo "Done: $P"
ls -1 "$P"
