#!/bin/bash
# Autonomous R2 downstream driver — runs the entire remaining pipeline unattended
# on the cluster after the evidence judge completes. Launched with nohup so it
# survives client disconnection AND does not depend on the agent being re-invoked.
#
#   nohup bash run_r2_pipeline.sh <WIN_JOBID> <SET_JOBID> >> outputs/r2_systematic/logs/pipeline.log 2>&1 &
#
# All selection decisions are made by the (label-free) code; this only chains the
# stages and submits/waits on the GPU jobs. Progress + failures -> pipeline.log.

set -uo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

WIN_JOB="${1:-19395515}"
SET_JOB="${2:-19395516}"
PY=.venv/bin/python

log(){ echo "[pipeline $(date -u +%H:%M:%S)] $*"; }
step(){ log "RUN: $*"; "$@" && log "OK: $1" || { log "FAILED: $* (exit $?)"; return 1; }; }

wait_jobs(){ # $1 = egrep pattern of job-id prefixes
  local pat="$1" n
  while :; do
    n=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -cE "$pat")
    [ "$n" -eq 0 ] && break
    sleep 120
  done
}

log "===== R2 pipeline driver START (win=$WIN_JOB set=$SET_JOB) ====="

# ---- Phase 1: wait for the judge ------------------------------------------
log "[1] waiting for judge arrays to finish..."
wait_jobs "^${WIN_JOB}|^${SET_JOB}"
NSET=$(ls outputs/r2_systematic/judge/set_judgments_fold*.csv 2>/dev/null | wc -l)
NWIN=$(ls outputs/r2_systematic/judge/window_judgments_fold*.csv 2>/dev/null | wc -l)
log "[1] judge done: $NSET set-fold CSVs, $NWIN window-fold CSVs"
[ "$NSET" -ge 5 ] || { log "ABORT: expected 5 set-fold CSVs, got $NSET"; exit 1; }

# ---- Phase 2: label-free R2 selection -------------------------------------
step $PY -m src.evaluation.r2_retrieval_metrics || exit 1
$PY - <<'EOF'
import json; s=json.load(open("outputs/r2_systematic/retrieval/r2_selection.json"))["selected_R2"]
print("[pipeline] SELECTED R2:", s)
EOF

# ---- Phase 3: build R2 dataset + evidence status + missing-policy audit ----
step $PY -m src.retrieval.build_r2_dataset --out-tag r2 || exit 1
step $PY -m src.evaluation.r2_encoder_eval --phase extract-status || exit 1
step $PY -m src.evaluation.r2_encoder_eval --phase audit || true

# ---- Phase 4: LLM evidence-policy datasets (filter / infofirst / fallback) --
step $PY -m src.retrieval.build_r2_llm_datasets || true

# ---- Phase 5: submit encoder (E0/E1/E2) + LLM (L1-L5), wait ----------------
ENC=$(sbatch --parsable run_r2_encoder.sbatch); log "[5] encoder job $ENC"
LLM=$(sbatch --parsable run_r2_llm.sbatch);     log "[5] llm array $LLM"
sleep 20
wait_jobs "^${ENC}|^${LLM}"
log "[5] training finished"

# ---- Phase 6: encoder + LLM metrics ---------------------------------------
step $PY -m src.evaluation.r2_encoder_eval --phase metrics || true
step $PY -m src.evaluation.r2_llm_eval || true

# ---- Phase 7: resolve E_FINAL / L_FINAL OOF paths -------------------------
read EF_OOF LF_GLOB < <($PY - <<'EOF'
import json, glob
ef=json.load(open("outputs/r2_systematic/encoder/e_final_selection.json"))["E_FINAL"]
lf=json.load(open("outputs/r2_systematic/llm/l_final_selection.json"))["L_FINAL"]
enc_map={"R2_status_quo":"r2_ctxm_corn_zero","R2_mask_none":"r2_ctxm_corn_mask","R2_drop_none":"r2_ctxm_corn_drop"}
print(f"outputs/cv/oof_predictions_{enc_map.get(ef,'r2_ctxm_corn_drop')}.csv "
      f"outputs/r2_systematic/llm/folds_{lf}/*.csv")
EOF
)
log "[7] E_FINAL OOF=$EF_OOF | L_FINAL folds=$LF_GLOB"

# ---- Phase 8: complementarity + leakage-safe cascades ---------------------
step $PY -m src.evaluation.r2_complementarity --encoder "$EF_OOF" --llm-folds "$LF_GLOB" || true
step $PY -m src.evaluation.r2_cascade         --encoder "$EF_OOF" --llm-folds "$LF_GLOB" || true

# ---- Phase 9: assemble report + manifest + status -------------------------
step $PY -m src.evaluation.build_r2_report || true

# ---- Phase 10: commit -----------------------------------------------------
git add -A
git commit -m "R2 corrective experiment: judge, selection, encoder/LLM, cascade, report

Judge=Qwen2.5-7B (constraint 11 relaxed per user; limitation documented).
Autonomous driver run.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" && log "[10] committed" || log "[10] nothing to commit / commit failed"

log "===== R2 pipeline driver DONE $(date -u) ====="
