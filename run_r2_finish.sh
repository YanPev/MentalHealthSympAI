#!/bin/bash
# Finisher: waits for the R2(hybrid) downstream re-run, refreshes eval/cascade/
# report on the new R2, clears the "re-running" banner, and commits + pushes.
#   sbatch run_r2_finish.sbatch <ENC_JOB> <LLM_JOB>
set -uo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=.venv/bin/python
ENC="${1:-19405517}"; LLM="${2:-19405519}"
log(){ echo "[finish $(date -u +%H:%M:%S)] $*"; }

log "waiting for encoder $ENC + llm $LLM ..."
while squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -qE "^${ENC}|^${LLM}"; do sleep 120; done
log "downstream re-run finished"

$PY -m src.evaluation.r2_encoder_eval --phase metrics || log "encoder metrics FAILED"
$PY -m src.evaluation.r2_llm_eval || log "llm eval FAILED"

read EF_OOF LF_GLOB < <($PY - <<'EOF'
import json
ef=json.load(open("outputs/r2_systematic/encoder/e_final_selection.json"))["E_FINAL"]
lf=json.load(open("outputs/r2_systematic/llm/l_final_selection.json"))["L_FINAL"]
m={"R2_status_quo":"r2_ctxm_corn_zero","R2_mask_none":"r2_ctxm_corn_mask","R2_drop_none":"r2_ctxm_corn_drop"}
print(f"outputs/cv/oof_predictions_{m.get(ef,'r2_ctxm_corn_zero')}.csv outputs/r2_systematic/llm/folds_{lf}/*.csv")
EOF
)
log "E_FINAL=$EF_OOF L_FINAL=$LF_GLOB"
$PY -m src.evaluation.r2_complementarity --encoder "$EF_OOF" --llm-folds "$LF_GLOB" || log "complementarity FAILED"
$PY -m src.evaluation.r2_cascade         --encoder "$EF_OOF" --llm-folds "$LF_GLOB" || log "cascade FAILED"

# clear the re-running banner, refresh report (now consistent with new R2)
$PY -c "import json; json.dump({'rerunning':False,'new_r2':'L0_CORE/hybrid_a25/top3'}, open('outputs/r2_systematic/downstream_status.json','w'), indent=2)"
$PY -m src.evaluation.build_r2_report

# commit + push refreshed (already-tracked) result files
git add -f outputs/r2_systematic/encoder/*.csv outputs/r2_systematic/encoder/*.json \
           outputs/r2_systematic/llm/*.csv outputs/r2_systematic/llm/*.json \
           outputs/r2_systematic/analysis/*.csv outputs/r2_systematic/cascade/*.csv \
           outputs/r2_systematic/cascade/*.json outputs/r2_systematic/*.json \
           outputs/r2_systematic/*.md outputs/r2_systematic/results_report.html 2>/dev/null
git add -A docs/ 2>/dev/null
git commit -q -m "R2(hybrid) downstream refreshed on L0_CORE/hybrid_a25/top3; report finalized

Encoder/LLM/cascade re-run on the hybrid-selected R2; banner cleared.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" && log "committed" || log "nothing to commit"
git push origin models/bert-classifier 2>&1 | tail -2 && log "pushed"
log "===== finisher DONE $(date -u) ====="
