#!/bin/bash
# Stage J -- Qwen2.5-7B staged-tolerant CoT SC5 on R2, five evidence conditions.
# 25-task array = 5 conditions x 5 folds (index-1: cond=(i-1)/5, fold=(i-1)%5+1).
#   L1 KEEP             r2_w3.csv                 retrieved windows unchanged
#   L2 FILTER_NONE      r2_filter_w3.csv          none-judged windows removed
#   L3 INFORMATIVE_1ST  r2_informative_first_w3   supports/against first, no none
#   L4 FALLBACK         r2_fallback_w3.csv        transcript when set none/empty
#   L5 LONG_CONTEXT     r2_w3.csv + full transcript for all 8 items
# Same prompt/mode/hparams as the frozen folds_tolerant_sc5 pool; only evidence
# changes. Requires the R2 LLM datasets (build_r2_dataset + build_r2_llm_datasets).
# Usage: sbatch run_r2_llm.sbatch
#SBATCH --job-name=r2_llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=1-25
#SBATCH --output=outputs/r2_systematic/logs/llm_%A_%a.out
#SBATCH --error=outputs/r2_systematic/logs/llm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

I=$((SLURM_ARRAY_TASK_ID - 1))
COND=$((I / 5)); FOLD=$((I % 5 + 1))
DP=data/processed
COMMON="--mode staged --tolerant --n-samples 5 --temperature 0.7 --max-new-tokens 1100 --sc-batch-size 1 --evidence-column retrieved_context_windows_hybrid_list"

case "$COND" in
  0) NAME=L1_keep;        DS=$DP/phq8_item_dataset_r2_w3.csv;                  EXTRA="" ;;
  1) NAME=L2_filter;      DS=$DP/phq8_item_dataset_r2_filter_w3.csv;           EXTRA="" ;;
  2) NAME=L3_infofirst;   DS=$DP/phq8_item_dataset_r2_informative_first_w3.csv; EXTRA="" ;;
  3) NAME=L4_fallback;    DS=$DP/phq8_item_dataset_r2_fallback_w3.csv;         EXTRA="" ;;
  4) NAME=L5_longctx;     DS=$DP/phq8_item_dataset_r2_w3.csv;                  EXTRA="--full-transcript-items 1,2,3,4,5,6,7,8 --max-transcript-tokens 8000" ;;
esac

OUTDIR=outputs/r2_systematic/llm/folds_${NAME}
mkdir -p "$OUTDIR"
echo "=== node: $(hostname) | $(date -u) | LLM $NAME fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_joint --fold "$FOLD" $COMMON $EXTRA \
    --dataset-path "$DS" \
    --output "${OUTDIR}/cot_fold${FOLD}.csv"

echo; echo "=== DONE $NAME fold $FOLD $(date -u) ==="
