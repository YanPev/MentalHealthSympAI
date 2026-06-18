#!/bin/bash
# Downstream test of item-aware HYBRID retrieval: 5-fold CoT on hybrid retrieval
# with the bare item query vs the item-aware (expanded) query, same fusion.
# Compared against the existing W5-hybrid CoT (outputs/cot/folds_w5).
# Usage: sbatch run_cot_hybrid_itemaware.sbatch <hybridbare|hybriditemaware>
#SBATCH --job-name=cot_hia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

COND="$1"   # hybridbare | hybriditemaware
case "$COND" in
  hybridbare)      COL=retrieved_context_windows_hybridbare_list ;;
  hybriditemaware) COL=retrieved_context_windows_hybriditemaware_list ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
FOLD="${SLURM_ARRAY_TASK_ID}"
OUTDIR="outputs/cot/folds_${COND}"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | CoT $COND | fold=$FOLD col=$COL ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/processed/phq8_item_dataset_hybrid_itemaware_w5.csv \
  --evidence-column "$COL" \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "${OUTDIR}/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE $COND fold $FOLD $(date -u) ==="
