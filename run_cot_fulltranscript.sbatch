#!/bin/bash
# Full-transcript experiment: feed Qwen-7B the WHOLE interview transcript per item
# (median ~1.3k tokens, fits the 32k context) instead of the ~350-token retrieved
# windows. Tests whether the severe-class misses were retrieval misses (recoverable)
# or a true label ceiling. Same folds; compare to the retrieved-evidence CoT (folds_w5).
#SBATCH --job-name=cot_full
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

FOLD="${SLURM_ARRAY_TASK_ID}"
mkdir -p outputs/cot/folds_fulltranscript

echo "=== node: $(hostname) | date: $(date -u) | Qwen-7B FULL TRANSCRIPT | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w5.csv \
  --evidence-column transcript_text \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "outputs/cot/folds_fulltranscript/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
