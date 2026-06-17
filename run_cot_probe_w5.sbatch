#!/bin/bash
# "Better evidence" experiment: re-run the 5-fold CoT probe with Hybrid-W5 windows
# presented as per-excerpt coherent blocks (the hybrid_list column), vs the old
# W3 keyword-blob pack. Greedy. One array task per fold -> outputs/cot/folds_w5/.
# Usage: sbatch run_cot_probe_w5.sbatch [n_samples]
#SBATCH --job-name=cot_w5
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

NSAMP="${1:-1}"
FOLD="${SLURM_ARRAY_TASK_ID}"
mkdir -p outputs/cot/folds_w5

echo "=== node: $(hostname) | date: $(date -u) | CoT W5-excerpts | fold=$FOLD n=$NSAMP ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w5.csv \
  --evidence-column retrieved_context_windows_hybrid_list \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" \
  --n-samples "$NSAMP" \
  --output "outputs/cot/folds_w5/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
