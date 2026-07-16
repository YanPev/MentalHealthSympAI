#!/bin/bash
# Stage G judge — set-level Qwen judgments on the hybrid rankings (L0_CORE x
# alpha {0.25,0.5,0.75}, Top-3/5/budget). Per-window judgments are config/retriever
# -independent so they are reused from the single-retriever run; only set-level
# needs re-judging for the new hybrid window combinations. Fold-sharded 1-5.
# Output: set_judgments_hybrid_fold{F}.csv. Usage: sbatch run_r2_judge_hybrid.sbatch
#SBATCH --job-name=r2judge_hyb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/r2_systematic/logs/judge_hyb_%A_%a.out
#SBATCH --error=outputs/r2_systematic/logs/judge_hyb_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "=== node: $(hostname) | $(date -u) | hybrid set judge | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.evaluation.judge_windows --mode set --fold "$FOLD" \
    --rankings outputs/r2_systematic/retrieval/retrieval_window_scores_hybrid.parquet \
    --configs L0_CORE --retrievers hybrid_a25 hybrid_a50 hybrid_a75 \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --n-samples 3 --temperature 0.4 --batch-size 12 --max-new-tokens 96 \
    --prefixes top3 top5 budget --tag "hybrid_fold${FOLD}"

echo; echo "=== DONE hybrid fold $FOLD $(date -u) ==="
