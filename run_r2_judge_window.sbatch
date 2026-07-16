#!/bin/bash
# Stage D per-window judge (rank-level diagnostics + BM25-vs-semantic
# complementarity): MentaLLaMA-chat-7B judges each unique (participant, item,
# window) in the Top-10 of any (config, retriever) ranking. Per-window judgment
# is config-independent, so uniques are judged once. Fold-sharded 1-5.
# Usage: sbatch run_r2_judge_window.sbatch
#SBATCH --job-name=r2judge_win
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/r2_systematic/logs/judge_win_%A_%a.out
#SBATCH --error=outputs/r2_systematic/logs/judge_win_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "=== node: $(hostname) | date: $(date -u) | window judge | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.evaluation.judge_windows --mode window --fold "$FOLD" \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --n-samples 3 --temperature 0.4 --batch-size 12 --max-new-tokens 96 --top-n 10

echo; echo "=== DONE window fold $FOLD $(date -u) ==="
