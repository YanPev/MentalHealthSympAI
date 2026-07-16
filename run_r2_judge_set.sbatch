#!/bin/bash
# Stage D set-level judge (the R2-selection driver): MentaLLaMA-chat-7B judges
# the COMBINED evidence of the Top-3 / Top-5 / TOKEN_BUDGET prefixes for every
# (config, retriever, participant, item). Fold-sharded 1-5 (participant-grouped).
# Independent family from the Qwen predictor. Usage: sbatch run_r2_judge_set.sbatch
#SBATCH --job-name=r2judge_set
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/r2_systematic/logs/judge_set_%A_%a.out
#SBATCH --error=outputs/r2_systematic/logs/judge_set_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "=== node: $(hostname) | date: $(date -u) | set judge | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.evaluation.judge_windows --mode set --fold "$FOLD" \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --n-samples 3 --temperature 0.4 --batch-size 12 --max-new-tokens 96 \
    --prefixes top3 top5 budget

echo; echo "=== DONE set fold $FOLD $(date -u) ==="
