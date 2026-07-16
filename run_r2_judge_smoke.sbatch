#!/bin/bash
# Stage D smoke: validate the INDEPENDENT judge (MentaLLaMA-chat-7B, LLaMA-2
# family != Qwen predictor) end-to-end on a tiny slice -- model load, LLaMA-2
# prompt format, JSON parse rate, status distribution, throughput -- BEFORE the
# full fold-sharded arrays. Usage: sbatch run_r2_judge_smoke.sbatch
#SBATCH --job-name=r2judge_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=outputs/r2_systematic/logs/judge_smoke_%j.out
#SBATCH --error=outputs/r2_systematic/logs/judge_smoke_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== node: $(hostname) | date: $(date -u) | judge smoke (MentaLLaMA-7B) ==="
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo; echo "###### window mode, 3 participants ######"
python -m src.evaluation.judge_windows --mode window --limit 3 \
    --model-name klyang/MentaLLaMA-chat-7B \
    --n-samples 3 --temperature 0.4 --batch-size 6 --max-new-tokens 80 --tag smoke

echo; echo "###### set mode, 3 participants ######"
python -m src.evaluation.judge_windows --mode set --limit 3 \
    --model-name klyang/MentaLLaMA-chat-7B \
    --n-samples 3 --temperature 0.4 --batch-size 6 --max-new-tokens 80 --tag smoke

echo; echo "=== DONE $(date -u) ==="
