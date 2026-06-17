#!/bin/bash
# Distillation Phase A: 7B teacher writes label-conditioned rationales for every
# example (~1752). Greedy; resumable (re-running continues where it stopped).
#SBATCH --job-name=distill_rat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) | Phase A teacher rationales ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.generate_rationales \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output outputs/cot/teacher_rationales_hybw3.jsonl

echo; echo "=== DONE $(date -u) ==="
