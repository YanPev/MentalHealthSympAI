#!/bin/bash
# Distillation Phase B: fine-tune the 1.5B student per fold on teacher rationales
# and predict the held-out fold. One array task per fold -> pooled distilled OOF.
# Depends on Phase A rationales existing.
#SBATCH --job-name=distill_stu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "=== node: $(hostname) | date: $(date -u) | Phase B distill fold $FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.distill_student \
  --fold "$FOLD" \
  --rationales outputs/cot/teacher_rationales_hybw3.jsonl \
  --student-model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3

echo; echo "=== DONE fold $FOLD $(date -u) ==="
