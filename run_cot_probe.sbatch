#!/bin/bash
# Few-shot chain-of-thought feasibility probe (Qwen2.5-7B-Instruct).
# Runs on fold 1 of the Hybrid-W3 dataset -- the exact rows of the best encoder
# OOF (MentalBERT+CORN) -- so it is a paired comparison. Greedy by default.
# Usage: sbatch run_cot_probe.sbatch [n_samples]   (n_samples>1 = self-consistency)
#SBATCH --job-name=cot_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

NSAMP="${1:-1}"

echo "=== node: $(hostname) | date: $(date -u) | CoT probe | n_samples=$NSAMP ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold 1 \
  --n-samples "$NSAMP" \
  --output outputs/cot/cot_probe_qwen_hybw3_fold1.csv

echo; echo "=== DONE $(date -u) ==="
