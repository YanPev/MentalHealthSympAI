#!/bin/bash
# 4-way LLM evidence-quality judge over the retrieval datasets: the frozen
# hybrid-w3 baseline (R0) plus the three expanded-dictionary variants.
# Label-free (the judge never sees gold labels) -> fold-safe global run.
# Usage: sbatch run_evidence_quality.sbatch
#SBATCH --job-name=evq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-4
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

case "${SLURM_ARRAY_TASK_ID}" in
  1) DS="data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv"; TAG="r0hybw3" ;;
  2) DS="data/processed/phq8_item_dataset_expbm25_w3.csv";               TAG="expbm25" ;;
  3) DS="data/processed/phq8_item_dataset_exphyb_bm25q_w3.csv";          TAG="exphyb_bm25q" ;;
  4) DS="data/processed/phq8_item_dataset_exphyb_bothq_w3.csv";          TAG="exphyb_bothq" ;;
esac

echo "=== node: $(hostname) | date: $(date -u) | evidence judge | ${TAG} ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.evaluation.llm_evidence_quality \
  --dataset "$DS" --tag "$TAG" --n-samples 3 --temperature 0.7

echo; echo "=== DONE ${TAG} $(date -u) ==="
