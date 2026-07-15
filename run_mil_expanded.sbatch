#!/bin/bash
# Phase 5: Attention-MIL CV on the R1 expanded-retrieval dataset (exphyb_bm25q),
# replicating the frozen mil_hybw3 baseline hyperparameters exactly.
# Produces oof_predictions_mil_exphybw3.csv. Usage: sbatch run_mil_expanded.sbatch
#SBATCH --job-name=mil_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) | Attention-MIL on R1 (exphyb_bm25q) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.models.mil_classifier \
    --dataset-path data/processed/phq8_item_dataset_exphyb_bm25q_w3.csv \
    --model-name mental/mental-bert-base-uncased \
    --num-epochs 8 --k-folds 5 --batch-size 8 --max-length 128 --seed 42 \
    --tag mil_exphybw3

echo; echo "=== DONE $(date -u) ==="
