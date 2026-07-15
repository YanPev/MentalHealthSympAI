#!/bin/bash
# Phase 5: MentalBERT + CORN CV on the expanded-retrieval datasets, replicating
# the frozen ctxm_corn_hybw3 baseline hyperparameters exactly (only the dataset
# changes). R1 = exphyb_bm25q (label-free selected), plus V1 = expbm25 diagnostic.
# Produces oof_predictions_ctxm_corn_{exphybw3,expbm25w3}.csv (folds identical
# to the frozen baseline, verified). Usage: sbatch run_cv_expanded.sbatch
#SBATCH --job-name=cv_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) | ctxm_corn CV on expanded retrieval ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

DP=data/processed
MODEL=mental/mental-bert-base-uncased
COMMON="--model-name $MODEL --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 --loss corn"

run () {  # tag dataset
  echo; echo "############ $1 ############"
  python -m src.models.cross_validate $COMMON \
    --dataset-path "$DP/$2" --evidence-column retrieved_context_windows_hybrid_pack \
    --tag "$1"
}

run ctxm_corn_exphybw3 phq8_item_dataset_exphyb_bm25q_w3.csv   # R1
run ctxm_corn_expbm25w3 phq8_item_dataset_expbm25_w3.csv       # V1 diagnostic

echo; echo "=== DONE $(date -u) ==="
