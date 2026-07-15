#!/bin/bash
# Phase 5: missing-evidence training policies (mask, drop) on the R1 dataset
# (exphyb_bm25q) with MentalBERT + CORN, same hyperparameters as the frozen
# ctxm_corn baseline. The judge's evidence_status drives which training rows
# are masked/dropped (post-split, train portion only -> folds + OOF coverage
# identical to the zero-policy baseline). zero is already covered by
# run_cv_expanded.sbatch (ctxm_corn_exphybw3). Usage: sbatch run_cv_policies.sbatch
#SBATCH --job-name=cv_pol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:30:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) | missing-policy CV on R1 (exphyb_bm25q) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

DP=data/processed
STATUS=outputs/cot/evidence_status_exphyb_bm25q.csv
MODEL=mental/mental-bert-base-uncased
COMMON="--model-name $MODEL --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 --loss corn"

run () {  # policy tag
  echo; echo "############ $2 (policy=$1) ############"
  python -m src.models.cross_validate $COMMON \
    --dataset-path "$DP/phq8_item_dataset_exphyb_bm25q_w3.csv" \
    --evidence-column retrieved_context_windows_hybrid_pack \
    --missing-policy "$1" --evidence-status-csv "$STATUS" \
    --tag "$2"
}

run mask ctxm_corn_exphybw3_mask
run drop ctxm_corn_exphybw3_drop

echo; echo "=== DONE $(date -u) ==="
