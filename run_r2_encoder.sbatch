#!/bin/bash
# Stage I -- MentalBERT+CORN on R2, three missing-evidence policies (E0/E1/E2).
# Exact frozen hyperparameters: mental-bert-base-uncased, CORN, 5 folds, 8
# epochs, batch 16, max_length 256, seed 42. Only the dataset + policy change.
# Requires (built on CPU after R2 selection):
#   data/processed/phq8_item_dataset_r2_w3.csv           (build_r2_dataset)
#   outputs/r2_systematic/encoder/r2_evidence_status.csv (r2_encoder_eval --phase extract-status)
# Usage: sbatch run_r2_encoder.sbatch
#SBATCH --job-name=r2_enc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=outputs/r2_systematic/logs/encoder_%j.out
#SBATCH --error=outputs/r2_systematic/logs/encoder_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

DS=data/processed/phq8_item_dataset_r2_w3.csv
STATUS=outputs/r2_systematic/encoder/r2_evidence_status.csv
MODEL=mental/mental-bert-base-uncased
COMMON="--model-name $MODEL --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 --loss corn --evidence-column retrieved_context_windows_hybrid_pack"

echo "=== node: $(hostname) | date: $(date -u) | R2 encoder E0/E1/E2 ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

echo; echo "###### E0 status_quo ######"
python -m src.models.cross_validate $COMMON --dataset-path "$DS" \
    --missing-policy zero --evidence-status-csv "$STATUS" \
    --output-dir outputs/cv --tag r2_ctxm_corn_zero

echo; echo "###### E1 mask_none ######"
python -m src.models.cross_validate $COMMON --dataset-path "$DS" \
    --missing-policy mask --evidence-status-csv "$STATUS" \
    --output-dir outputs/cv --tag r2_ctxm_corn_mask

echo; echo "###### E2 drop_none ######"
python -m src.models.cross_validate $COMMON --dataset-path "$DS" \
    --missing-policy drop --evidence-status-csv "$STATUS" \
    --output-dir outputs/cv --tag r2_ctxm_corn_drop

echo; echo "=== DONE $(date -u) ==="
