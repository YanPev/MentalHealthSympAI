#!/bin/bash
#SBATCH --job-name=phq8_longtail
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

# Long-tail ordinal losses on the SAME config as the headline MentalBERT+CORN+Hybrid-W3
# (only --loss differs), so results compare directly to outputs/cv/cv_metrics_ctxm_corn_hybw3.json.
COMMON="--dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
        --evidence-column retrieved_context_windows_hybrid_pack \
        --model-name mental/mental-bert-base-uncased \
        --num-epochs 8 --k-folds 5 --batch-size 16 --max-length 256 --seed 42"

echo
echo "############ 1/2 : class-balanced CORN (severe-class) ############"
python -m src.models.cross_validate $COMMON --loss corn_balanced --tag ctxm_cornbal_hybw3

echo
echo "############ 2/2 : focal CORN (hard-example focus) ############"
python -m src.models.cross_validate $COMMON --loss corn_focal --focal-gamma 2.0 --tag ctxm_cornfocal_hybw3

echo
echo "=== DONE $(date -u) ==="
