#!/bin/bash
#SBATCH --job-name=phq8_mil
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

echo
echo "############ Attention-MIL : MentalBERT, item-aware hybrid-W3 windows ############"
python -m src.models.mil_classifier \
    --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
    --model-name mental/mental-bert-base-uncased \
    --num-epochs 8 --k-folds 5 --batch-size 8 --max-length 128 --seed 42 \
    --tag mil_hybw3

echo
echo "=== DONE $(date -u) ==="
