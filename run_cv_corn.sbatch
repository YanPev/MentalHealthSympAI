#!/bin/bash
#SBATCH --job-name=phq8_corn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

# CORN ordinal loss, retrieval, 8 epochs, 5-fold participant CV (matches the CE baseline setup).
COMMON="--evidence-column retrieved_utterances --loss corn --num-epochs 8 --k-folds 5 --batch-size 16 --max-length 256"

echo
echo "############ CORN CV 1/2 : BERT-base ############"
python -m src.models.cross_validate $COMMON --model-name bert-base-uncased --tag bert_corn

echo
echo "############ CORN CV 2/2 : MentalBERT ############"
python -m src.models.cross_validate $COMMON --model-name mental/mental-bert-base-uncased --tag mentalbert_corn

echo
echo "=== DONE $(date -u) ==="
