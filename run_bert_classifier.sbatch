#!/bin/bash
#SBATCH --job-name=phq8_bert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI

# Use the project venv (torch 2.8 + cu128, transformers 4.57).
source .venv/bin/activate

# Model is pre-cached here; run offline so we don't depend on compute-node internet.
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

COMMON="--model-name bert-base-uncased --num-epochs 3 --batch-size 16 --max-length 256 --eval-split validation"

echo
echo "############ CONDITION 1/2: BASELINE (no retrieval) ############"
python -m src.models.train_transformer_classifier \
    $COMMON \
    --evidence-column baseline_utterances \
    --output-name predictions_no_retrieval_bert.csv

echo
echo "############ CONDITION 2/2: RETRIEVAL ############"
python -m src.models.train_transformer_classifier \
    $COMMON \
    --evidence-column retrieved_utterances \
    --output-name predictions_retrieval_bert.csv

echo
echo "=== DONE $(date -u) ==="
