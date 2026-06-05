#!/bin/bash
#SBATCH --job-name=mbert_sweep
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

MODEL="mental/mental-bert-base-uncased"
SWEEP_DIR="outputs/sweep_mentalbert"

# Sweep on VALIDATION (model selection); test stays held out.
for EP in 3 5 8; do
  for LR in 1e-5 2e-5 3e-5 5e-5; do
    echo
    echo "############ MentalBERT | retrieval | epochs=${EP} lr=${LR} (eval=validation) ############"
    python -m src.models.train_transformer_classifier \
        --model-name "$MODEL" \
        --evidence-column retrieved_utterances \
        --eval-split validation \
        --num-epochs "$EP" \
        --learning-rate "$LR" \
        --batch-size 16 --max-length 256 \
        --output-dir "$SWEEP_DIR" \
        --output-name "mbert_ret_e${EP}_lr${LR}.csv"
  done
done

echo
echo "=== DONE $(date -u) ==="
