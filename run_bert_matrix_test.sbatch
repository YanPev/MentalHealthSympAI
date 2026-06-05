#!/bin/bash
#SBATCH --job-name=phq8_matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI

source .venv/bin/activate

# All models pre-cached here; run offline.
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

COMMON="--num-epochs 3 --batch-size 16 --max-length 256 --eval-split test"

run () {
    local model="$1"; local tag="$2"; local col="$3"; local cond="$4"
    echo
    echo "############ ${tag} | ${cond} (eval=test) ############"
    python -m src.models.train_transformer_classifier \
        $COMMON \
        --model-name "$model" \
        --evidence-column "$col" \
        --output-name "predictions_${tag}_${cond}_test.csv"
}

# bert-base-uncased
run "bert-base-uncased"              "bert"       "baseline_utterances"  "no_retrieval"
run "bert-base-uncased"              "bert"       "retrieved_utterances" "retrieval"

# MentalBERT
run "mental/mental-bert-base-uncased" "mentalbert" "baseline_utterances"  "no_retrieval"
run "mental/mental-bert-base-uncased" "mentalbert" "retrieved_utterances" "retrieval"

echo
echo "=== DONE $(date -u) ==="
