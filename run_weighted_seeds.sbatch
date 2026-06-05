#!/bin/bash
#SBATCH --job-name=phq8_seeds
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

# Best weighted config from earlier: retrieval + balanced weights, 8 epochs, lr 2e-5.
COMMON="--evidence-column retrieved_utterances --num-epochs 8 --batch-size 16 --max-length 256 --eval-split test --class-weights balanced"
SEEDS="42 123 7 2024 31"
OUTDIR="outputs/seeds_weighted"

run () {
    local model="$1"; local tag="$2"; local seed="$3"
    echo
    echo "############ ${tag} | retrieval+weighted | seed=${seed} (eval=test) ############"
    python -m src.models.train_transformer_classifier \
        $COMMON \
        --model-name "$model" \
        --seed "$seed" \
        --output-dir "$OUTDIR" \
        --output-name "${tag}_ret_weighted_s${seed}.csv"
}

for S in $SEEDS; do
    run "bert-base-uncased"               "bert"       "$S"
    run "mental/mental-bert-base-uncased" "mentalbert" "$S"
done

echo
echo "=== DONE $(date -u) ==="
