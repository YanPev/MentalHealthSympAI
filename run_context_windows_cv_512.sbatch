#!/bin/bash
#SBATCH --job-name=phq8_ctx512
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

DP=data/processed
# max_length 512 to fit the richer context packs (W5 was 82-95% truncated at 256).
COMMON="--model-name bert-base-uncased --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 512 --seed 42 --loss corn"

run () {
  echo; echo "############ $1 (max_len=512) ############"
  python -m src.models.cross_validate $COMMON \
    --dataset-path "$DP/$2" --evidence-column "$3" --tag "$1"
}

run ctx_corn_old_512    phq8_item_dataset_full_bm25.csv                  retrieved_utterances
run ctx_corn_hybw3_512  phq8_item_dataset_context_windows_hybrid_w3.csv  retrieved_context_windows_hybrid_pack
run ctx_corn_hybw5_512  phq8_item_dataset_context_windows_hybrid_w5.csv  retrieved_context_windows_hybrid_pack
run ctx_corn_bm25w3_512 phq8_item_dataset_context_windows_bm25_w3.csv    retrieved_context_windows_bm25_pack
run ctx_corn_bm25w5_512 phq8_item_dataset_context_windows_bm25.csv       retrieved_context_windows_bm25_pack

echo; echo "=== DONE $(date -u) ==="
