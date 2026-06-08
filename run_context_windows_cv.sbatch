#!/bin/bash
#SBATCH --job-name=phq8_ctxcv
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

DP=data/processed
COMMON="--model-name bert-base-uncased --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42"

# args: tag  dataset  evidence_col  loss_flags
run () {
  echo; echo "############ $1 ############"
  python -m src.models.cross_validate $COMMON \
    --dataset-path "$DP/$2" --evidence-column "$3" --tag "$1" $4
}

CE="--loss cross_entropy --class-weights balanced"
CORN="--loss corn"

# Evidence conditions x {weighted-CE, CORN}. Hybrid + old baseline first (priority).
run ctx_ce_old     phq8_item_dataset_full_bm25.csv                  retrieved_utterances                  "$CE"
run ctx_corn_old   phq8_item_dataset_full_bm25.csv                  retrieved_utterances                  "$CORN"
run ctx_ce_hybw3   phq8_item_dataset_context_windows_hybrid_w3.csv  retrieved_context_windows_hybrid_pack "$CE"
run ctx_corn_hybw3 phq8_item_dataset_context_windows_hybrid_w3.csv  retrieved_context_windows_hybrid_pack "$CORN"
run ctx_ce_hybw5   phq8_item_dataset_context_windows_hybrid_w5.csv  retrieved_context_windows_hybrid_pack "$CE"
run ctx_corn_hybw5 phq8_item_dataset_context_windows_hybrid_w5.csv  retrieved_context_windows_hybrid_pack "$CORN"
run ctx_ce_bm25w3  phq8_item_dataset_context_windows_bm25_w3.csv    retrieved_context_windows_bm25_pack   "$CE"
run ctx_corn_bm25w3 phq8_item_dataset_context_windows_bm25_w3.csv   retrieved_context_windows_bm25_pack   "$CORN"
run ctx_ce_bm25w5  phq8_item_dataset_context_windows_bm25.csv       retrieved_context_windows_bm25_pack   "$CE"
run ctx_corn_bm25w5 phq8_item_dataset_context_windows_bm25.csv      retrieved_context_windows_bm25_pack   "$CORN"

echo; echo "=== DONE $(date -u) ==="
