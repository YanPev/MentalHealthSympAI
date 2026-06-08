#!/bin/bash
# Usage: sbatch run_mbert_context.sbatch <ce|corn|ceplain> <max_length>
# Runs MentalBERT on all 5 evidence conditions for one loss/length, mirroring the
# BERT context-window matrix (ctx_*). Tags: ctxm_<loss>_<cond>[_512].
#SBATCH --job-name=mbert_ctx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

LOSS_TAG="$1"; ML="$2"
case "$LOSS_TAG" in
  ce)      FLAGS="--loss cross_entropy --class-weights balanced" ;;
  ceplain) FLAGS="--loss cross_entropy --class-weights none" ;;
  corn)    FLAGS="--loss corn" ;;
  *) echo "bad loss tag $LOSS_TAG"; exit 1 ;;
esac
SUF=""; [ "$ML" = "512" ] && SUF="_512"

echo "=== node: $(hostname) | date: $(date -u) | MentalBERT | loss=$LOSS_TAG ml=$ML ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

DP=data/processed
MODEL=mental/mental-bert-base-uncased
COMMON="--model-name $MODEL --k-folds 5 --num-epochs 8 --batch-size 16 --max-length $ML --seed 42 $FLAGS"

run () {  # cond dataset evidence_col
  echo; echo "############ ctxm_${LOSS_TAG}_${1}${SUF} ############"
  python -m src.models.cross_validate $COMMON --dataset-path "$DP/$2" --evidence-column "$3" \
    --tag "ctxm_${LOSS_TAG}_${1}${SUF}"
}

run old    phq8_item_dataset_full_bm25.csv                  retrieved_utterances
run hybw3  phq8_item_dataset_context_windows_hybrid_w3.csv  retrieved_context_windows_hybrid_pack
run hybw5  phq8_item_dataset_context_windows_hybrid_w5.csv  retrieved_context_windows_hybrid_pack
run bm25w3 phq8_item_dataset_context_windows_bm25_w3.csv    retrieved_context_windows_bm25_pack
run bm25w5 phq8_item_dataset_context_windows_bm25.csv       retrieved_context_windows_bm25_pack

echo; echo "=== DONE $(date -u) ==="
