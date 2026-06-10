#!/bin/bash
# Usage: sbatch run_seed_confirm.sbatch <bert|mbert> <hybw3|hybw5>
# Multi-seed confirmation of the top MentalBERT cells: CORN + Hybrid, 5 seeds,
# both models, same folds per seed (paired). Tags: seedconf_<model>_<cfg>_s<seed>.
#SBATCH --job-name=phq8_seedc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MTAG="$1"; CFG="$2"
case "$MTAG" in
  bert)  MODEL=bert-base-uncased ;;
  mbert) MODEL=mental/mental-bert-base-uncased ;;
  *) echo "bad model $MTAG"; exit 1 ;;
esac
case "$CFG" in
  hybw3) DS=phq8_item_dataset_context_windows_hybrid_w3.csv ;;
  hybw5) DS=phq8_item_dataset_context_windows_hybrid_w5.csv ;;
  *) echo "bad cfg $CFG"; exit 1 ;;
esac
COL=retrieved_context_windows_hybrid_pack

echo "=== node: $(hostname) | $(date -u) | $MODEL | $CFG (CORN) ==="
COMMON="--model-name $MODEL --dataset-path data/processed/$DS --evidence-column $COL \
  --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --loss corn"

for SEED in 42 7 31 123 2024; do
  echo; echo "############ seedconf_${MTAG}_${CFG}_s${SEED} ############"
  python -m src.models.cross_validate $COMMON --seed $SEED --tag "seedconf_${MTAG}_${CFG}_s${SEED}"
done

echo; echo "=== DONE $(date -u) ==="
