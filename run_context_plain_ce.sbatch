#!/bin/bash
#SBATCH --job-name=phq8_ctxce
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

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

DP=data/processed
# PLAIN cross-entropy (no class weights) to complete the CE vs weighted-CE vs CORN triad.
COMMON="--model-name bert-base-uncased --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 --loss cross_entropy --class-weights none"

run () { echo; echo "############ $1 ############"; python -m src.models.cross_validate $COMMON --dataset-path "$DP/$2" --evidence-column "$3" --tag "$1"; }

run ctx_ceplain_old    phq8_item_dataset_full_bm25.csv                  retrieved_utterances
run ctx_ceplain_hybw3  phq8_item_dataset_context_windows_hybrid_w3.csv  retrieved_context_windows_hybrid_pack
run ctx_ceplain_hybw5  phq8_item_dataset_context_windows_hybrid_w5.csv  retrieved_context_windows_hybrid_pack
run ctx_ceplain_bm25w3 phq8_item_dataset_context_windows_bm25_w3.csv    retrieved_context_windows_bm25_pack
run ctx_ceplain_bm25w5 phq8_item_dataset_context_windows_bm25.csv       retrieved_context_windows_bm25_pack

echo; echo "=== DONE $(date -u) ==="
