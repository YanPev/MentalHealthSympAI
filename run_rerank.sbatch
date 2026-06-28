#!/bin/bash
#SBATCH --job-name=phq8_rerank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
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

echo
echo "############ Step 1/2 : cross-encoder rerank windows -> reranked_pack ############"
python -m src.retrieval.rerank_windows --top-k 5 \
    --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
    --out-path data/processed/phq8_item_dataset_reranked_w3.csv

echo
echo "############ Step 2/2 : CV on reranked evidence (MentalBERT + CORN) ############"
python -m src.models.cross_validate \
    --dataset-path data/processed/phq8_item_dataset_reranked_w3.csv \
    --evidence-column reranked_pack \
    --model-name mental/mental-bert-base-uncased \
    --loss corn --num-epochs 8 --k-folds 5 --batch-size 16 --max-length 256 --seed 42 \
    --tag ctxm_corn_rerank_w3

echo
echo "=== DONE $(date -u) ==="
