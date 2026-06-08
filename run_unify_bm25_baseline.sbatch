#!/bin/bash
#SBATCH --job-name=phq8_unif
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

# Unify the earlier model/loss baselines onto the BM25 utterance dataset
# (phq8_item_dataset_full_bm25.csv) so ALL project numbers share one baseline.
DS=data/processed/phq8_item_dataset_full_bm25.csv
COMMON="--dataset-path $DS --evidence-column retrieved_utterances --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42"

run () { echo; echo "############ $1 ############"; python -m src.models.cross_validate $COMMON --model-name "$2" --tag "$1" $3; }

run bm25base_bert_ce   bert-base-uncased               "--loss cross_entropy --class-weights balanced"
run bm25base_bert_corn bert-base-uncased               "--loss corn"
run bm25base_mbert_ce  mental/mental-bert-base-uncased  "--loss cross_entropy --class-weights balanced"
run bm25base_mbert_corn mental/mental-bert-base-uncased "--loss corn"

echo; echo "=== DONE $(date -u) ==="
