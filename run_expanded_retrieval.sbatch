#!/bin/bash
# Phase 2 of the expanded-retrieval plan: (1) build the three
# expanded-dictionary retrieval variants on the production hybrid path (w3
# bank), (2) keyword hit-rate under both lexicons for each variant
# (tag-suffixed, baseline JSON untouched). Run AFTER run_lexicon_brainstorm
# suggestions have been reviewed/merged into symptom_lexicon.py.
# Usage: sbatch run_expanded_retrieval.sbatch
#SBATCH --job-name=exp_retr
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

echo "=== node: $(hostname) | date: $(date -u) | expanded retrieval build ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

echo; echo "--- 1. build expanded retrieval variants (w3) ---"
python -m src.retrieval.build_expanded_prod_hybrid --window w3

echo; echo "--- 2. keyword hit-rates (both lexicons, per variant) ---"
for V in expbm25 exphyb_bm25q exphyb_bothq; do
  DS="data/processed/phq8_item_dataset_${V}_w3.csv"
  python -m src.evaluation.retrieval_window_analysis \
    --dataset "$DS" --lexicon glossary --out-tag "${V}_glossary"
  python -m src.evaluation.retrieval_window_analysis \
    --dataset "$DS" --lexicon expanded --out-tag "${V}_expanded"
done
# expanded-lexicon hit-rate on the FROZEN hybw3 evidence (R0 comparison point)
python -m src.evaluation.retrieval_window_analysis \
  --lexicon expanded --out-tag "r0hybw3_expanded"

echo; echo "=== DONE $(date -u) ==="
