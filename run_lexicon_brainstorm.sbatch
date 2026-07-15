#!/bin/bash
# One-off LLM lexicon brainstorm (suggestions only; human-reviewed before any
# term enters src/retrieval/symptom_lexicon.py). Runs before the expanded
# retrieval build so accepted terms can be merged into the dictionary first.
# Usage: sbatch run_lexicon_brainstorm.sbatch
#SBATCH --job-name=lex_brain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) | date: $(date -u) | lexicon brainstorm ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.brainstorm_lexicon --n-samples 3

echo; echo "=== DONE $(date -u) ==="
