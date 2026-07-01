#!/bin/bash
# LLM-scored evidence-relevance hit rate: replaces the keyword-substring proxy
# in outputs/cot/retrieval_window_analysis.json with a direct LLM judgment of
# whether retrieved evidence is on-topic for each PHQ-8 item.
# Usage: sbatch run_llm_evidence_relevance.sbatch
#SBATCH --job-name=llm_relevance
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

echo "=== node: $(hostname) | date: $(date -u) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.evaluation.llm_evidence_relevance \
  --model-name Qwen/Qwen2.5-7B-Instruct

echo; echo "=== DONE $(date -u) ==="
