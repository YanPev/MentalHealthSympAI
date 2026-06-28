#!/bin/bash
#SBATCH --job-name=phq8_coltsc5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

echo
echo "###### Self-consistency x5 TOLERANT staged CoT — graded confidence for the cascade router — Qwen2.5-7B, 5-fold OOF ######"
# Same prompt/evidence/folds as the greedy tolerant run, but N=5 sampled chains @ T=0.7
# so prob_0..3 become per-item vote fractions (the LLM confidence the cascade needs).
for F in 1 2 3 4 5; do
  echo
  echo "==== fold $F ===="
  python -m src.llm.cot_joint --fold $F \
      --mode staged --tolerant \
      --n-samples 5 --temperature 0.7 \
      --max-new-tokens 1100 \
      --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
      --evidence-column retrieved_context_windows_hybrid_list \
      --output outputs/cot/folds_tolerant_sc5/cot_tolerant_sc5_fold${F}.csv
done

echo
echo "=== DONE $(date -u) ==="
