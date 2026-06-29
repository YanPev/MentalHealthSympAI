#!/bin/bash
#SBATCH --job-name=phq8_coltsc5dv
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo
echo "###### SC x5 TOLERANT staged CoT WITH per-item difficult-vote tracking (difficult_frac) — robust cascade gate ######"
# Same config as run_cot_tolerant_sc5.sbatch, but the updated cot_joint.py now stores
# difficult_frac = fraction of the 5 chains that put the item in the disjoint difficult
# set (difficult - confident). New output dir so the prior sc5 run is preserved.
for F in 1 2 3 4 5; do
  echo
  echo "==== fold $F ===="
  python -m src.llm.cot_joint --fold $F \
      --mode staged --tolerant \
      --n-samples 5 --temperature 0.7 \
      --max-new-tokens 1100 \
      --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
      --evidence-column retrieved_context_windows_hybrid_list \
      --output outputs/cot/folds_tolerant_sc5_dv/cot_tolerant_sc5_dv_fold${F}.csv
done

echo
echo "=== DONE $(date -u) ==="
