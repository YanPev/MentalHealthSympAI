#!/bin/bash
#SBATCH --job-name=phq8_cotstaged
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
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
echo "###### Staged CoT (confident -> clinical context -> difficult -> final 8) — Qwen2.5-7B, full 5-fold OOF ######"
# Same evidence (Hybrid-W3) and same OOF eval slices as run_cot_joint.sbatch, so
# staged-vs-joint isolates the prompting strategy. Bigger token budget because the
# staged output carries confident + clinical_context + difficult + final scores.
for F in 1 2 3 4 5; do
  echo
  echo "==== fold $F ===="
  python -m src.llm.cot_joint --fold $F \
      --mode staged \
      --max-new-tokens 1100 \
      --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
      --evidence-column retrieved_context_windows_hybrid_list \
      --output outputs/cot/folds_staged/cot_staged_fold${F}.csv
done

echo
echo "=== DONE $(date -u) ==="
