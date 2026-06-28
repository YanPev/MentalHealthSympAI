#!/bin/bash
#SBATCH --job-name=phq8_cotstaged2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:45:00
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
echo "###### Staged CoT v2: severe-class nudge + item-adaptive evidence (full transcript for NoInterest=1 & Appetite=5) — Qwen2.5-7B, 5-fold OOF ######"
# Same fold slices as staged v1 so the deltas isolate the two changes:
#   --severe-nudge            : difficult stage stops under-calling persistent symptoms (lift c3)
#   --full-transcript-items 1,5 : NoInterest & Appetite scored from the whole interview (W3 starves them)
for F in 1 2 3 4 5; do
  echo
  echo "==== fold $F ===="
  python -m src.llm.cot_joint --fold $F \
      --mode staged --severe-nudge \
      --full-transcript-items 1,5 \
      --max-transcript-tokens 6000 \
      --max-new-tokens 1100 \
      --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
      --evidence-column retrieved_context_windows_hybrid_list \
      --output outputs/cot/folds_staged_v2/cot_staged_v2_fold${F}.csv
done

echo
echo "=== DONE $(date -u) ==="
