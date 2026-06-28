#!/bin/bash
#SBATCH --job-name=phq8_cotjoint32b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI

source .venv/bin/activate

export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Bigger Qwen, 4-bit so it fits a single 3090. Pre-download the checkpoint into
# .hf_cache from a NETWORKED node first (compute nodes are offline):
#   HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache \
#     .venv/bin/huggingface-cli download Qwen/Qwen2.5-32B-Instruct
MODEL=Qwen/Qwen2.5-32B-Instruct

echo "=== node: $(hostname) | date: $(date -u) | model: $MODEL ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

echo
echo "############ Joint CoT (all 8 items together) — ${MODEL} 4-bit, full 5-fold OOF ############"
for F in 1 2 3 4 5; do
  echo
  echo "==== fold $F ===="
  python -m src.llm.cot_joint --fold $F \
      --model-name "$MODEL" --load-4bit \
      --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
      --evidence-column retrieved_context_windows_hybrid_list \
      --output outputs/cot/folds_joint/cot_joint_32b_fold${F}.csv
done

echo
echo "=== DONE $(date -u) ==="
