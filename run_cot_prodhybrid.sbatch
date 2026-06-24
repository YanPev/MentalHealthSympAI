#!/bin/bash
# Item-aware expansion on the PRODUCTION hybrid retriever: 5-fold CoT on the
# bare-query vs item-aware-query production-hybrid evidence (exact off-branch
# retriever). Matched control; compare to each other and to folds_w5 + ensemble.
# Usage: sbatch run_cot_prodhybrid.sbatch <bare|itemaware>
#SBATCH --job-name=cot_ph
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

V="$1"   # bare | itemaware
FOLD="${SLURM_ARRAY_TASK_ID}"
OUTDIR="outputs/cot/folds_prodhybrid_${V}"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | prod-hybrid $V | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset-path "data/processed/phq8_item_dataset_prodhybrid_${V}_w5.csv" \
  --evidence-column retrieved_context_windows_prodhybrid_list \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "${OUTDIR}/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE $V fold $FOLD $(date -u) ==="
