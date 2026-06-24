#!/bin/bash
# Tame over-attribution: re-run CoT with the STRICT (conservative / abstain-to-0)
# prompt on both evidence types, so item-adaptive can be rebuilt consistently.
# Usage: sbatch run_cot_strict.sbatch <w5|full>
#SBATCH --job-name=cot_strict
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

EV="$1"   # w5 | full
case "$EV" in
  w5)   COL=retrieved_context_windows_hybrid_list; OUTDIR=outputs/cot/folds_strict_w5 ;;
  full) COL=transcript_text;                        OUTDIR=outputs/cot/folds_strict_full ;;
  *) echo "bad ev $EV"; exit 1 ;;
esac
FOLD="${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | STRICT CoT $EV | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct --strict \
  --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w5.csv \
  --evidence-column "$COL" \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "${OUTDIR}/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE $EV fold $FOLD $(date -u) ==="
