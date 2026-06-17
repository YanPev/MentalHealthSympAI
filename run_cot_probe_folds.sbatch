#!/bin/bash
# Step 1 of the CoT plan: run the few-shot CoT probe across ALL 5 folds so we get
# a pooled OOF (n=1752) directly comparable to the encoder, plus a per-fold spread.
# One array task per fold -> runs in parallel on separate 3090s (~30 min each).
# Usage: sbatch run_cot_probe_folds.sbatch [n_samples]   (default 1 = greedy)
#SBATCH --job-name=cot_folds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

NSAMP="${1:-1}"
FOLD="${SLURM_ARRAY_TASK_ID}"
# Keep greedy (folds/) and self-consistency (folds_scN/) results separate.
if [ "$NSAMP" -gt 1 ]; then OUTDIR="outputs/cot/folds_sc${NSAMP}"; else OUTDIR="outputs/cot/folds"; fi
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | CoT probe | fold=$FOLD n_samples=$NSAMP -> $OUTDIR ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" \
  --n-samples "$NSAMP" \
  --output "${OUTDIR}/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
