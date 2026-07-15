#!/bin/bash
# Retry of run_cot_probe_expfolds.sbatch folds 2-4, which OOM'd twice (jobs
# 19371881, 19376654): five 1100-token self-consistency chains generated in one
# batch peak ~21GB on those long-transcript folds and exceed a 24GB 3090.
# Folds 1 and 5 already succeeded and are kept as-is. This retry adds
# --sc-batch-size 1 (generate the 5 chains one at a time; i.i.d. samples, so the
# vote is statistically identical) plus expandable_segments. Output appends to
# outputs/cot/folds_exphybw3/.
# Usage: sbatch run_cot_probe_expfolds_retry.sbatch
#SBATCH --job-name=cot_exp_retry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:30:00
#SBATCH --array=2-4
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD="${SLURM_ARRAY_TASK_ID}"
OUTDIR="outputs/cot/folds_exphybw3"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | CoT SC5 staged-tolerant | R1 exphyb_bm25q | fold=$FOLD (retry, sc-batch 1) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_joint --fold "$FOLD" \
    --mode staged --tolerant \
    --n-samples 5 --temperature 0.7 --sc-batch-size 1 \
    --max-new-tokens 1100 \
    --dataset-path data/processed/phq8_item_dataset_exphyb_bm25q_w3.csv \
    --evidence-column retrieved_context_windows_hybrid_list \
    --output "${OUTDIR}/cot_tolerant_sc5_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
