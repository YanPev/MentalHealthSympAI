#!/bin/bash
# Phase 5: staged-tolerant self-consistency-x5 CoT (Qwen2.5-7B) on the R1
# expanded-retrieval evidence (exphyb_bm25q), 5-fold OOF. Same prompt/mode/
# hyperparameters as the frozen folds_tolerant_sc5 R0 pool; only the retrieval
# evidence changes, so the cascade can be rebuilt on R1. R0 pools are reused
# as-is (not recomputed). Output: outputs/cot/folds_exphybw3/.
# Usage: sbatch run_cot_probe_expfolds.sbatch
#SBATCH --job-name=cot_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FOLD="${SLURM_ARRAY_TASK_ID}"
OUTDIR="outputs/cot/folds_exphybw3"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | CoT SC5 staged-tolerant | R1 exphyb_bm25q | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_joint --fold "$FOLD" \
    --mode staged --tolerant \
    --n-samples 5 --temperature 0.7 \
    --max-new-tokens 1100 \
    --dataset-path data/processed/phq8_item_dataset_exphyb_bm25q_w3.csv \
    --evidence-column retrieved_context_windows_hybrid_list \
    --output "${OUTDIR}/cot_tolerant_sc5_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
