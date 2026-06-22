#!/bin/bash
# Domain-model experiment: MentaLLaMA-chat-13B (mental-health-tuned LLaMA-2),
# 4-bit on one 3090, on the SAME W5-hybrid evidence/folds as the Qwen-7B CoT,
# so the only variable is the model. Compare to Qwen-7B-W5 (0.360/0.306).
# Usage: sbatch run_cot_mentallama.sbatch
#SBATCH --job-name=cot_mlama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-5
#SBATCH --output=outputs/logs/slurm_%A_%a.out
#SBATCH --error=outputs/logs/slurm_%A_%a.out

set -euo pipefail
cd /home/yanivpev/MentalHealthSympAI
source .venv/bin/activate
export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

SIZE="${1:-13b}"   # 7b (bf16) | 13b (4-bit)
case "$SIZE" in
  7b)  MODEL=klyang/MentaLLaMA-chat-7B;  QUANT="";          OUTDIR=outputs/cot/folds_mentallama7b ;;
  13b) MODEL=klyang/MentaLLaMA-chat-13B; QUANT="--load-4bit"; OUTDIR=outputs/cot/folds_mentallama ;;
  *) echo "bad size $SIZE"; exit 1 ;;
esac
FOLD="${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | MentaLLaMA-$SIZE | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m src.llm.cot_probe \
  --model-name "$MODEL" $QUANT \
  --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w5.csv \
  --evidence-column retrieved_context_windows_hybrid_list \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "${OUTDIR}/cot_probe_qwen_hybw3_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
