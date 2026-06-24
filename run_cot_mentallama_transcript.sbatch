#!/bin/bash
# Full-transcript experiment: MentaLLaMA-chat-13B (mental-health-tuned LLaMA-2),
# 4-bit on one 3090. Unlike the snippet probe, this gives the model the WHOLE
# interview transcript and scores all 8 PHQ-8 items in one zero-shot pass per
# participant. Tests whether the encoder's ceiling is retrieval (what evidence
# it sees) vs reasoning -- the LLM here gets its natural full-context advantage.
# Same folds/labels as the other probes for a row-for-row comparison.
# Usage: sbatch run_cot_mentallama_transcript.sbatch          # 13b 4-bit
#        sbatch run_cot_mentallama_transcript.sbatch 7b       # 7b  bf16
#SBATCH --job-name=cot_mlama_tx
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
  7b)  MODEL=klyang/MentaLLaMA-chat-7B;  QUANT="";          OUTDIR=outputs/cot/folds_mentallama7b_tx ;;
  13b) MODEL=klyang/MentaLLaMA-chat-13B; QUANT="--load-4bit"; OUTDIR=outputs/cot/folds_mentallama_tx ;;
  *) echo "bad size $SIZE"; exit 1 ;;
esac
FOLD="${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) | date: $(date -u) | MentaLLaMA-$SIZE full-transcript | fold=$FOLD ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"

# Any item-dataset with a transcript_text column works; w3 is the smallest.
python -m src.llm.cot_probe \
  --model-name "$MODEL" $QUANT \
  --full-transcript \
  --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv \
  --transcript-column transcript_text \
  --max-context 4096 --max-new-tokens 768 \
  --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
  --fold "$FOLD" --n-samples 1 \
  --output "${OUTDIR}/cot_transcript_fold${FOLD}.csv"

echo; echo "=== DONE fold $FOLD $(date -u) ==="
