"""
One-off offline-LLM brainstorm pass for the tiered symptom lexicon.

Asks Qwen2.5-7B-Instruct, per PHQ-8 item, for additional lay paraphrases and
clinical terms beyond what symptom_lexicon.py already contains. The output is
SUGGESTIONS ONLY: a human reviews outputs/cot/lexicon_brainstorm.json and
pastes accepted terms into src/retrieval/symptom_lexicon.py (with provenance
comments). Nothing is consumed programmatically, so unreviewed terms can never
enter retrieval.

Label-free by construction: the prompt sees only the PHQ-8 item definition and
the existing lexicon -- never DAIC-WOZ transcripts or labels.

    python -m src.llm.brainstorm_lexicon            # needs a GPU
"""

from pathlib import Path
import argparse
import json
import re

from src.data.phq8_items import PHQ8_ITEMS
from src.retrieval.symptom_lexicon import CLINICAL, SYNONYMS, flat_terms

PR = Path(__file__).resolve().parents[2]
OUT_JSON = PR / "outputs" / "cot" / "lexicon_brainstorm.json"

SYSTEM_PROMPT = (
    "You are a clinical-language lexicographer helping expand a retrieval "
    "dictionary for a depression-screening system. Given a PHQ-8 symptom item "
    "and the terms the dictionary already contains, propose ADDITIONAL terms "
    "that people use for this symptom:\n"
    '  "lay": everyday spoken paraphrases and multi-word expressions '
    "(as said in a casual interview),\n"
    '  "clinical": professional/DSM-5-register terminology.\n\n'
    "Rules: do not repeat any term already in the dictionary; keep terms "
    "specific to THIS symptom (not depression in general); no sentences, just "
    "short terms/phrases; 10-15 per list.\n"
    'Respond with ONLY a JSON object: {"lay": [...], "clinical": [...]}.'
)


def build_user(item_text: str, name: str) -> str:
    existing = sorted(set(flat_terms(name)))
    return (f"PHQ-8 item: {item_text}\n"
            f"Symptom key: {name}\n"
            f"Already in the dictionary (do not repeat): {', '.join(existing)}")


def parse_terms(text: str):
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    lay = [str(t).lower().strip() for t in obj.get("lay", []) if str(t).strip()]
    cli = [str(t).lower().strip() for t in obj.get("clinical", []) if str(t).strip()]
    return {"lay": lay, "clinical": cli}


def main():
    ap = argparse.ArgumentParser(description="LLM lexicon brainstorm (suggestions only)")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="independent sampled brainstorms per item (union kept)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=qcfg, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    out = {}
    for item_id, item in sorted(PHQ8_ITEMS.items()):
        name = item["symptom_name"]
        existing = set(flat_terms(name))
        lay, cli = set(), set()
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user(item["item_text"], name)}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        for _ in range(args.n_samples):
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_terms(text)
            if parsed:
                lay.update(t for t in parsed["lay"] if t not in existing)
                cli.update(t for t in parsed["clinical"] if t not in existing)
        out[name] = {"item_id": item_id, "item_text": item["item_text"],
                     "suggested_lay": sorted(lay), "suggested_clinical": sorted(cli)}
        print(f"{name:14s} +{len(lay)} lay, +{len(cli)} clinical suggestions")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"model": args.model_name, "n_samples": args.n_samples,
         "note": "SUGGESTIONS ONLY -- review and paste accepted terms into "
                 "src/retrieval/symptom_lexicon.py; nothing reads this file.",
         "per_item": out}, indent=2))
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
