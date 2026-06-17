"""
Phase B of distillation: fine-tune a small student (Qwen2.5-1.5B-Instruct) to
reproduce the teacher's chain-of-thought + answer, then predict one held-out fold.

Leakage discipline: for fold F we train ONLY on rationales of participants in the
other four folds and predict fold F. Run once per fold (array 1-5) -> a pooled
distilled OOF directly comparable to the 7B CoT and the encoder.

Student input  : the *prediction* prompt (system + item + evidence, NO label).
Student target : {"reasoning": <teacher rationale>, "label": <gold>}.
So the student learns to produce correct reasoning AND the answer from the
question alone -- a cheap, offline stand-in for the 7B teacher.

    python -m src.llm.distill_student --fold 1 \
        --rationales outputs/cot/teacher_rationales_hybw3.jsonl
"""

from pathlib import Path
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.cot_probe import SYSTEM_PROMPT, parse_output

NUM_LABELS = 4
MAX_LEN = 700


def build_prompt_messages(item_text, evidence):
    """The student's INPUT (no label) -- identical at train and inference time."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PHQ-8 item: {item_text}\nEvidence: {evidence}"}]


def target_text(reasoning, label):
    return json.dumps({"reasoning": reasoning, "label": int(label)})


class SFTDataset:
    """Tokenize (prompt -> target); mask prompt tokens so loss is on the answer only."""

    def __init__(self, rows, tok):
        self.ex = []
        for r in rows:
            msgs = build_prompt_messages(r["item_text"], r["evidence"])
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full = prompt + target_text(r["reasoning"], r["label"]) + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
            labels = list(f_ids)
            for i in range(min(len(p_ids), len(f_ids))):
                labels[i] = -100
            self.ex.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids),
                            "labels": labels})

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    import torch
    m = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = m - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    p = argparse.ArgumentParser(description="Distill student, predict one fold (Phase B)")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--rationales", default=str(PROJECT_ROOT / "outputs" / "cot" /
                                               "teacher_rationales_hybw3.jsonl"))
    p.add_argument("--baseline-oof", default=str(PROJECT_ROOT / "outputs" / "cv" /
                                                 "oof_predictions_ctxm_corn_hybw3.csv"))
    p.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out_path = Path(args.output or (PROJECT_ROOT / "outputs" / "cot" / "folds_distill" /
                                    f"cot_probe_distill_fold{args.fold}.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- data + fold split ----
    rat = pd.read_json(args.rationales, lines=True)
    rat["participant_id"] = rat["participant_id"].astype(str)
    enc = pd.read_csv(args.baseline_oof)[["participant_id", "item_id", "fold"]]
    enc["participant_id"] = enc["participant_id"].astype(str)
    rat = rat.merge(enc, on=["participant_id", "item_id"], how="inner")
    assert not rat["fold"].isna().any(), "some rationales have no fold mapping"

    train_rows = rat[rat.fold != args.fold].to_dict("records")
    eval_rows = rat[rat.fold == args.fold].to_dict("records")
    # leakage guard: no participant in both
    assert not (set(r["participant_id"] for r in train_rows) &
                set(r["participant_id"] for r in eval_rows))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print(f"[B] distilling fold {args.fold} | student {args.student_model}")
    print(f"    train {len(train_rows)}  eval {len(eval_rows)}  device {device}")
    print("=" * 64)

    tok = AutoTokenizer.from_pretrained(args.student_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, device_map=device)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- train (manual loop; small + transparent) ----
    ds = SFTDataset(train_rows, tok)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        opt.zero_grad()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            tot += loss.item() * args.grad_accum
            nb += 1
        print(f"  epoch {ep}/{args.epochs}  loss {tot/nb:.4f}  ({time.time()-t0:.0f}s)")

    # ---- predict held-out fold ----
    model.eval()
    rows = []
    for j, r in enumerate(eval_rows):
        msgs = build_prompt_messages(r["item_text"], r["evidence"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        lbl, reasoning = parse_output(text)
        rec = {"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
               "item_name": r["item_name"], "label": int(r["label"]), "prediction": lbl}
        for c in range(NUM_LABELS):
            rec[f"prob_{c}"] = 1.0 if c == lbl else 0.0
        rec["reasoning"] = reasoning
        rows.append(rec)
        if (j + 1) % 50 == 0:
            print(f"  predicted {j+1}/{len(eval_rows)}")

    pred = pd.DataFrame(rows)
    pred.to_csv(out_path, index=False)

    from sklearn.metrics import f1_score, mean_absolute_error, cohen_kappa_score
    y, pp = pred.label.to_numpy(), pred.prediction.to_numpy()
    print("-" * 64)
    print(f"[B] fold {args.fold} distilled student | "
          f"macroF1 {f1_score(y, pp, average='macro', labels=[0,1,2,3], zero_division=0):.3f} "
          f"QWK {cohen_kappa_score(y, pp, weights='quadratic', labels=[0,1,2,3]):.3f} "
          f"MAE {mean_absolute_error(y, pp):.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
