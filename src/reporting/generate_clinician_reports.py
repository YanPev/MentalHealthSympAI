"""Phase 7 — render clinician-facing reports from the structured schema.

Turns each participant record from report_schema.py into the "Dear clinician"
letter the research plan specifies. The default renderer is DETERMINISTIC: every
figure is copied verbatim from the schema, so the letter cannot fabricate a
score, a confidence, or an evidence snippet — the right property for a clinical
artifact. An optional `--use-llm` path asks Qwen to smooth the narrative prose
only (the structured findings block is always template-filled and appended
unchanged), for teams that prefer softer phrasing.

The letter separates, as required: high- vs low-confidence findings, missing
evidence, contradictory evidence, and inferences needing clinician verification.

    python -m src.reporting.generate_clinician_reports --limit 12
    python -m src.reporting.generate_clinician_reports --use-llm   # optional GPU polish

Reads outputs/cot/clinician_report_schema.json (run report_schema.py first) and
writes outputs/reports/participant_<id>.md.
"""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]
COT = ROOT / "outputs" / "cot"
REPORTS = ROOT / "outputs" / "reports"
DEF_SCHEMA = COT / "clinician_report_schema.json"

DISCLAIMER = (
    "These findings are model-generated decision support, not a diagnosis. They "
    "are intended to highlight areas that may warrant further clinical assessment "
    "or discussion during treatment, and every item flagged for verification "
    "should be confirmed against the clinician's own judgement.")


def _evidence_clause(it):
    """Phrase the evidence line by the JUDGE's verdict, not the model's softmax —
    so a model-confident score built on absent/contradictory evidence is shown as
    an inference to verify, never as a fabricated 'supporting' quote."""
    status = it["evidence_status"]
    snip = it["evidence_snippets"][0] if it["evidence_snippets"] else None
    if status == "supports" and snip:
        return f"Supporting evidence: \"{snip}\""
    if status == "against":
        return ("Note: interview evidence appears to *contradict* this score — "
                "model inference, verify.")
    if status == "ambiguous":
        return (f"Evidence ambiguous" + (f": \"{snip}\"" if snip else "") +
                " — interpret with caution.")
    return ("No clear interview evidence for this item; score is a model "
            "inference and should be verified.")


def _item_line(it):
    name, txt = it["name"], it["item_text"]
    if it["predicted_score"] is None:
        return (f"- **{name}** ({txt}): insufficient evidence was found in the "
                f"interview, and no reliable score was assigned.")
    return (f"- **{name}** ({txt}): predicted score **{it['predicted_score']}/3**, "
            f"confidence {it['confidence_category']} ({it['confidence']}). "
            f"{_evidence_clause(it)}")


def render(schema):
    t = schema["phq_total"]
    L = [f"Dear clinician,", "",
         "Based on the most recent interview, the system identified several "
         "symptoms that may require further attention.", "",
         f"The predicted overall depression severity is **{t['band']}** "
         f"(total {t['predicted_score']}/24), with **{t['confidence_category']}** "
         f"confidence. The participant's own questionnaire self-report totals "
         f"{t['self_report_total']}/24 ({t['self_report_band']}).", ""]

    hi = [it for it in schema["items"]
          if it["predicted_score"] is not None and it["confidence_category"] == "high"]
    lo = [it for it in schema["items"]
          if it["predicted_score"] is not None and it["confidence_category"] in ("moderate", "low")]

    L.append("**High-confidence findings:**")
    L += [_item_line(it) for it in hi] or ["- (none)"]
    L += ["", "**Lower-confidence findings (interpret with caution):**"]
    L += [_item_line(it) for it in lo] or ["- (none)"]

    if schema["no_evidence_items"]:
        L += ["", "**Symptoms with no sufficient interview evidence:**",
              "- " + ", ".join(schema["no_evidence_items"]) +
              " — the transcript did not provide enough information to estimate "
              "these items reliably."]

    contra = [it for it in schema["items"] if it["flags"]["contradiction"]]
    if contra:
        L += ["", "**Questionnaire / interview inconsistencies (verify):**"]
        for it in contra:
            L.append(f"- **{it['name']}**: self-reported {it['questionnaire_self_report']}/3 "
                     f"but the interview evidence is `{it['evidence_status']}`"
                     + (f" (predicted {it['predicted_score']}/3)"
                        if it["predicted_score"] is not None else "")
                     + (f" — {it['evidence_reason']}" if it["evidence_reason"] else "") + ".")

    nd = [it for it in schema["items"] if it["flags"]["not_discussed"]]
    if nd:
        L += ["", "**Self-reported but not discussed in the interview:**",
              "- " + ", ".join(it["name"] for it in nd) + "."]
    io = [it for it in schema["items"] if it["flags"]["interview_only"]]
    if io:
        L += ["", "**Suggested by the interview but not self-reported:**",
              "- " + ", ".join(it["name"] for it in io) + "."]

    if schema["followup_items"]:
        L += ["", "**Areas requiring clinician verification / follow-up:**"]
        L += [f"- {f['name']} ({', '.join(f['why'])})" for f in schema["followup_items"]]

    L += ["", "---", DISCLAIMER]
    return "\n".join(L)


def llm_polish(schema, narrative, model, tok):
    """Optional: ask the LLM to rewrite only the opening narrative paragraph in
    warmer prose. The structured findings block is appended unchanged."""
    import torch
    prompt = (
        "Rewrite the following clinical summary opening in 2-3 warm, precise "
        "sentences for a treating clinician. Do NOT invent or change any number, "
        "score, or symptom. Keep it factual.\n\n" + narrative.split("\n\n**High")[0])
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(ids, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=180, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    opener = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    rest = "\n\n**High" + narrative.split("\n\n**High", 1)[1]
    return opener + "\n" + rest


def main():
    ap = argparse.ArgumentParser(description="Render clinician reports from schema")
    ap.add_argument("--schema", default=str(DEF_SCHEMA))
    ap.add_argument("--out-dir", default=str(REPORTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-llm", action="store_true",
                    help="Polish the opening narrative with Qwen (needs GPU). The "
                         "structured findings are always template-filled.")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    schemas = json.loads(Path(args.schema).read_text())
    if args.limit:
        schemas = schemas[:args.limit]
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    model = tok = None
    if args.use_llm:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu").eval()

    for s in schemas:
        md = render(s)
        if args.use_llm:
            md = llm_polish(s, md, model, tok)
        (outdir / f"participant_{s['participant_id']}.md").write_text(md)
    print(f"wrote {len(schemas)} reports to {outdir}/")
    # echo one example to the console
    print("\n" + "=" * 70 + "\nEXAMPLE\n" + "=" * 70)
    print(render(schemas[0]))


if __name__ == "__main__":
    main()
