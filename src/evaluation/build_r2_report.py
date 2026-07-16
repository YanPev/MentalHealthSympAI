"""
Final assembly -- run_manifest.json, status_report.md, results_report.html.

Reads every R2 artifact that exists and renders a focused, decision-oriented HTML
report (brief §17, 17 sections). Missing artifacts render as "pending" rather
than failing, so the report can be built incrementally as stages complete.

    .venv/bin/python -m src.evaluation.build_r2_report
"""

from pathlib import Path
import glob
import html
import json

import pandas as pd

PR = Path(__file__).resolve().parents[2]
R2 = PR / "outputs" / "r2_systematic"
LEX, RET, JUD, ENC, LLM, ANA, CAS = (R2 / d for d in
    ("lexicons", "retrieval", "judge", "encoder", "llm", "analysis", "cascade"))


def _read_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _table(df, cols=None, floatfmt=4):
    if df is None or len(df) == 0:
        return "<p class='pending'>pending</p>"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.round(floatfmt) if floatfmt else df
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_manifest():
    files = sorted(str(Path(p).relative_to(PR)) for p in
                   glob.glob(str(R2 / "**" / "*"), recursive=True) if Path(p).is_file())
    manifest = {
        "experiment": "R2 corrective retrieval experiment",
        "judge_model": "klyang/MentaLLaMA-chat-7B (LLaMA-2 family; independent of Qwen predictor)",
        "predictor_encoder": "mental/mental-bert-base-uncased + CORN (5-fold, seed 42, 8 epochs, bs16, maxlen256)",
        "predictor_llm": "Qwen2.5-7B-Instruct staged-tolerant CoT, SC5 @ T0.7",
        "folds": "StratifiedGroupKFold(5, shuffle, seed42) grouped by participant",
        "window_bank": "w3",
        "r2_selection": _read_json(RET / "r2_selection.json"),
        "e_final": _read_json(ENC / "e_final_selection.json"),
        "artifacts": files,
    }
    (R2 / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_status():
    lines = ["# R2 corrective experiment — status report", ""]
    checks = [
        ("Lexicon L0-L3 + clinical audit", LEX / "clinical_audit.csv"),
        ("Semantic multi-queries", R2 / "semantic_queries" / "semantic_queries.json"),
        ("Rankings (BM25 + semantic Top-20)", RET / "retrieval_window_scores.parquet"),
        ("Judge window judgments", JUD / "window_judgments_fold1.csv"),
        ("Judge set judgments", JUD / "set_judgments_fold1.csv"),
        ("R2 selection", RET / "r2_selection.json"),
        ("Encoder metrics", ENC / "encoder_metrics_overall.csv"),
        ("Missing-policy audit", ENC / "missing_policy_audit.csv"),
        ("LLM metrics", LLM / "llm_metrics_overall.csv"),
        ("Complementarity", ANA / "complementarity_overall.csv"),
        ("Cascade metrics", CAS / "cascade_metrics.csv"),
    ]
    for name, p in checks:
        lines.append(f"- [{'x' if Path(p).exists() else ' '}] {name}  (`{Path(p).name}`)")
    (R2 / "status_report.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


SECTIONS = [
    ("1. Why the previous R1 selection needed correcting",
     "R1 (exphyb_bm25q) lumped lay+clinical lexicons together, used an LLM-brainstormed "
     "clinical tier without provenance, encoded a concatenated keyword dump as the semantic "
     "query, never evaluated semantic-only or varied K/alpha, judged evidence with the SAME "
     "Qwen family as the predictor, and its cascade thresholds were picked on the OOF. Its "
     "measured gains over R0 were not significant and it degraded the project-best cascade."),
    ("2. Frozen R0 and previous R1", None),
    ("3. Clinical lexicon audit", None),
    ("4. Core vs Lay vs Clinical vs Full", None),
    ("5. Corrected semantic multi-query method", None),
    ("6. BM25 vs semantic evidence quality", None),
    ("7. Evidence complementarity", None),
    ("8. Whether hybrid was justified", None),
    ("9. Top-3 vs Top-5 vs token-budget", None),
    ("10. Selected R2 + label-free rule", None),
    ("11. MentalBERT+CORN + missing-policy audit", None),
    ("12. Qwen retrieved vs long-context", None),
    ("13. Encoder vs LLM by item and evidence status", None),
    ("14. Encoder-first vs LLM-first cascade", None),
    ("15. Severe-class results", None),
    ("16. Final recommended system", None),
    ("17. Limitations", None),
]


def build_html():
    sel = _read_json(RET / "r2_selection.json")
    parts = ["""<style>
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}
    h1{border-bottom:3px solid #2c5f8a} h2{margin-top:2rem;color:#2c5f8a;border-bottom:1px solid #ddd}
    table{border-collapse:collapse;margin:.5rem 0;font-size:.85rem} th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}
    th:first-child,td:first-child{text-align:left} th{background:#eef3f8}
    .pending{color:#a00;font-style:italic} .lab{background:#eef;padding:2px 6px;border-radius:3px;font-size:.8rem}
    code{background:#f4f4f4;padding:1px 4px} .note{color:#555;font-size:.9rem}
    </style>"""]
    parts.append("<h1>R2 corrective retrieval experiment — results</h1>")
    parts.append("<p class='note'>Independent judge: <b>MentaLLaMA-chat-7B</b> (LLaMA-2, "
                 "different family from the Qwen2.5-7B predictor). All retrieval selection is "
                 "label-free. Every claim tags its source artifact; effects with a 95% CI that "
                 "includes 0 are <b>not</b> called significant.</p>")
    if sel:
        r2 = sel.get("selected_R2", {})
        parts.append(f"<p><span class='lab'>Selected R2</span> config=<b>{r2.get('config')}</b> "
                     f"retriever=<b>{r2.get('retriever')}</b> prefix=<b>{r2.get('prefix')}</b> "
                     f"(informative rate {r2.get('informative_rate')})</p>")

    data = {
        "3. Clinical lexicon audit": _table(_read_csv(LEX / "clinical_audit.csv").groupby(
            ["item", "decision"]).size().rename("n").reset_index() if _read_csv(LEX / "clinical_audit.csv") is not None else None),
        "4. Core vs Lay vs Clinical vs Full": _table(_read_csv(RET / "retrieval_metrics_overall.csv"),
            ["config", "retriever", "prefix", "informative_rate", "none_rate", "ambiguous_rate"]),
        "6. BM25 vs semantic evidence quality": _table(_read_csv(RET / "retrieval_metrics_overall.csv"),
            ["config", "retriever", "prefix", "informative_rate", "supports_rate", "against_rate", "none_rate"]),
        "7. Evidence complementarity": _table(_read_csv(RET / "retrieval_complementarity.csv")),
        "10. Selected R2 + label-free rule": f"<pre>{html.escape(json.dumps(sel, indent=2))}</pre>" if sel else "<p class='pending'>pending</p>",
        "11. MentalBERT+CORN + missing-policy audit": _table(_read_csv(ENC / "encoder_metrics_overall.csv"),
            ["run", "macro_f1", "qwk", "mae", "f1_class3", "severe_recall", "false_severe_rate"]),
        "12. Qwen retrieved vs long-context": _table(_read_csv(LLM / "llm_metrics_overall.csv")),
        "13. Encoder vs LLM by item and evidence status": _table(_read_csv(ANA / "complementarity_by_evidence.csv")),
        "14. Encoder-first vs LLM-first cascade": _table(_read_csv(CAS / "cascade_metrics.csv"),
            ["model", "macro_f1", "qwk", "mae", "severe_recall", "false_severe_rate"]),
    }
    for title, body in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        if body:
            parts.append(f"<p>{html.escape(body)}</p>")
        if title in data:
            parts.append(data[title])
        elif not body:
            parts.append("<p class='pending'>pending — populated when the stage completes</p>")
    (PR / "docs" / "r2_results_report.html").write_text("\n".join(parts))
    (R2 / "results_report.html").write_text("\n".join(parts))
    return PR / "docs" / "r2_results_report.html"


def main():
    build_manifest()
    build_status()
    p = build_html()
    print(f"Wrote run_manifest.json, status_report.md, and {p}")


if __name__ == "__main__":
    main()
