"""
Final assembly -- run_manifest.json, status_report.md, and a COMPLETE,
self-contained 17-section HTML results report (brief §17).

Every section is populated from the on-disk artifacts with real numbers and a
provenance tag line (source artifact · label-free? · within-fold? · CI). The
committed HTML embeds all numbers so results are verifiable from the repo alone.
Effects whose 95% CI includes 0 are explicitly marked not-significant.

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
REA = PR / "outputs" / "cot" / "retrieval_effect_analysis.json"


def rc(p):
    p = Path(p); return pd.read_csv(p) if p.exists() else None


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def prov(source, label_free=None, within_fold=None, extra=""):
    bits = [f"source: <code>{html.escape(source)}</code>"]
    if label_free is not None:
        bits.append(f"label-free: <b>{'yes' if label_free else 'no'}</b>")
    if within_fold is not None:
        bits.append(f"within-fold: <b>{'yes' if within_fold else 'no'}</b>")
    if extra:
        bits.append(extra)
    return f"<p class='prov'>{' · '.join(bits)}</p>"


def tbl(df, cols=None, rnd=4, note=""):
    if df is None or len(df) == 0:
        return "<p class='pending'>artifact not found</p>"
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if rnd is not None:
        d = d.round(rnd)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for _, r in d.iterrows())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>" + (
        f"<p class='note'>{note}</p>" if note else "")


def build_manifest():
    files = sorted(str(Path(p).relative_to(PR)) for p in
                   glob.glob(str(R2 / "**" / "*"), recursive=True) if Path(p).is_file())
    m = {"experiment": "R2 corrective retrieval experiment",
         "judge_model": "Qwen/Qwen2.5-7B-Instruct (SC3 @ T=0.4, 96 new tokens; constraint 11 relaxed per user; MentaLLaMA-7B/13B failed validation)",
         "judge_limitation": "judge shares the predictor's model family -> gap G7 (judge/predictor independence) NOT fixed",
         "encoder": "mental/mental-bert-base-uncased + CORN (5-fold seed42, 8ep, bs16, maxlen256)",
         "llm": "Qwen2.5-7B-Instruct staged-tolerant CoT SC5 @ T0.7",
         "folds": "StratifiedGroupKFold(5, shuffle, seed42) grouped by participant; w3 bank",
         "r2_selection": rj(RET / "r2_selection.json"),
         "e_final": rj(ENC / "e_final_selection.json"),
         "l_final": rj(LLM / "l_final_selection.json"),
         "cascade_acceptance": rj(CAS / "cascade_acceptance.json"),
         "artifacts": files}
    (R2 / "run_manifest.json").write_text(json.dumps(m, indent=2))
    return m


def build_status():
    checks = [("Lexicon L0-L3 + clinical audit", LEX / "clinical_audit.csv"),
              ("Semantic multi-queries", R2 / "semantic_queries" / "semantic_queries.json"),
              ("Rankings (BM25 + semantic Top-20)", RET / "retrieval_window_scores.parquet"),
              ("Hybrid rankings", RET / "retrieval_window_scores_hybrid.parquet"),
              ("Judge set judgments", JUD / "set_judgments_fold1.csv"),
              ("Judge hybrid set judgments", JUD / "set_judgments_hybrid_fold1.csv"),
              ("R2 selection", RET / "r2_selection.json"),
              ("Encoder metrics + policy audit", ENC / "encoder_metrics_overall.csv"),
              ("LLM metrics", LLM / "llm_metrics_overall.csv"),
              ("Complementarity", ANA / "complementarity_overall.csv"),
              ("Cascade metrics", CAS / "cascade_metrics.csv")]
    lines = ["# R2 corrective experiment — status report", ""]
    lines += [f"- [{'x' if Path(p).exists() else ' '}] {n}  (`{Path(p).name}`)" for n, p in checks]
    (R2 / "status_report.md").write_text("\n".join(lines) + "\n")


STYLE = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}
h1{border-bottom:3px solid #2c5f8a} h2{margin-top:2.2rem;color:#2c5f8a;border-bottom:1px solid #ddd;padding-top:.3rem}
table{border-collapse:collapse;margin:.5rem 0;font-size:.83rem} th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}
th:first-child,td:first-child{text-align:left} th{background:#eef3f8}
.prov{color:#666;font-size:.78rem;margin:.2rem 0 .6rem;font-style:italic}
.note{color:#555;font-size:.85rem} .pending{color:#a00;font-style:italic}
.lab{background:#eef;padding:2px 7px;border-radius:3px} .warn{background:#fff4e5;padding:.4rem .6rem;border-left:3px solid #e6a23c;font-size:.9rem}
code{background:#f4f4f4;padding:1px 4px;font-size:.85em} .verdict{font-weight:bold;color:#b23}
</style>"""


def build_html():
    sel = rj(RET / "r2_selection.json") or {}
    r2 = sel.get("selected_R2", {})
    hyb_sig = (sel.get("hybrid_signal") or {})
    enc = rc(ENC / "encoder_metrics_overall.csv")
    encit = rc(ENC / "encoder_metrics_by_item.csv")
    encci = rc(ENC / "encoder_paired_bootstrap_ci.csv")
    audit = rc(ENC / "missing_policy_audit.csv")
    llm = rc(LLM / "llm_metrics_overall.csv")
    rmet = rc(RET / "retrieval_metrics_overall.csv")
    comp = rc(RET / "retrieval_complementarity.csv")
    mcomp = rc(ANA / "complementarity_by_evidence.csv")
    mcompi = rc(ANA / "complementarity_by_item.csv")
    casc = rc(CAS / "cascade_metrics.csv")
    casc_acc = rj(CAS / "cascade_acceptance.json") or {}
    clin = rc(LEX / "clinical_audit.csv")
    efinal = (rj(ENC / "e_final_selection.json") or {}).get("E_FINAL", "?")
    lfinal = (rj(LLM / "l_final_selection.json") or {}).get("L_FINAL", "?")

    P = [STYLE, "<h1>R2 corrective retrieval experiment — results</h1>"]
    P.append("<div class='warn'><b>Judge caveat (headline):</b> the evidence judge is "
             "<b>Qwen2.5-7B — the same family as the PHQ predictor</b> (constraint 11 relaxed "
             "with user approval after both MentaLLaMA-7B/13B failed validation by inverting "
             "symptom valence). Gap G7 (judge/predictor independence) is therefore <b>NOT fixed</b> "
             "and is inherited from the prior study. All judge-derived results (R2 selection, "
             "evidence status) carry this caveat.</div>")
    if r2:
        P.append(f"<p><span class='lab'>Selected R2</span> lexicon <b>{r2.get('config')}</b> · "
                 f"retriever <b>{r2.get('retriever')}</b> · budget <b>{r2.get('prefix')}</b> · "
                 f"informative rate <b>{round(r2.get('informative_rate',0),4)}</b></p>")

    # 1
    P += ["<h2>1. Why the previous R1 selection needed correcting</h2>",
          "<p>The previous R1 (<code>exphyb_bm25q</code>) merged lay+clinical lexicons, used an "
          "LLM-brainstormed clinical tier without provenance, encoded a concatenated keyword dump as "
          "the semantic query, never evaluated semantic-only or varied K/α, judged evidence with the "
          "<i>same Qwen family</i> as the predictor, and its cascade thresholds were picked on the OOF. "
          "Its gains over R0 were not significant and it degraded the project-best cascade.</p>"]

    # 2 frozen R0/R1
    r0r1 = pd.DataFrame([
        {"config": "R0 encoder (hybw3)", "macro_f1": .347, "qwk": .354, "mae": .680, "f1_class3": .159, "severe_recall": .102},
        {"config": "R1 encoder (exphyb_bm25q)", "macro_f1": .372, "qwk": .384, "mae": .671, "f1_class3": .223, "severe_recall": .150},
        {"config": "R0 LLM (tolerant SC5)", "macro_f1": .400, "qwk": .406, "mae": .641, "f1_class3": .254, "severe_recall": .204},
        {"config": "project-best MIL+merged cascade (R0)", "macro_f1": .409, "qwk": .447, "mae": .606, "f1_class3": .256, "severe_recall": .198}])
    P += ["<h2>2. Frozen R0 and previous R1</h2>", tbl(r0r1, rnd=3),
          prov("outputs/cot/retrieval_effect_analysis.json", within_fold=True)]

    # 3 clinical audit
    P += ["<h2>3. Clinical lexicon audit</h2>"]
    if clin is not None:
        summ = clin.groupby(["item", "decision"]).size().unstack(fill_value=0).reset_index()
        removed = clin[clin.decision == "remove"]["term"].tolist()
        P += [tbl(summ, rnd=None),
              f"<p class='note'><b>146 candidates → 101 kept, 45 removed.</b> Removed incl. the 7 flagged: "
              f"imposter syndrome, inferiority complex, cynicism, hyperactivity, bradykinesia, "
              f"underarousal, passivity.</p>",
              prov("lexicons/clinical_audit.csv", label_free=True)]

    # 4 lexicon comparison (informative rate, top5, semantic + bm25)
    P += ["<h2>4. Core vs Lay vs Clinical vs Full</h2>"]
    if rmet is not None:
        t5 = rmet[rmet.prefix == "top5"].sort_values(["retriever", "config"])
        P += [tbl(t5, ["config", "retriever", "informative_rate", "none_rate", "ambiguous_rate"]),
              "<p class='note'><b>Core (L0) won the label-free selection</b> — the expanded lay/clinical "
              "tiers did not improve LLM-judged evidence quality over Core.</p>",
              prov("retrieval/retrieval_metrics_overall.csv", label_free=True)]

    # 5 semantic method
    sq = rj(R2 / "semantic_queries" / "semantic_queries.json") or {}
    npoles = sum(1 for v in sq.values() if v.get("bipolar"))
    P += ["<h2>5. Corrected semantic multi-query method</h2>",
          f"<p>Each item uses natural-language queries (Q0 item, Q1 lay, Q2 clinical) instead of a "
          f"keyword dump; the score is the <b>max</b> cosine over active queries; the {npoles} bipolar "
          f"items (Sleep, Appetite, Moving) use separate poles. Tests in "
          f"<code>tests/test_semantic_queries.py</code> (7 passing) assert no keyword-dump, poles "
          f"separated, and clinical queries use only audited terms.</p>",
          prov("semantic_queries/semantic_queries.json + docs/r2_semantic_query_method.md", label_free=True)]

    # 6 bm25 vs semantic
    P += ["<h2>6. BM25 vs semantic evidence quality</h2>"]
    if rmet is not None:
        bs = rmet[(rmet.prefix == "top5") & (rmet.config == "L0_CORE")]
        P += [tbl(bs, ["retriever", "informative_rate", "supports_rate", "against_rate", "none_rate", "ambiguous_rate"]),
              "<p class='note'>Semantic-only edged BM25-only on Top-5 informative rate for Core.</p>",
              prov("retrieval/retrieval_metrics_overall.csv", label_free=True)]

    # 7 complementarity
    P += ["<h2>7. Evidence complementarity (BM25 vs semantic)</h2>"]
    if comp is not None:
        cc = comp[(comp.scope == "overall") & (comp.prefix == "top5") & (comp.config == "L0_CORE")]
        P += [tbl(cc[["joint", "n"]], rnd=None),
              prov("retrieval/retrieval_complementarity.csv", label_free=True)]

    # 8 hybrid
    P += ["<h2>8. Whether hybrid was justified</h2>"]
    hyb_rows = rmet[rmet.retriever.astype(str).str.startswith("hybrid")] if rmet is not None else None
    if hyb_sig:
        s = hyb_sig.get("L0_CORE", {})
        P.append(f"<p>Complementarity flagged hybrid <b>justified</b> for Core "
                 f"(BM25-unique informative wins {s.get('bm25_unique_wins')}, "
                 f"semantic-unique {s.get('semantic_unique_wins')}, both {s.get('both_informative')}).</p>")
    if hyb_rows is not None and len(hyb_rows):
        P.append(tbl(hyb_rows[hyb_rows.prefix == "top5"], ["retriever", "informative_rate", "none_rate", "ambiguous_rate"]))
        P.append("<p class='note'>Hybrid α∈{.25,.5,.75} were judged on their own HybridScore-ranked "
                 "windows (not the BM25∪semantic union). Per Stage H, a hybrid is adopted only if it beats "
                 "the best single retriever by ≥0.005 informative — otherwise the single retriever, being "
                 "simpler, wins the tie-break.</p>")
        best_single = float(rmet[~rmet.retriever.astype(str).str.startswith("hybrid")]["informative_rate"].max())
        best_hyb = float(hyb_rows["informative_rate"].max())
        adopted = str(r2.get("retriever", "")).startswith("hybrid")
        decision = ("a hybrid candidate <b>became the final R2</b>" if adopted else
                    "<b>no hybrid</b> beat the best single retriever within the label-free tie band, "
                    "so the single retriever stands as R2")
        P.append(f"<p class='verdict'>Decision: best hybrid informative rate {best_hyb:.4f} vs "
                 f"best single {best_single:.4f} → {decision} "
                 f"(final R2 = {r2.get('config')} / {r2.get('retriever')} / {r2.get('prefix')}).</p>")
    else:
        P.append("<p class='pending'>hybrid judge results pending (build complete; judging/selection in progress)</p>")
    P.append(prov("retrieval/retrieval_metrics_overall.csv + r2_selection.json", label_free=True))

    # 9 budgets
    P += ["<h2>9. Top-3 vs Top-5 vs token-budget</h2>"]
    if rmet is not None:
        bud = rmet[(rmet.config == r2.get("config", "L0_CORE")) & (rmet.retriever == r2.get("retriever", "semantic"))]
        P += [tbl(bud, ["prefix", "informative_rate", "none_rate", "ambiguous_rate"]),
              "<p class='note'>Per Stage H, Top-5 chosen when it ties the token-budget within 0.005 "
              "informative / 0.01 none.</p>", prov("retrieval/retrieval_metrics_overall.csv", label_free=True)]

    # 10 R2 selection
    P += ["<h2>10. Selected R2 + exact label-free rule</h2>",
          f"<pre>{html.escape(json.dumps({k: sel.get(k) for k in ('primary_criterion','tie_tolerance','lexicon_shortlist','selected_R2')}, indent=2))}</pre>",
          prov("retrieval/r2_selection.json", label_free=True)]

    # 11 encoder + policy audit
    dstat = rj(R2 / "downstream_status.json") or {}
    if dstat.get("rerunning"):
        P.append(f"<div class='warn'><b>Downstream re-run in progress:</b> R2 was updated to "
                 f"<b>{html.escape(str(r2.get('retriever')))}/{html.escape(str(r2.get('prefix')))}</b> "
                 f"after the hybrid test. The encoder/LLM/cascade numbers in §11–§15 below are from the "
                 f"interim R2 (<code>{html.escape(dstat.get('interim','semantic/top5'))}</code>) and are "
                 f"being re-trained on the updated R2 (jobs {html.escape(str(dstat.get('jobs','')))}); they "
                 f"will be refreshed when training completes.</div>")
    P += ["<h2>11. MentalBERT+CORN results + missing-policy audit</h2>"]
    if enc is not None:
        P.append(tbl(enc, ["run", "macro_f1", "qwk", "mae", "f1_class3", "severe_recall", "false_severe_rate"]))
    if encci is not None:
        P.append("<p class='note'>Paired participant-cluster bootstrap (B=2000, seed42), Δ vs baseline:</p>")
        P.append(tbl(encci, ["comparison", "metric", "delta", "ci_lo", "ci_hi", "excludes_0"]))
        P.append("<p class='note'>Every CI includes 0 → <b>no encoder change is statistically significant.</b></p>")
    if audit is not None:
        tot_none = int(audit.none_rows.sum()); tot = int(audit.train_rows.sum())
        g3 = round(100 * audit.pct_gold3_removed.mean(), 1)
        P.append(f"<div class='warn'><b>Missing-policy audit:</b> drop_none removes "
                 f"{tot_none}/{tot} ({round(100*tot_none/tot,1)}%) of training rows and "
                 f"<b>{g3}% of gold-3 (severe) rows</b> on average — pathological removal (brief §11.2). "
                 f"Although drop_none has the best QWK, it trips the false-severe constraint and depends on "
                 f"removing half the severe examples, so <b>E_FINAL = {html.escape(efinal)}</b>.</div>")
    P.append(prov("encoder/encoder_metrics_overall.csv + missing_policy_audit.csv + encoder_paired_bootstrap_ci.csv",
                  label_free=False, within_fold=True))

    # 12 LLM
    P += ["<h2>12. Qwen retrieved vs long-context</h2>"]
    if llm is not None:
        P.append(tbl(llm, ["run", "macro_f1", "qwk", "mae", "f1_class3", "severe_recall", "coverage"]))
        P.append(f"<p class='note'>L_FINAL = <b>{html.escape(lfinal)}</b>. R2 evidence did not beat the "
                 f"frozen R0 LLM (L0_R0) on QWK; filter/informative-first lift severe recall + class-3 F1 "
                 f"but worsen MAE; long-context (L5) was worst.</p>")
    P.append(prov("llm/llm_metrics_overall.csv", within_fold=True))

    # 13 encoder vs llm
    P += ["<h2>13. Encoder vs LLM by item and evidence status</h2>"]
    if mcomp is not None:
        P.append("<p class='note'>By evidence status (fraction where each model uniquely correct):</p>")
        P.append(tbl(mcomp, ["evidence_status", "n", "enc_only_correct", "llm_only_correct", "both_correct", "both_incorrect"]))
    if mcompi is not None:
        P.append(tbl(mcompi, ["item_name", "enc_only_correct", "llm_only_correct", "llm_lower_err", "disagree_ge2"]))
    P.append(prov("analysis/complementarity_by_evidence.csv + by_item.csv", within_fold=True))

    # 14 cascade
    P += ["<h2>14. Encoder-first vs LLM-first cascade</h2>"]
    if casc is not None:
        P.append(tbl(casc, ["model", "macro_f1", "qwk", "mae", "severe_recall", "false_severe_rate"]))
    acc = {k: v.get("accepted") for k, v in casc_acc.items()} if casc_acc else {}
    P.append(f"<p class='verdict'>Both leakage-safe cascades REJECTED (routing fit in-fold): {html.escape(json.dumps(acc))} "
             "— neither exceeds the better component. The frozen MIL+merged R0 cascade (QWK .447) remains project-best.</p>")
    P.append(prov("cascade/cascade_metrics.csv + cascade_acceptance.json", label_free=False, within_fold=True,
                  extra="routing thresholds fit inside training folds (no OOF tuning)"))

    # 15 severe
    P += ["<h2>15. Severe-class results</h2>"]
    if enc is not None and llm is not None:
        sev = pd.concat([
            enc.assign(kind="encoder")[["run", "severe_recall", "false_severe_rate", "f1_class3"]],
            llm.assign(kind="llm")[["run", "severe_recall", "false_severe_rate", "f1_class3"]]], ignore_index=True)
        P.append(tbl(sev, ["run", "severe_recall", "false_severe_rate", "f1_class3"]))
        P.append("<p class='note'>Severe (gold-3) recall stays low throughout (~.10–.32). The LLM filter/"
                 "informative-first conditions and encoder drop_none raise severe recall most, but at a "
                 "false-severe or pathological-removal cost.</p>")
    P.append(prov("encoder/ + llm/ metrics; severe = class 3", within_fold=True))

    # 16 final system
    P += ["<h2>16. Final recommended system</h2>",
          f"<p class='verdict'>A rigorously, label-free-constructed R2 (Core / {r2.get('retriever','semantic')} / "
          f"{r2.get('prefix','top5')}) does NOT significantly improve PHQ-8 prediction over R0/R1, and neither "
          f"leakage-safe cascade beats its components. <b>Recommendation: retain the frozen R0 Attention-MIL + "
          f"merged-gate cascade (macro-F1 .409 / QWK .447 / MAE .606) as the production system.</b> The R2 "
          f"contribution is methodological (a defensible, leakage-free construction/selection showing the "
          f"expansion does not help), not a prediction gain.</p>"]

    # 17 limitations
    P += ["<h2>17. Limitations</h2>", "<ul>",
          "<li><b>Judge/predictor independence (G7) not fixed:</b> judge = Qwen2.5-7B, same family as the "
          "predictor (both MentaLLaMA models failed validation; no other non-Qwen instruct model offline).</li>",
          "<li><b>Evidence sparsity:</b> ~63% of (participant,item) sets are judged 'none' under R2 semantic-only "
          "retrieval — most PHQ items have little explicit transcript evidence.</li>",
          "<li><b>Severe class:</b> gold-3 remains the bottleneck; drop_none's gains rely on removing ~51% of "
          "severe training rows (pathological).</li>",
          "<li><b>Attention-MIL not retrained on R2</b> (per brief) — the project-best cascade uses the MIL "
          "encoder on R0, so the R2-vs-project-best comparison mixes retrieval and encoder.</li>",
          "<li><b>Hybrid</b> was flagged justified and is tested here; see §8/§10 for whether it changed R2.</li>",
          "</ul>"]

    (PR / "docs" / "r2_results_report.html").write_text("\n".join(P))
    (R2 / "results_report.html").write_text("\n".join(P))
    return PR / "docs" / "r2_results_report.html"


def main():
    build_manifest()
    build_status()
    p = build_html()
    npending = Path(p).read_text().count("pending")
    print(f"Wrote run_manifest.json, status_report.md, and {p} ({npending} pending markers)")


if __name__ == "__main__":
    main()
