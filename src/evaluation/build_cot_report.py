"""
Feasibility report: few-shot CoT (Qwen2.5-7B) vs the best encoder
(MentalBERT + CORN), evaluated on the SAME fold-1 rows of Hybrid-W3.

Reads:
  outputs/cot/cot_probe_qwen_hybw3_fold1.csv        (CoT predictions)
  outputs/cv/oof_predictions_ctxm_corn_hybw3.csv    (encoder OOF; fold 1 = slice)
Writes:
  outputs/cot/cot_feasibility_report.html
  outputs/cot/cot_feasibility_metrics.json

Inline-SVG figures, matching the house style of build_context_report.py.
This is a go/no-go probe on ONE fold (n=352) -- treat numbers as directional,
not as a leaderboard result.

    python -m src.evaluation.build_cot_report
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score, confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COT_PRED = PROJECT_ROOT / "outputs" / "cot" / "cot_probe_qwen_hybw3_fold1.csv"
FOLDS_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds"
FOLD_GLOB = "cot_probe_qwen_hybw3_fold*.csv"
ENC_OOF = PROJECT_ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUT_HTML = PROJECT_ROOT / "outputs" / "cot" / "cot_feasibility_report.html"
OUT_JSON = PROJECT_ROOT / "outputs" / "cot" / "cot_feasibility_metrics.json"
LABELS = [0, 1, 2, 3]

COT_COLOR = "#7c3aed"      # violet = the new CoT model
ENC_COLOR = "#16a34a"      # green  = the established best encoder


# --------------------------------------------------------------------------- metrics

def metric_block(y, p):
    y, p = np.asarray(y), np.asarray(p)
    err = np.abs(y - p)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)),
        "mae": float(mean_absolute_error(y, p)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS)),
        "f1_per_class": [float(x) for x in
                         f1_score(y, p, average=None, labels=LABELS, zero_division=0)],
        "exact_rate": float((err == 0).mean()),
        "off_by_one_rate": float((err == 1).mean()),
        "far_off_rate": float((err >= 2).mean()),
    }


def per_item(df, name_lookup):
    out = {}
    for iid in range(1, 9):
        g = df[df.item_id == iid]
        if len(g) == 0:
            continue
        out[iid] = {
            "name": name_lookup[iid],
            "macro_f1": float(f1_score(g.label, g.prediction, average="macro",
                                       labels=LABELS, zero_division=0)),
            "mae": float(mean_absolute_error(g.label, g.prediction)),
        }
    return out


# --------------------------------------------------------------------------- SVG

def grouped_bars(rows, title, vmax, lower_better=False):
    """rows: list of (label, cot_val, enc_val). Two bars per row."""
    n = len(rows); rh = 34; padl = 150; padr = 64; padt = 28; padb = 6
    W = 640; plot_w = W - padl - padr; H = padt + n * rh + padb
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    # legend
    p.append(f'<rect x="{padl}" y="6" width="11" height="11" fill="{COT_COLOR}"/>'
             f'<text x="{padl+15}" y="15" fill="#64748b">CoT (Qwen-7B)</text>')
    p.append(f'<rect x="{padl+150}" y="6" width="11" height="11" fill="{ENC_COLOR}"/>'
             f'<text x="{padl+165}" y="15" fill="#64748b">MentalBERT+CORN</text>')
    for i, (lab, cv, ev) in enumerate(rows):
        y = padt + i * rh
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        for k, (v, col) in enumerate([(cv, COT_COLOR), (ev, ENC_COLOR)]):
            yy = y + 3 + k * (rh / 2 - 2)
            bw = plot_w * (max(v, 0) / vmax)
            p.append(f'<rect x="{padl}" y="{yy:.1f}" width="{bw:.1f}" height="{rh/2-4}" rx="2" fill="{col}"/>')
            p.append(f'<text x="{padl+bw+5:.1f}" y="{yy+rh/2-6:.1f}" fill="#1e293b" font-size="10">{v:.3f}</text>')
    p.append('</svg>')
    arrow = " (lower = better)" if lower_better else ""
    return f'<p class="note" style="margin-bottom:2px"><b>{title}{arrow}</b></p>' + "".join(p)


def confusion_svg(df, title, accent):
    cm = confusion_matrix(df.label, df.prediction, labels=LABELS)
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)
    cell = 50; x0, y0 = 80, 40; W, H = 320, 270
    r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    p.append(f'<text x="{x0+2*cell}" y="16" text-anchor="middle" fill="#64748b" font-size="10">predicted</text>')
    for j in range(4):
        p.append(f'<text x="{x0+j*cell+cell/2}" y="{y0-5}" text-anchor="middle" fill="#475569">{j}</text>')
    for i in range(4):
        p.append(f'<text x="{x0-8}" y="{y0+i*cell+cell/2+4}" text-anchor="end" fill="#475569">{i}</text>')
        for j in range(4):
            a = 0.10 + 0.85 * rn[i, j]
            tc = "#fff" if rn[i, j] > 0.5 else "#1e293b"
            x = x0 + j * cell; y = y0 + i * cell
            p.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" rx="3" fill="rgba({r},{g},{b},{a:.2f})"/>')
            p.append(f'<text x="{x+(cell-3)/2}" y="{y+(cell-3)/2+4}" text-anchor="middle" fill="{tc}" font-weight="600">{cm[i,j]}</text>')
    p.append('</svg>')
    return (f'<div style="flex:1;min-width:280px"><p class="note" style="margin-bottom:2px">'
            f'<b>{title}</b></p>' + "".join(p) + '</div>')


def stacked_error_svg(cot_m, enc_m, title):
    """Error structure: exact / off-by-one / far-off as stacked proportion bars."""
    W = 640; padl = 150; padr = 20; padt = 28; rh = 40; H = padt + 2 * rh + 6
    seg = [("exact", "exact_rate", "#16a34a"),
           ("off-by-one", "off_by_one_rate", "#f59e0b"),
           ("far off (≥2)", "far_off_rate", "#dc2626")]
    plot_w = W - padl - padr
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    lx = padl
    for name, key, col in seg:
        p.append(f'<rect x="{lx}" y="6" width="11" height="11" fill="{col}"/>'
                 f'<text x="{lx+15}" y="15" fill="#64748b">{name}</text>')
        lx += 95
    for i, (lab, m) in enumerate([("CoT (Qwen-7B)", cot_m), ("MentalBERT+CORN", enc_m)]):
        y = padt + i * rh; x = padl
        p.append(f'<text x="{padl-8}" y="{y+rh/2:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        for _, key, col in seg:
            w = plot_w * m[key]
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{rh-12}" fill="{col}"/>')
            if w > 30:
                p.append(f'<text x="{x+w/2:.1f}" y="{y+(rh-12)/2+4:.1f}" text-anchor="middle" '
                         f'fill="#fff" font-size="10">{m[key]*100:.0f}%</text>')
            x += w
    p.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p)


def per_item_bars(cot_pi, enc_pi, metric, vmax, title, lower_better=False):
    items = []
    for iid in range(1, 9):
        if iid in cot_pi and iid in enc_pi:
            items.append((enc_pi[iid]["name"], cot_pi[iid][metric], enc_pi[iid][metric]))
    items.sort(key=lambda t: t[2], reverse=not lower_better)
    return grouped_bars(items, title, vmax, lower_better=lower_better)


# --------------------------------------------------------------------------- HTML

def main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Build CoT-vs-encoder feasibility report")
    ap.add_argument("--folds-dir", default=str(FOLDS_DIR))
    ap.add_argument("--out-html", default=str(OUT_HTML))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    a = ap.parse_args()
    folds_dir = Path(a.folds_dir)
    out_html, out_json = Path(a.out_html), Path(a.out_json)

    # Prefer the per-fold files (pooled OOF); fall back to the single fold-1 file.
    fold_files = sorted(glob.glob(str(folds_dir / FOLD_GLOB)))
    if fold_files:
        cot = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
        src_desc = f"{len(fold_files)} fold file(s)"
    elif COT_PRED.exists():
        cot = pd.read_csv(COT_PRED)
        src_desc = COT_PRED.name
    else:
        raise SystemExit(f"No CoT predictions found in {FOLDS_DIR} or {COT_PRED}")
    cot["participant_id"] = cot["participant_id"].astype(str)

    enc_all = pd.read_csv(ENC_OOF)
    enc_all["participant_id"] = enc_all["participant_id"].astype(str)

    # Attach the fold id from the encoder OOF, then restrict the encoder to the
    # exact folds CoT has run -> always an apples-to-apples paired comparison.
    key = ["participant_id", "item_id"]
    cot = cot.merge(enc_all[key + ["fold"]], on=key, how="left")
    folds_present = sorted(int(f) for f in cot["fold"].dropna().unique())
    enc = enc_all[enc_all["fold"].isin(folds_present)].copy()

    paired = cot.merge(enc[key + ["prediction"]], on=key, suffixes=("", "_enc"))
    assert len(paired) == len(cot), f"pairing dropped rows: {len(paired)} vs {len(cot)}"

    name_lookup = {iid: cot[cot.item_id == iid].item_name.iloc[0]
                   for iid in cot.item_id.unique()}

    cot_m = metric_block(cot.label, cot.prediction)
    enc_m = metric_block(enc.label, enc.prediction)
    cot_pi = per_item(cot, name_lookup)
    enc_pi = per_item(enc, name_lookup)
    agree = float((paired.prediction == paired.prediction_enc).mean())

    # ---- per-fold spread (paired) ------------------------------------------
    KEYS = ("macro_f1", "qwk", "mae")
    perfold = []
    for f in folds_present:
        c, e = cot[cot.fold == f], enc[enc.fold == f]
        cm, em = metric_block(c.label, c.prediction), metric_block(e.label, e.prediction)
        perfold.append({"fold": f, "n": int(len(c)),
                        "cot": {k: cm[k] for k in KEYS},
                        "enc": {k: em[k] for k in KEYS}})
    n_folds = len(perfold)

    def mstd(side, k):
        a = np.array([pf[side][k] for pf in perfold])
        return float(a.mean()), float(a.std(ddof=1)) if n_folds > 1 else 0.0

    def paired_delta(k, lower=False):
        d = np.array([pf["cot"][k] - pf["enc"][k] for pf in perfold])
        wins = int((d < 0).sum()) if lower else int((d > 0).sum())
        sd = float(d.std(ddof=1)) if n_folds > 1 else 0.0
        # simple paired t (informational only; n_folds is small)
        t = float(d.mean() / (sd / np.sqrt(n_folds))) if sd > 0 else float("nan")
        return float(d.mean()), sd, wins, t

    summary = {"source": src_desc, "folds": folds_present, "n": cot_m["n"],
               "agreement_rate": agree, "cot": cot_m, "encoder": enc_m,
               "per_fold": perfold, "cot_per_item": cot_pi, "encoder_per_item": enc_pi}
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- figures ----
    fig_head = grouped_bars(
        [("macro-F1", cot_m["macro_f1"], enc_m["macro_f1"]),
         ("QWK (ordinal)", cot_m["qwk"], enc_m["qwk"]),
         ("accuracy", cot_m["accuracy"], enc_m["accuracy"])],
        "Fig 1 · Headline metrics — CoT vs best encoder (higher = better)", 0.5)
    fig_mae = grouped_bars([("MAE", cot_m["mae"], enc_m["mae"])],
                           "Fig 2 · Mean absolute error", 1.0, lower_better=True)
    fig_pc = grouped_bars(
        [(f"class {c} F1", cot_m["f1_per_class"][c], enc_m["f1_per_class"][c])
         for c in LABELS],
        "Fig 3 · Per-severity-class F1 (class 2/3 = the hard severe classes)", 0.7)
    fig_cm = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
              + confusion_svg(cot, "Fig 4a · CoT (Qwen-7B)", COT_COLOR)
              + confusion_svg(enc, "Fig 4b · MentalBERT+CORN", ENC_COLOR)
              + '</div>')
    fig_err = stacked_error_svg(cot_m, enc_m,
                                "Fig 5 · Error structure (off-by-one = adjacent severity)")
    fig_item_f1 = per_item_bars(cot_pi, enc_pi, "macro_f1", 0.6,
                                "Fig 6 · Per-item macro-F1")
    fig_item_mae = per_item_bars(cot_pi, enc_pi, "mae", 1.2,
                                 "Fig 7 · Per-item MAE", lower_better=True)

    def delta(a, b, lower=False):
        d = a - b
        good = (d < 0) if lower else (d > 0)
        col = "#16a34a" if good else "#dc2626"
        return f'<span style="color:{col}">{d:+.3f}</span>'

    verdict_win = (cot_m["macro_f1"] >= enc_m["macro_f1"] - 0.03)
    dmean, dsd, wins, tstat = paired_delta("macro_f1")
    robust = f" Per-fold: CoT wins macro-F1 on {wins}/{n_folds} folds (Δ {dmean:+.3f}±{dsd:.3f})." if n_folds > 1 else ""
    if cot_m["macro_f1"] >= enc_m["macro_f1"]:
        verdict = ("GO — few-shot CoT, with <b>no training</b>, <b>matches or beats</b> the "
                   "fully-trained best encoder on pooled macro-F1." + robust +
                   " Worth investing in self-consistency then teacher-generation + distillation.")
    elif verdict_win:
        verdict = ("PROMISING — CoT lands within ~0.03 macro-F1 of a model trained on "
                   "~1,400 examples, with <b>no training</b>." + robust +
                   " Better exemplars / self-consistency / distillation are likely to close the gap.")
    else:
        verdict = ("WEAK — out-of-the-box CoT trails the encoder by more than 0.03 macro-F1." + robust +
                   " Before more LLM effort, check reasoning quality vs. the noisy keyword evidence.")

    # ---- per-fold spread card (only meaningful with >1 fold) ----
    if n_folds > 1:
        pf_body = ""
        for pf in perfold:
            d = pf["cot"]["macro_f1"] - pf["enc"]["macro_f1"]
            col = "#16a34a" if d > 0 else "#dc2626"
            pf_body += (f"<tr><td>fold {pf['fold']}</td><td class='sec'>{pf['n']}</td>"
                        f"<td>{pf['cot']['macro_f1']:.3f}</td><td>{pf['enc']['macro_f1']:.3f}</td>"
                        f"<td style='color:{col}'>{d:+.3f}</td>"
                        f"<td>{pf['cot']['qwk']:.3f}</td><td>{pf['enc']['qwk']:.3f}</td>"
                        f"<td>{pf['cot']['mae']:.3f}</td><td>{pf['enc']['mae']:.3f}</td></tr>")
        f1c, f1e = mstd("cot", "macro_f1"), mstd("enc", "macro_f1")
        qc, qe = mstd("cot", "qwk"), mstd("enc", "qwk")
        mc, me = mstd("cot", "mae"), mstd("enc", "mae")
        dq = paired_delta("qwk"); dmae = paired_delta("mae", lower=True)
        perfold_card = f"""
 <div class="card"><h2>Fold-by-fold validation (the point of this run)</h2>
 <table><thead><tr><th>Fold</th><th>n</th><th>F1 CoT</th><th>F1 enc</th><th>ΔF1</th>
 <th>QWK CoT</th><th>QWK enc</th><th>MAE CoT</th><th>MAE enc</th></tr></thead>
 <tbody>{pf_body}
 <tr style="font-weight:600;border-top:2px solid #cbd5e1"><td>mean±std</td><td class="sec">—</td>
 <td>{f1c[0]:.3f}±{f1c[1]:.3f}</td><td>{f1e[0]:.3f}±{f1e[1]:.3f}</td>
 <td>{dmean:+.3f}</td><td>{qc[0]:.3f}±{qc[1]:.3f}</td><td>{qe[0]:.3f}±{qe[1]:.3f}</td>
 <td>{mc[0]:.3f}±{mc[1]:.3f}</td><td>{me[0]:.3f}±{me[1]:.3f}</td></tr>
 </tbody></table>
 <p class="note" style="margin-top:8px">Paired across {n_folds} folds: macro-F1 Δ {dmean:+.3f} (CoT wins {wins}/{n_folds}, t={tstat:.2f}),
 QWK Δ {dq[0]:+.3f} (wins {dq[2]}/{n_folds}), MAE Δ {dmae[0]:+.3f} (CoT better on {dmae[2]}/{n_folds}; lower=better).
 Small n_folds → treat t as directional, not a formal test.</p>
 </div>"""
    else:
        perfold_card = ('<div class="card"><div class="caveat"><b>Single fold only.</b> '
                        'Run <code>run_cot_probe_folds.sbatch</code> for the full 5-fold spread.</div></div>')

    scope = (f"pooled out-of-fold over {n_folds} folds (n={cot_m['n']})" if n_folds > 1
             else f"fold {folds_present[0]} only (n={cot_m['n']})")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoT feasibility probe — PHQ-8</title>
<style>
 body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}}
 header{{background:#0f172a;color:#fff;padding:22px 28px}}
 header h1{{margin:0;font-size:21px}} .sub{{color:#cbd5e1;font-size:13px;margin-top:4px}}
 .meta{{margin-top:8px;font-size:12px;color:#94a3b8;display:flex;gap:16px;flex-wrap:wrap}}
 .wrap{{max-width:920px;margin:20px auto;padding:0 18px}}
 .card{{background:#fff;border-radius:12px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card h2{{margin:0 0 10px;font-size:16px}}
 .note{{font-size:12px;color:#64748b}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}}
 th,td{{padding:6px 9px;border-bottom:1px solid #e2e8f0;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:#f8fafc;color:#475569;font-weight:600}}
 .sec{{color:#94a3b8}}
 .pill{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}}
 .caveat{{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;border-radius:6px;font-size:13px}}
</style></head><body>
<header><h1>Chain-of-Thought feasibility probe — PHQ-8 item severity</h1>
<div class="sub">Few-shot CoT (Qwen2.5-7B-Instruct, no training) vs the best encoder (MentalBERT + CORN), same Hybrid-W3 evidence, paired on identical rows · {scope}</div>
<div class="meta"><span>🤖 Qwen2.5-7B-Instruct</span><span>🆚 MentalBERT+CORN</span><span>🖥️ RTX 3090 (SLURM)</span><span>folds: {",".join(str(f) for f in folds_present)}</span></div></header>
<div class="wrap">

 <div class="card"><h2>TL;DR — {('CoT is competitive' if verdict_win else 'encoder still ahead')}</h2>
 <p>{verdict}</p>
 <table><thead><tr><th>Metric</th><th>CoT (Qwen-7B)</th><th>MentalBERT+CORN</th><th>Δ (CoT−enc)</th></tr></thead><tbody>
 <tr><td>macro-F1</td><td><b>{cot_m['macro_f1']:.3f}</b></td><td>{enc_m['macro_f1']:.3f}</td><td>{delta(cot_m['macro_f1'],enc_m['macro_f1'])}</td></tr>
 <tr><td>QWK (ordinal)</td><td><b>{cot_m['qwk']:.3f}</b></td><td>{enc_m['qwk']:.3f}</td><td>{delta(cot_m['qwk'],enc_m['qwk'])}</td></tr>
 <tr><td>MAE (lower better)</td><td><b>{cot_m['mae']:.3f}</b></td><td>{enc_m['mae']:.3f}</td><td>{delta(cot_m['mae'],enc_m['mae'],lower=True)}</td></tr>
 <tr><td class="sec">accuracy (weak metric)</td><td class="sec">{cot_m['accuracy']:.3f}</td><td class="sec">{enc_m['accuracy']:.3f}</td><td>{delta(cot_m['accuracy'],enc_m['accuracy'])}</td></tr>
 <tr><td>model agreement</td><td colspan="3">{agree*100:.0f}% of items predicted identically</td></tr>
 </tbody></table></div>
{perfold_card}
 <div class="card"><h2>Headline comparison</h2>{fig_head}<div style="margin-top:14px">{fig_mae}</div></div>

 <div class="card"><h2>Where the signal is</h2>{fig_pc}
 <p class="note" style="margin-top:8px">Class 2 (more-than-half) and class 3 (nearly-every-day) are the starved severe classes that all unweighted encoders collapse on. Whether CoT recovers any of them is the most interesting feasibility signal.</p>
 </div>

 <div class="card"><h2>Confusion matrices (row-normalised shading, counts shown)</h2>{fig_cm}</div>

 <div class="card"><h2>Error structure — is CoT "less wrong"?</h2>{fig_err}
 <p class="note" style="margin-top:8px">Off-by-one = predicted an adjacent severity. The project's near-miss finding is that the bottleneck is the boundary between neighbouring severities; a CoT model that errs by one rather than far off is behaving the way CORN was designed to.</p>
 </div>

 <div class="card"><h2>Per-item breakdown</h2>{fig_item_f1}<div style="margin-top:14px">{fig_item_mae}</div></div>

 <div class="card"><h2>Read this number carefully</h2>
 <div class="caveat"><b>{scope}.</b> {'Pooled over all folds, but greedy single-chain and crafted exemplars — still a feasibility result, not a tuned benchmark.' if n_folds > 1 else 'Single-fold go/no-go probe — wide CI, do not quote as the project result.'} DAIC-WOZ is public, so Qwen may have seen related data — a strong score is not automatically a clean win. Encoder = its OOF slice on the same folds (full 5-fold pooled OOF macro-F1 was 0.348).</div>
 <p class="note" style="margin-top:10px">Next steps if GO/PROMISING: (1) add self-consistency (sample N chains, majority vote) for calibrated probabilities; (2) use a strong teacher to generate rationales over the train folds; (3) distill into a small open model that runs offline on the 3090s.</p>
 </div>

</div></body></html>"""

    out_html.write_text(html)
    print(f"Saved report : {out_html}")
    print(f"Saved metrics: {out_json}")
    print(f"\nCoT     macro-F1 {cot_m['macro_f1']:.3f} | QWK {cot_m['qwk']:.3f} | "
          f"MAE {cot_m['mae']:.3f} | acc {cot_m['accuracy']:.3f}")
    print(f"Encoder macro-F1 {enc_m['macro_f1']:.3f} | QWK {enc_m['qwk']:.3f} | "
          f"MAE {enc_m['mae']:.3f} | acc {enc_m['accuracy']:.3f}")
    print(f"Agreement: {agree*100:.0f}%")


if __name__ == "__main__":
    main()
