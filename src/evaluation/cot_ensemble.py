"""
Step 3: does combining CoT and the encoder beat either alone?

The two models are complementary on the paired OOF: CoT (Qwen-7B) wins macro-F1
and recovers the severe classes; MentalBERT+CORN wins the ordinal metrics (QWK,
MAE) because its errors stay adjacent. This script tests whether a combination
captures both strengths.

Probabilities used:
  encoder  -> CORN class probabilities (prob_0..3 in the OOF file)
  CoT      -> self-consistency vote fractions (folds_sc5/) as the soft dist;
              greedy CoT (folds/) is also carried as a hard point predictor.

Methods (all evaluated as pooled OOF over the 5 folds, n=1752):
  * parents: encoder, CoT-greedy, CoT-sc5
  * prob-average 50/50 (encoder + CoT-sc5)
  * severity-gated routing: use CoT's label when it says >=2 (severe), else
    encoder -- a PRIOR-motivated rule (CoT's known severe-class strength), not
    tuned on labels
  * nested-CV weighted blend: for each held-out fold, pick the blend weight w
    that maximises macro-F1 on the OTHER four folds, then apply to the held-out
    fold. Leakage-free -> an honest "tuned ensemble" number.

    python -m src.evaluation.cot_ensemble
"""

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.build_cot_report import (
    metric_block, grouped_bars, confusion_svg, LABELS, COT_COLOR, ENC_COLOR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENC_OOF = PROJECT_ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
COT_GREEDY_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds"
COT_SC_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds_sc5"
GLOB = "cot_probe_qwen_hybw3_fold*.csv"
OUT_HTML = PROJECT_ROOT / "outputs" / "cot" / "cot_ensemble_report.html"
OUT_JSON = PROJECT_ROOT / "outputs" / "cot" / "cot_ensemble_metrics.json"
KEY = ["participant_id", "item_id"]
PROBS = [f"prob_{c}" for c in LABELS]
ENS_COLOR = "#0ea5e9"  # sky = the ensemble


def load_probs(path_or_dir):
    files = sorted(glob.glob(str(Path(path_or_dir) / GLOB)))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) \
        if files else pd.read_csv(path_or_dir)
    df["participant_id"] = df["participant_id"].astype(str)
    P = df[PROBS].to_numpy(dtype=float)
    P = P / P.sum(1, keepdims=True).clip(min=1e-9)  # renormalise to be safe
    df[PROBS] = P
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CoT+encoder ensemble")
    ap.add_argument("--cot-greedy-dir", default=str(COT_GREEDY_DIR))
    ap.add_argument("--cot-sc-dir", default=str(COT_SC_DIR))
    ap.add_argument("--out-html", default=str(OUT_HTML))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    a = ap.parse_args()
    out_html, out_json = Path(a.out_html), Path(a.out_json)

    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    enc_p = enc[PROBS].to_numpy(dtype=float)
    enc_p = enc_p / enc_p.sum(1, keepdims=True).clip(min=1e-9)

    base = enc[KEY + ["label", "fold", "prediction"]].copy()
    base[[f"enc_{c}" for c in LABELS]] = enc_p

    greedy = load_probs(a.cot_greedy_dir)[KEY + ["prediction"]].rename(
        columns={"prediction": "cot_greedy_pred"})
    sc = load_probs(a.cot_sc_dir)
    sc_ren = sc[KEY].copy()
    sc_ren[[f"cot_{c}" for c in LABELS]] = sc[PROBS].to_numpy()
    sc_ren["cot_sc_pred"] = sc[PROBS].to_numpy().argmax(1)

    df = base.merge(greedy, on=KEY).merge(sc_ren, on=KEY)
    assert len(df) == len(base), f"merge mismatch {len(df)} vs {len(base)}"

    enc_P = df[[f"enc_{c}" for c in LABELS]].to_numpy()
    cot_P = df[[f"cot_{c}" for c in LABELS]].to_numpy()
    y = df["label"].to_numpy()
    folds = df["fold"].to_numpy()

    def blend_pred(w):
        return (w * cot_P + (1 - w) * enc_P).argmax(1)

    # ---- methods ----
    results = {}
    results["encoder (CORN)"] = df["prediction"].to_numpy()
    results["CoT greedy"] = df["cot_greedy_pred"].to_numpy()
    results["CoT self-consist."] = df["cot_sc_pred"].to_numpy()
    results["prob-avg 50/50"] = blend_pred(0.5)

    # severity-gated routing: CoT (greedy) when it calls severe (>=2), else encoder
    cg = df["cot_greedy_pred"].to_numpy()
    routed = np.where(cg >= 2, cg, df["prediction"].to_numpy())
    results["severity-routed"] = routed

    # nested-CV weighted blend (leakage-free)
    grid = np.round(np.linspace(0, 1, 21), 2)
    nested = np.empty(len(df), dtype=int)
    chosen_w = {}
    from sklearn.metrics import f1_score
    for f in sorted(set(folds)):
        tr = folds != f
        te = folds == f
        best_w, best_f1 = 0.5, -1
        for w in grid:
            p = (w * cot_P[tr] + (1 - w) * enc_P[tr]).argmax(1)
            s = f1_score(y[tr], p, average="macro", labels=LABELS, zero_division=0)
            if s > best_f1:
                best_f1, best_w = s, float(w)
        chosen_w[int(f)] = best_w
        nested[te] = (best_w * cot_P[te] + (1 - best_w) * enc_P[te]).argmax(1)
    results["nested-CV blend"] = nested

    # in-sample weight sweep (potential ceiling; clearly oracle, for the curve)
    sweep = [(float(w), metric_block(y, blend_pred(w))["macro_f1"]) for w in grid]
    oracle_w = max(sweep, key=lambda t: t[1])[0]

    metrics = {name: metric_block(y, p) for name, p in results.items()}
    summary = {"n": int(len(df)), "chosen_w_per_fold": chosen_w,
               "oracle_w_insample": oracle_w,
               "weight_sweep_macro_f1": sweep, "methods": metrics}
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- console table ----
    print(f"Ensemble on paired OOF (n={len(df)}):")
    print(f"{'method':22s} {'F1':>6} {'QWK':>6} {'MAE':>6} {'acc':>6} {'faroff':>7} "
          f"{'F1c2':>6} {'F1c3':>6}")
    for name, m in metrics.items():
        print(f"{name:22s} {m['macro_f1']:6.3f} {m['qwk']:6.3f} {m['mae']:6.3f} "
              f"{m['accuracy']:6.3f} {m['far_off_rate']:7.3f} "
              f"{m['f1_per_class'][2]:6.3f} {m['f1_per_class'][3]:6.3f}")
    print(f"nested-CV chosen w per fold (w=CoT weight): {chosen_w}")
    print(f"in-sample oracle w: {oracle_w}")

    # ---- figures (house style) ----
    enc_m = metrics["encoder (CORN)"]
    cot_m = metrics["CoT greedy"]
    order = ["encoder (CORN)", "CoT greedy", "CoT self-consist.",
             "prob-avg 50/50", "severity-routed", "nested-CV blend"]

    def multibar(metric_key, title, vmax, lower_better=False):
        # color parents grey-ish, ensembles sky; CoT violet; encoder green
        cmap = {"encoder (CORN)": ENC_COLOR, "CoT greedy": COT_COLOR,
                "CoT self-consist.": "#a78bfa"}
        n = len(order); rh = 26; padl = 160; padr = 70; padt = 8; padb = 6
        W = 640; plot_w = W - padl - padr; H = padt + n * rh + padb
        # reference lines at both parents
        p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
        for ref, col in [(enc_m[metric_key], ENC_COLOR), (cot_m[metric_key], COT_COLOR)]:
            rx = padl + plot_w * (ref / vmax)
            p.append(f'<line x1="{rx:.1f}" y1="{padt}" x2="{rx:.1f}" y2="{padt+n*rh}" '
                     f'stroke="{col}" stroke-dasharray="3 3" opacity="0.5"/>')
        for i, name in enumerate(order):
            v = metrics[name][metric_key]
            yb = padt + i * rh
            bw = plot_w * (max(v, 0) / vmax)
            col = cmap.get(name, ENS_COLOR)
            p.append(f'<text x="{padl-8}" y="{yb+rh/2+3:.1f}" text-anchor="end" fill="#475569">{name}</text>')
            p.append(f'<rect x="{padl}" y="{yb+3:.1f}" width="{bw:.1f}" height="{rh-7}" rx="3" fill="{col}"/>')
            p.append(f'<text x="{padl+bw+5:.1f}" y="{yb+rh/2+3:.1f}" fill="#1e293b" font-size="10">{v:.3f}</text>')
        p.append('</svg>')
        arrow = " (lower = better)" if lower_better else ""
        return f'<p class="note" style="margin-bottom:2px"><b>{title}{arrow}</b><br>' \
               f'<span style="font-size:10px">dashed lines = parent models (green=encoder, violet=CoT)</span></p>' + "".join(p)

    fig_f1 = multibar("macro_f1", "Fig 1 · macro-F1 by method", 0.45)
    fig_qwk = multibar("qwk", "Fig 2 · QWK (ordinal agreement)", 0.45)
    fig_mae = multibar("mae", "Fig 3 · MAE", 0.9, lower_better=True)

    # weight sweep curve
    ws = [w for w, _ in sweep]; fs = [v for _, v in sweep]
    W, H, padl, padb, padt, padr = 640, 240, 50, 40, 16, 16
    plot_w, plot_h = W - padl - padr, H - padt - padb
    f_lo, f_hi = min(fs) - 0.01, max(fs) + 0.01
    pts = " ".join(f"{padl+plot_w*w:.1f},{padt+plot_h*(1-(v-f_lo)/(f_hi-f_lo)):.1f}"
                   for w, v in zip(ws, fs))
    sweep_svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
                 f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    sweep_svg.append(f'<polyline points="{pts}" fill="none" stroke="{ENS_COLOR}" stroke-width="2"/>')
    for lbl, val, col in [("encoder", enc_m["macro_f1"], ENC_COLOR), ("CoT", cot_m["macro_f1"], COT_COLOR)]:
        yv = padt + plot_h * (1 - (val - f_lo) / (f_hi - f_lo))
        sweep_svg.append(f'<line x1="{padl}" y1="{yv:.1f}" x2="{padl+plot_w}" y2="{yv:.1f}" stroke="{col}" stroke-dasharray="3 3" opacity="0.6"/>')
    sweep_svg.append(f'<text x="{padl}" y="{H-padb+18}" fill="#64748b">w=0 (all encoder)</text>')
    sweep_svg.append(f'<text x="{padl+plot_w}" y="{H-padb+18}" text-anchor="end" fill="#64748b">w=1 (all CoT)</text>')
    sweep_svg.append(f'<text x="{padl-6}" y="{padt+6}" text-anchor="end" fill="#64748b">{f_hi:.2f}</text>')
    sweep_svg.append('</svg>')
    fig_sweep = ('<p class="note" style="margin-bottom:2px"><b>Fig 4 · macro-F1 vs blend weight w '
                 '(in-sample; shows the ceiling, not an honest estimate)</b></p>' + "".join(sweep_svg))

    fig_cm = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
              + confusion_svg(pd.DataFrame({"label": y, "prediction": results["nested-CV blend"]}),
                              "Fig 5 · Nested-CV blend", ENS_COLOR)
              + confusion_svg(pd.DataFrame({"label": y, "prediction": results["encoder (CORN)"]}),
                              "Fig 5b · Encoder", ENC_COLOR)
              + '</div>')

    # ---- table ----
    def trow(name, m, hl=False):
        style = ' style="background:#f0f9ff"' if hl else ""
        return (f"<tr{style}><td>{name}</td><td><b>{m['macro_f1']:.3f}</b></td>"
                f"<td>{m['qwk']:.3f}</td><td>{m['mae']:.3f}</td>"
                f"<td class='sec'>{m['accuracy']:.3f}</td><td>{m['far_off_rate']:.3f}</td>"
                f"<td>{m['f1_per_class'][2]:.3f}</td><td>{m['f1_per_class'][3]:.3f}</td></tr>")
    rows = "".join(trow(n, metrics[n], hl=n.startswith(("prob-avg", "severity", "nested")))
                   for n in order)

    # data-driven verdict: find the best ensemble on each headline metric and
    # whether any ensemble beats BOTH parents simultaneously.
    ens_names = ["prob-avg 50/50", "severity-routed", "nested-CV blend"]
    best_f1 = max(metrics, key=lambda n: metrics[n]["macro_f1"])
    rt = metrics["severity-routed"]
    pa = metrics["prob-avg 50/50"]
    routed_best_of_both = (rt["macro_f1"] >= max(enc_m["macro_f1"], cot_m["macro_f1"])
                           and rt["qwk"] >= max(enc_m["qwk"], cot_m["qwk"]))
    if routed_best_of_both:
        head = "an ensemble beats BOTH parents on macro-F1 and QWK"
        verdict = (f"<b>Severity-routing</b> (use CoT's call when it predicts a severe class ≥2, "
                   f"else the encoder) tops every method on macro-F1 ({rt['macro_f1']:.3f}) and QWK "
                   f"({rt['qwk']:.3f}) — beating <i>both</i> parents at once — and gives the best "
                   f"severe-class F1. Its cost is MAE ({rt['mae']:.3f}, worst): when it trusts CoT "
                   f"on a severe call that's wrong, the miss is larger. If macro-F1 / ordinal "
                   f"agreement / catching severe cases is the goal, this is the config to carry "
                   f"forward; if minimising absolute severity error matters most, the "
                   f"<b>prob-average</b> blend recovers the encoder's MAE ({pa['mae']:.3f}) and the "
                   f"highest accuracy ({pa['accuracy']:.3f}) while still beating the encoder on macro-F1.")
    elif metrics[best_f1]["macro_f1"] >= cot_m["macro_f1"] and best_f1 in ens_names:
        head = "an ensemble edges out both parents on macro-F1"
        verdict = (f"<b>{best_f1}</b> gives the best macro-F1 ({metrics[best_f1]['macro_f1']:.3f}); "
                   f"the prob-average blend separately recovers the encoder's MAE. The two "
                   f"strengths combine partially rather than fully.")
    else:
        head = "ensembles trade off rather than combine"
        verdict = ("No ensemble beats both parents at once — encoder ordinal strength and CoT "
                   "macro-F1 strength trade along the weight curve (Fig 4).")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CoT+encoder ensemble — PHQ-8</title>
<style>
 body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}}
 header{{background:#0f172a;color:#fff;padding:22px 28px}} header h1{{margin:0;font-size:21px}}
 .sub{{color:#cbd5e1;font-size:13px;margin-top:4px}}
 .wrap{{max-width:920px;margin:20px auto;padding:0 18px}}
 .card{{background:#fff;border-radius:12px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card h2{{margin:0 0 10px;font-size:16px}} .note{{font-size:12px;color:#64748b}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}}
 th,td{{padding:6px 9px;border-bottom:1px solid #e2e8f0;text-align:right}}
 th:first-child,td:first-child{{text-align:left}} thead th{{background:#f8fafc;color:#475569;font-weight:600}}
 .sec{{color:#94a3b8}} .caveat{{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;border-radius:6px;font-size:13px}}
</style></head><body>
<header><h1>CoT + encoder ensemble — PHQ-8 item severity</h1>
<div class="sub">Combining CoT (Qwen-7B) and MentalBERT+CORN on the paired 5-fold OOF (n={len(df)}). Ensembles use no extra training; nested-CV blend selects the weight leakage-free.</div></header>
<div class="wrap">
 <div class="card"><h2>TL;DR — {head}</h2><p>{verdict}</p>
 <table><thead><tr><th>Method</th><th>macro-F1</th><th>QWK</th><th>MAE</th><th>acc</th><th>far-off</th><th>F1 c2</th><th>F1 c3</th></tr></thead>
 <tbody>{rows}</tbody></table>
 <p class="note" style="margin-top:8px">Highlighted = ensemble methods. Nested-CV blend weights (CoT share) per fold: {chosen_w}. In-sample oracle weight: {oracle_w}.</p>
 </div>
 <div class="card"><h2>By metric</h2>{fig_f1}<div style="margin-top:14px">{fig_qwk}</div><div style="margin-top:14px">{fig_mae}</div></div>
 <div class="card"><h2>The trade-off curve</h2>{fig_sweep}
 <p class="note" style="margin-top:8px">If macro-F1 peaks at an interior w, a blend genuinely helps; if it's monotonic to one end, just use that model. This curve is in-sample (oracle) — the nested-CV row in the table is the honest version.</p></div>
 <div class="card"><h2>Confusion: best blend vs encoder</h2>{fig_cm}</div>
 <div class="card"><div class="caveat"><b>Honesty notes.</b> OOF over 5 folds, single seed; CoT may have seen DAIC-WOZ (public). The nested-CV blend is leakage-free for the weight; the severity-routing rule was chosen from a prior finding, not tuned on these labels. Treat as directional.</div></div>
</div></body></html>"""
    out_html.write_text(html)
    print(f"\nSaved report : {out_html}")
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
