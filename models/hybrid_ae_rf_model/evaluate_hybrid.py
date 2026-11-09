#!/usr/bin/env python
"""
Evaluate the hybrid (AE + Random Forest) and the confidence-gated hybrid in the same
two table formats as your RF-vs-XGBoost and AE-vs-IF comparisons.

Nothing is trained or tuned here. It loads:
  models/hybrid_ae_rf/hybrid_rf.joblib       (the 85-feature Random Forest)
  models/autoencoder/best_autoencoder_log1p.keras   (frozen AE)
  results/hybrid_ae_rf/gate_config.json      (tau / p_cut chosen on VALIDATION by confidence_gate.py)

Place in the PROJECT ROOT next to train_hybrid_ae_rf.py and confidence_gate.py.

  python evaluate_hybrid.py --split val     # dry run on validation (check the code works)
  python evaluate_hybrid.py --split test    # final numbers for the report

Outputs (results/hybrid_ae_rf/):
  hybrid_eval_multiclass_<split>.csv   -> "RF vs XGBoost" format (12-class metrics)
  hybrid_eval_binary_<split>.csv       -> "AE vs IF" format (attack vs normal metrics)
  hybrid_eval_raw_<split>.json         -> raw numbers and timing breakdown
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from confidence_gate import CONFIG_FILE, apply_rule
from train_hybrid_ae_rf import (DATA, MOD, RES, fuse, load_feature_model, load_names,
                                load_split, log, save_json)

# Columns of the tables (names from gate_config.json). Override with --configs.
DEFAULT_CONFIGS = ["nogate_fpr0.5pct", "gate_p0.99_fpr0.5pct",
                   "nogate_fpr1.0pct", "gate_p0.99_fpr1.0pct"]


# ============================================================
# 1. Scoring: AE pass + RF probabilities (timed)
# ============================================================
def run_pipeline(rf, fm, X):
    t0 = time.perf_counter()
    Xf, err = fuse(fm, X)
    t_fuse = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = rf.predict_proba(Xf)
    t_rf = time.perf_counter() - t0
    assert int(rf.classes_[0]) == 0, "Column 0 of predict_proba must be BENIGN"
    return {"proba": proba, "err": err, "p_benign": proba[:, 0],
            "rf_pred": rf.classes_[proba.argmax(axis=1)], "t_fuse": t_fuse, "t_rf": t_rf}


# ============================================================
# 2. Variants: stacked RF, and RF + rule/gate
# ============================================================
def multiclass_pred(ctx, classes, flag):
    """RF label, except flows the rule newly flags get the RF's most likely ATTACK class."""
    pred = ctx["rf_pred"].copy()
    extra = flag & (pred == 0)
    if extra.any():
        pred[extra] = classes[1:][ctx["proba"][extra][:, 1:].argmax(axis=1)]
    return pred


def gate_score(ctx, p_cut):
    """
    Continuous score that traces the rule's own ROC curve as tau is swept:
      RF-flagged flows rank first, then gate-eligible flows ranked by AE error, then the rest.
    """
    p, err, rf_flag = ctx["p_benign"], ctx["err"], ctx["rf_pred"] != 0
    mask = np.ones(len(p), dtype=bool) if p_cut is None else p < p_cut
    s = np.zeros(len(p))
    elig = mask & ~rf_flag
    s[elig] = 1.0 + err[elig] / (1.0 + err[elig])        # in (1, 2), monotone in error
    s[rf_flag] = 2.0 + (1.0 - p[rf_flag])                # in [2, 3]
    return s


def make_variant(ctx, classes, cfg=None):
    t0 = time.perf_counter()
    rf_flag = ctx["rf_pred"] != 0
    if cfg is None:                                       # plain stacked RF
        flag, pred = rf_flag, ctx["rf_pred"]
        score = 1.0 - ctx["p_benign"]
    else:
        flag = apply_rule(ctx["p_benign"], ctx["err"], rf_flag, cfg["p_cut"], cfg["tau"])
        pred = multiclass_pred(ctx, classes, flag)
        score = gate_score(ctx, cfg["p_cut"])
    return {"flag": flag, "pred": pred, "score": score, "t_rule": time.perf_counter() - t0}


# ============================================================
# 3. Metrics
# ============================================================
def multiclass_column(y, v, labels, seconds):
    benign, attack = y == 0, y != 0
    pred = v["pred"]
    missed, false_alarms = int((pred[attack] == 0).sum()), int((pred[benign] != 0).sum())
    return {
        "Accuracy (%)": 100 * accuracy_score(y, pred),
        "Macro_Precision (%)": 100 * precision_score(y, pred, labels=labels, average="macro", zero_division=0),
        "Macro_Recall (%)": 100 * recall_score(y, pred, labels=labels, average="macro", zero_division=0),
        "Macro_F1 (%)": 100 * f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "Weighted F1 (%)": 100 * f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
        "FPR (%)": 100 * float((pred[benign] != 0).mean()),
        "Attack Detection Rate (%)": 100 * float((pred[attack] != 0).mean()),
        "MCC (%)": 100 * matthews_corrcoef(y, pred),
        "Missed Attacks / False Alarms": f"{missed:,} / {false_alarms:,}",
        "Inference Latency (us)": 1e6 * seconds / len(y),
    }


def binary_column(y, v, ps_flag, ps_score, seconds):
    yb = (y != 0).astype(int)
    flag = v["flag"]
    tp, fp = int((flag & (yb == 1)).sum()), int((flag & (yb == 0)).sum())
    fn, tn = int((~flag & (yb == 1)).sum()), int((~flag & (yb == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ps_labels = np.concatenate([np.zeros(int((yb == 0).sum())), np.ones(len(ps_score))])
    ps_scores = np.concatenate([v["score"][yb == 0], ps_score])
    return {
        "Accuracy (%)": 100 * (tp + tn) / len(yb),
        "Precision (%)": 100 * precision,
        "Recall / detection rate (%)": 100 * recall,
        "F1-score (%)": 100 * f1,
        "False alarm rate, FAR = FPR (%)": 100 * fp / (fp + tn),
        "True negative rate (%)": 100 * tn / (fp + tn),
        "Average precision, AP (%)": 100 * average_precision_score(yb, v["score"]),
        "ROC-AUC (%)": 100 * roc_auc_score(yb, v["score"]),
        "Held-out PortScan recall (%)": 100 * float(ps_flag.mean()),
        "Held-out PortScan ROC-AUC (%)": 100 * roc_auc_score(ps_labels, ps_scores),
        "Prediction Time(s)": seconds,
    }


def format_table(columns):
    df = pd.DataFrame(columns)

    def fmt(metric, x):
        if isinstance(x, str):
            return x
        if "Time" in metric or "Latency" in metric:
            return f"{x:.2f}"
        if "FPR" in metric or "FAR" in metric or "True negative" in metric:
            return f"{x:.4f}"
        return f"{x:.2f}"

    return df.apply(lambda col: [fmt(m, x) for m, x in zip(df.index, col)])


# ============================================================
# 4. Main
# ============================================================
def label_of(c):
    budget = f"{c['budget'] * 100:.1f}% FPR budget"
    if c["p_cut"] is None:
        return f"Hybrid + plain rule ({budget})"
    return f"Hybrid + gate p<{c['p_cut']} ({budget})"


def main(split, config_names):
    all_cfgs = {c["name"]: c for c in json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
    missing = [n for n in config_names if n not in all_cfgs]
    if missing:
        raise SystemExit(f"Unknown configs {missing}. Available: {list(all_cfgs)}")

    rf = joblib.load(MOD / "hybrid_rf.joblib")
    fm = load_feature_model()
    names = load_names()
    labels = list(range(len(names)))
    classes = rf.classes_

    X, y_s = load_split(split)
    y = y_s.to_numpy()
    X_ps = pd.read_parquet(DATA / "X_heldout_portscan.parquet").reset_index(drop=True)

    fuse(fm, X.iloc[:2000])                                # warm-up so timing is not skewed
    ctx = run_pipeline(rf, fm, X)
    ctx_ps = run_pipeline(rf, fm, X_ps)
    base_seconds = ctx["t_fuse"] + ctx["t_rf"]

    variants = {"Hybrid (stacked RF only)": (make_variant(ctx, classes), make_variant(ctx_ps, classes))}
    for n in config_names:
        variants[label_of(all_cfgs[n])] = (make_variant(ctx, classes, all_cfgs[n]),
                                           make_variant(ctx_ps, classes, all_cfgs[n]))

    multi, binary, raw = {}, {}, {}
    for label, (v, v_ps) in variants.items():
        seconds = base_seconds + v["t_rule"]
        multi[label] = multiclass_column(y, v, labels, seconds)
        binary[label] = binary_column(y, v, v_ps["flag"], v_ps["score"], seconds)
        raw[label] = {"multiclass": {k: (float(x) if not isinstance(x, str) else x) for k, x in multi[label].items()},
                      "binary": {k: float(x) for k, x in binary[label].items()}}
    raw["_timing_seconds"] = {"ae_pass": ctx["t_fuse"], "rf_predict_proba": ctx["t_rf"], "n_rows": int(len(y))}

    t_multi, t_bin = format_table(multi), format_table(binary)
    t_multi.to_csv(RES / f"hybrid_eval_multiclass_{split}.csv")
    t_bin.to_csv(RES / f"hybrid_eval_binary_{split}.csv")
    save_json(RES / f"hybrid_eval_raw_{split}.json", raw)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    log(f"=== {split.upper()} | RF-vs-XGBoost format (12-class) ===\n" + t_multi.to_string())
    log(f"=== {split.upper()} | AE-vs-IF format (attack vs normal) ===\n" + t_bin.to_string())
    log(f"Saved tables to {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate hybrid and gated hybrid")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    a = ap.parse_args()
    if a.split == "test":
        log("Reporting on the TEST set with settings fixed on validation. Do not change configs after this.")
    main(a.split, a.configs)