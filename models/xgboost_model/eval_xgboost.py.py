#!/usr/bin/env python
"""
Benchmark summary for SUPERVISED models (Random Forest / XGBoost) - same metrics for both,
so Member 1 and Member 2 results can be compared side by side.

Run from the project root:
    python models/report_supervised.py --model models/xgboost/xgb_tuned.joblib --name "XGBoost (tuned)"
    python models/report_supervised.py --model models/random_forest/rf_tuned.joblib --name "Random Forest (tuned)"

    --split val    (default) validation set - use this while models are still being compared
    --split test   the official one-time test evaluation; needs --confirm-test
    --train-seconds 2920   optional: training time to show in the summary

Outputs: results/benchmarks/<name>_<split>_summary.json and <name>_<split>_per_class.csv
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score)

DATA = Path("data/processed")
OUT = Path("results/benchmarks")
RARE_MIN = 10           # classes with fewer rows than this in the split are "too small to judge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--train-seconds", type=float, default=None)
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()

    if a.split == "test" and not a.confirm_test:
        raise SystemExit("Test set is touched once, at the very end. Add --confirm-test when the team is ready.")

    X = pd.read_parquet(DATA / f"X_{a.split}_supervised.parquet")
    y = pd.read_parquet(DATA / f"y_{a.split}_supervised.parquet")["Label"].to_numpy()
    mapping = json.loads((DATA / "label_mapping.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    labels = list(range(len(names)))
    model = joblib.load(a.model)

    t = time.perf_counter()
    proba = model.predict_proba(X)
    infer = time.perf_counter() - t
    pred = np.asarray(proba).argmax(1)

    # ---- per-class
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=names)
    reliable = per_class["support"] >= RARE_MIN
    small = per_class.index[~reliable].tolist()

    # NIDS view: benign (0) vs any attack
    benign, attack = y == 0, y != 0
    fpr = (pred[benign] != 0).mean()
    detection = (pred[attack] != 0).mean()
    detected = attack & (pred != 0)
    type_acc = (pred[detected] == y[detected]).mean()        # detected attacks given the right attack type

    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")

    size_mb = Path(a.model).stat().st_size / 1e6
    m = {
        "model": a.name, "split": a.split, "n_flows": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "macro_f1_excl_small_classes": float(per_class.loc[reliable, "f1"].mean()),
        "small_classes_excluded": small,
        "weighted_f1": f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "macro_roc_auc_ovr": auc,
        "false_positive_rate": fpr,
        "attack_detection_rate": detection,
        "attack_type_accuracy": type_acc,
        "false_alarms": int((pred[benign] != 0).sum()),
        "missed_attacks": int((pred[attack] == 0).sum()),
        "inference_us_per_flow": 1e6 * infer / len(y),
        "throughput_flows_per_s": len(y) / infer,
        "model_size_mb": size_mb,
        "train_seconds": a.train_seconds,
    }

    # ---- print
    line = "-" * 56
    print(f"\n{a.name.upper()} - {'TEST METRICS (OFFICIAL BENCHMARK)' if a.split == 'test' else 'VALIDATION METRICS'}")
    print(line)
    rows = [
        ("Accuracy:", f"{m['accuracy']:.4f}"),
        ("Macro Precision:", f"{m['macro_precision']:.4f}"),
        ("Macro Recall:", f"{m['macro_recall']:.4f}"),
        ("Macro F1-Score:", f"{m['macro_f1']:.4f}"),
        ("Macro F1 (excl. tiny classes):", f"{m['macro_f1_excl_small_classes']:.4f}"),
        ("Weighted F1-Score:", f"{m['weighted_f1']:.4f}"),
        ("MCC:", f"{m['mcc']:.4f}"),
        ("Macro ROC-AUC (OvR):", f"{m['macro_roc_auc_ovr']:.4f}"),
        ("", ""),
        ("Attack Detection Rate:", f"{m['attack_detection_rate']:.4%}"),
        ("False Positive Rate (FPR):", f"{m['false_positive_rate']:.4%}"),
        ("Attack-Type Accuracy:", f"{m['attack_type_accuracy']:.4%}"),
        ("False Alarms / Missed Attacks:", f"{m['false_alarms']:,} / {m['missed_attacks']:,}"),
        ("", ""),
        ("Inference Latency/Flow:", f"{m['inference_us_per_flow']:.2f} microseconds"),
        ("Throughput:", f"{m['throughput_flows_per_s']:,.0f} flows/s"),
        ("Model Size:", f"{size_mb:.1f} MB"),
    ]
    if a.train_seconds:
        rows.append(("Training Time:", f"{a.train_seconds:,.0f} s ({a.train_seconds / 60:.1f} min)"))
    for k, v in rows:
        print(f"{k:<32}{v}")
    if small:
        print(f"\n(excluded from 'excl. tiny classes': {', '.join(small)} - fewer than {RARE_MIN} flows)")

    print(f"\nPER-CLASS RESULTS ({a.split}, {len(y):,} flows)")
    print(line)
    print(per_class.assign(**{c: per_class[c].round(4) for c in ("precision", "recall", "f1")}).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", a.name.lower()).strip("_")
    json.dump(m, open(OUT / f"{slug}_{a.split}_summary.json", "w"), indent=2, default=float)
    per_class.to_csv(OUT / f"{slug}_{a.split}_per_class.csv")
    print(f"\n[+] Saved to {OUT / (slug + '_' + a.split)}_summary.json / _per_class.csv")


if __name__ == "__main__":
    main()