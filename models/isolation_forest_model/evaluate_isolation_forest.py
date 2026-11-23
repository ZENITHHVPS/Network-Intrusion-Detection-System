"""
src/models/evaluate_isolation_forest.py

Trains + evaluates the Isolation Forest on the SAME BASIS and with the SAME OUTPUT
FILES as the Autoencoder (results/autoencoder/...), so the two models can be
compared file-for-file.

Same basis =
  * fit on X_train_anomaly_benign only
  * choose hyperparameters on VALIDATION ROC-AUC
  * calibrate the decision threshold on VALIDATION (F1-maximising threshold)
  * report TEST metrics once, plus held-out PortScan recall at that threshold

Run from the project root:
    python -m src.models.evaluate_isolation_forest            # data/processed        -> results/isolation_forest/main
    python -m src.models.evaluate_isolation_forest --log1p    # data/processed/log1p  -> results/isolation_forest/log1p
    python -m src.models.evaluate_isolation_forest --quick    # pilot (50k rows, tiny grid)
Options:
    --seeds N              seeds for the variance study (default 5)
    --attack-labels-test   optional path to a parquet with the multi-class test labels
                           (one column, same row order as y_test_anomaly_binary.parquet).
                           If omitted the script looks for y_test*.parquet files.

Outputs (in results/isolation_forest/<tag>/):
    hyperparameter_tuning.csv          one row per config (val ROC-AUC, PR-AUC, F1, threshold, train time)
    isolation_forest_test_metrics.csv  same columns as autoencoder_test_metrics.csv (+ size/latency)
    if_academic_table_metrics.csv      Accuracy / Recall / FAR / F1 / AP / ROC-AUC / Pred. time
    if_all_combined_metrics.json       everything above in one file
    if_per_attack_recall.csv           recall per attack family (+ held-out PortScan)
    if_seed_runs.csv                   one row per seed (best config refit, thresholds re-derived on val)
    if_seed_variance_summary.csv       mean / std / min / max over seeds
    if_threshold_sensitivity.csv       test metrics at several validation-derived thresholds
    scores/if_test_scores.parquet      per-row test scores, labels, predictions
    scores/if_heldout_scores.npy       held-out PortScan scores
    plots/confusion_matrix.png  roc_curve.png  pr_curve.png  error_distributions.png
    plots/tuning_curve.png             (Isolation Forest has no loss curve: this is its analogue,
                                        validation ROC-AUC vs max_samples / trees)
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

SEED = 42
ROOT = Path(__file__).resolve().parents[2]


#HELPERS
def anomaly_score(model, X):
    return -model.score_samples(X)          # higher = more anomalous


def load_labels(path):
    """One-column label parquet -> 1-D int array (safe for 1-row files)."""
    return pd.read_parquet(path).iloc[:, 0].to_numpy().astype(int)


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_attack_labels(data_dir, n_rows, override=None):
    """Look for multi-class test labels (attack family per row). Returns a Series of str or None."""
    cands = [Path(override)] if override else sorted(
        p for p in data_dir.glob("y_test*.parquet") if "binary" not in p.name.lower())
    for p in cands:
        try:
            s = pd.read_parquet(p).iloc[:, 0]
        except Exception:
            continue
        if len(s) == n_rows and s.nunique() > 2:
            print(f"    per-attack labels: {p.name}")
            return s.astype(str).reset_index(drop=True)
    print("    [!] no multi-class test labels found -> per-attack recall will be skipped "
          "(pass --attack-labels-test <parquet>)")
    return None


#PLOTS
def save_plots(plots_dir, title, y_test, s_test, thr, cm, tuning_df):
    plots_dir.mkdir(parents=True, exist_ok=True)

    # confusion matrix (counts + row %)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    row_pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Isolation Forest confusion matrix ({title})")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, s_test)
    op_fpr = cm[0, 1] / max(cm[0].sum(), 1)
    op_tpr = cm[1, 1] / max(cm[1].sum(), 1)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, s_test):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.scatter([op_fpr], [op_tpr], c="red", zorder=3, label="operating point")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve ({title})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # PR
    p, r, _ = precision_recall_curve(y_test, s_test)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(r, p, label=f"AP = {average_precision_score(y_test, s_test):.4f}")
    ax.axhline(y_test.mean(), color="k", ls="--", lw=0.8, label=f"prevalence {y_test.mean():.3f}")
    ax.scatter([op_tpr], [cm[1, 1] / max(cm[:, 1].sum(), 1)], c="red", zorder=3, label="operating point")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve ({title})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    # score distributions (the Isolation Forest analogue of the AE reconstruction-error plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(s_test.min(), thr) - 0.01, s_test.max() + 0.01, 80)
    ax.hist(s_test[y_test == 0], bins=bins, alpha=0.6, label="Benign", color="tab:blue")
    ax.hist(s_test[y_test == 1], bins=bins, alpha=0.6, label="Attack", color="tab:red")
    ax.axvline(thr, color="k", ls="--", label=f"threshold = {thr:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title(f"Score distributions ({title})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "error_distributions.png", dpi=150)
    plt.close(fig)

    # tuning curve (analogue of the training-loss curve)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (ne, mf), g in tuning_df.groupby(["n_estimators", "max_features"]):
        g = g.sort_values("max_samples")
        ax.plot(g["max_samples"], g["val_roc_auc"], marker="o",
                label=f"{int(ne)} trees, max_features={mf}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("max_samples")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title(f"Hyperparameter tuning curve ({title})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "tuning_curve.png", dpi=150)
    plt.close(fig)


#############################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--attack-labels-test", default=None)
    args = ap.parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 2)

    data_dir = ROOT / "data" / "processed" / ("log1p" if args.log1p else "")
    tag = ("log1p" if args.log1p else "main") + ("_quick" if args.quick else "")
    title = "log1p" if args.log1p else "base"
    out_dir = ROOT / "results" / "isolation_forest" / tag
    scores_dir, plots_dir = out_dir / "scores", out_dir / "plots"
    for d in (out_dir, scores_dir, plots_dir, ROOT / "models"):
        d.mkdir(parents=True, exist_ok=True)

    #load
    X_train = pd.read_parquet(data_dir / "X_train_anomaly_benign.parquet")
    X_val = pd.read_parquet(data_dir / "X_val_anomaly.parquet")
    y_val = load_labels(data_dir / "y_val_anomaly_binary.parquet")
    X_test = pd.read_parquet(data_dir / "X_test_anomaly.parquet")
    y_test = load_labels(data_dir / "y_test_anomaly_binary.parquet")
    X_port = pd.read_parquet(data_dir / "X_heldout_portscan.parquet")
    print(f"[*] {data_dir}")
    attack_labels = find_attack_labels(data_dir, len(X_test), args.attack_labels_test)
    if args.quick:
        rng = np.random.RandomState(SEED)
        X_train = X_train.sample(n=min(50_000, len(X_train)), random_state=SEED)
        vi = np.sort(rng.choice(len(X_val), min(50_000, len(X_val)), replace=False))
        X_val, y_val = X_val.iloc[vi], y_val[vi]
        ti = np.sort(rng.choice(len(X_test), min(50_000, len(X_test)), replace=False))
        X_test, y_test = X_test.iloc[ti], y_test[ti]
        if attack_labels is not None:
            attack_labels = attack_labels.iloc[ti].reset_index(drop=True)
        X_port = X_port.sample(n=min(50_000, len(X_port)), random_state=SEED)
    print(f"    train(benign)={len(X_train):,} val={len(X_val):,} "
          f"test={len(X_test):,} portscan={len(X_port):,}")

    #hyperparameter tuning (validation only)
    grid = {"n_estimators": [100, 200, 300],
            "max_samples": [256, 512, 1024, 2048],
            "max_features": [0.5, 1.0]}
    if args.quick:
        grid = {"n_estimators": [100], "max_samples": [256], "max_features": [1.0]}

    rows, best = [], None
    keys = list(grid)
    t_tune = time.perf_counter()
    for vals in itertools.product(*grid.values()):
        params = dict(zip(keys, vals))
        t0 = time.perf_counter()
        m = IsolationForest(contamination="auto", n_jobs=-1, random_state=SEED, **params).fit(X_train)
        train_s = time.perf_counter() - t0
        s_val = anomaly_score(m, X_val)
        thr_c = best_f1_threshold(y_val, s_val)
        row = {**params,
               "val_roc_auc": float(roc_auc_score(y_val, s_val)),
               "val_pr_auc": float(average_precision_score(y_val, s_val)),
               "val_f1": float(f1_score(y_val, (s_val >= thr_c).astype(int))),
               "calibrated_threshold": thr_c,
               "train_time_sec": train_s}
        rows.append(row)
        print(f"   {params}  val ROC-AUC={row['val_roc_auc']:.4f}  val F1={row['val_f1']:.4f}  ({train_s:.0f}s)")
        if best is None or row["val_roc_auc"] > best["row"]["val_roc_auc"]:
            best = {"row": row, "model": m, "params": params}
    tune_total_s = time.perf_counter() - t_tune
    tuning_df = pd.DataFrame(rows)
    tuning_df.to_csv(out_dir / "hyperparameter_tuning.csv", index=False)

    model, params, thr = best["model"], best["params"], best["row"]["calibrated_threshold"]
    s_val = anomaly_score(model, X_val)
    print(f"\n[+] Best (val ROC-AUC): {params}   threshold={thr:.6f}")

    #test (once)
    t0 = time.perf_counter()
    s_test = anomaly_score(model, X_test)
    pred_s = time.perf_counter() - t0                       # time to score the whole test set
    s_port = anomaly_score(model, X_port)
    mt = metrics_at(y_test, s_test, thr)
    pred = (s_test >= thr).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    port_recall = float((s_port >= thr).mean())

    #single-flow latency (per-call overhead included), 100 random test rows
    idx = np.random.RandomState(SEED).choice(len(X_test), min(100, len(X_test)), replace=False)
    t0 = time.perf_counter()
    for i in idx:
        model.score_samples(X_test.iloc[[i]])
    single_flow_s = (time.perf_counter() - t0) / len(idx)

    model_path = ROOT / "models" / f"isolation_forest_{tag}_ae_basis.joblib"
    joblib.dump(model, model_path)
    model_mb = os.path.getsize(model_path) / 1e6

    m = {
        "model_name": "Isolation Forest",
        "input_dim": X_test.shape[1],
        **{f"best_{k}": v for k, v in params.items()},
        "calibrated_threshold": thr,
        "test_accuracy": mt["accuracy"],
        "test_roc_auc": float(roc_auc_score(y_test, s_test)),
        "test_pr_auc": float(average_precision_score(y_test, s_test)),
        "test_f1": mt["f1"],
        "test_precision": mt["precision"],
        "test_recall": mt["recall"],
        "test_fpr": mt["fpr"],
        "inference_latency_seconds_per_sample": pred_s / len(X_test),
        "single_flow_latency_seconds": single_flow_s,
        "train_time_sec_best_model": best["row"]["train_time_sec"],
        "tuning_total_time_sec": tune_total_s,
        "model_size_mb": model_mb,
        "heldout_portscan_recall": port_recall,
    }
    pd.DataFrame([m]).to_csv(out_dir / "isolation_forest_test_metrics.csv", index=False)

    academic = pd.DataFrame([{
        "Models": "IF",
        "Accuracy (%)": round(100 * m["test_accuracy"], 2),
        "Recall (%)": round(100 * m["test_recall"], 2),
        "FAR (%)": round(100 * m["test_fpr"], 2),
        "F1-Score (%)": round(100 * m["test_f1"], 2),
        "AP (%)": round(100 * m["test_pr_auc"], 2),
        "ROC-AUC (%)": round(100 * m["test_roc_auc"], 2),
        "Pred. time (s)": round(pred_s, 2),
    }])
    academic.to_csv(out_dir / "if_academic_table_metrics.csv", index=False)

    #scores
    pd.DataFrame({"score": s_test.astype("float32"), "y_true": y_test.astype("int8"),
                  "pred": pred.astype("int8")}, index=X_test.index
                 ).to_parquet(scores_dir / "if_test_scores.parquet")
    np.save(scores_dir / "if_heldout_scores.npy", s_port.astype("float32"))

    #per-attack recall
    per_attack = None
    if attack_labels is not None:
        lab = attack_labels.to_numpy()
        out = []
        for name in sorted(set(lab)):
            mask = lab == name
            is_benign = name.strip().lower() == "benign"
            detected = int(((pred == 1) & mask).sum()) if not is_benign else int(((pred == 0) & mask).sum())
            out.append({"attack": name, "n_flows": int(mask.sum()),
                        "metric": "specificity (TNR)" if is_benign else "recall",
                        "detected_or_correct": detected,
                        "rate": detected / max(int(mask.sum()), 1),
                        "mean_score": float(s_test[mask].mean())})
        out.append({"attack": "PortScan (held-out, never seen)", "n_flows": len(s_port),
                    "metric": "recall", "detected_or_correct": int((s_port >= thr).sum()),
                    "rate": port_recall, "mean_score": float(s_port.mean())})
        per_attack = pd.DataFrame(out).sort_values("rate")
        per_attack.to_csv(out_dir / "if_per_attack_recall.csv", index=False)

    #threshold sensitivity (all thresholds derived from VALIDATION)
    benign_val = s_val[y_val == 0]
    rules = {"best_f1": thr}
    for q in (90, 95, 98, 99, 99.5):
        rules[f"benign_p{q}"] = float(np.percentile(benign_val, q))
    sens = []
    for name, t in rules.items():
        r = metrics_at(y_test, s_test, t)
        sens.append({"rule": name, "threshold": t, **r,
                     "portscan_recall": float((s_port >= t).mean())})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(out_dir / "if_threshold_sensitivity.csv", index=False)

    #seed variance (best config refit; thresholds re-derived on val each time)
    seed_rows = []
    for s in [SEED + k for k in range(args.seeds)]:
        if s == SEED:
            ms = model
        else:
            t0 = time.perf_counter()
            ms = IsolationForest(contamination="auto", n_jobs=-1, random_state=s, **params).fit(X_train)
            print(f"   seed {s}: refit in {time.perf_counter() - t0:.0f}s")
        sv, st, sp = anomaly_score(ms, X_val), anomaly_score(ms, X_test), anomaly_score(ms, X_port)
        ts = best_f1_threshold(y_val, sv)
        r = metrics_at(y_test, st, ts)
        seed_rows.append({"seed": s, "threshold": ts,
                          "test_roc_auc": float(roc_auc_score(y_test, st)),
                          "test_pr_auc": float(average_precision_score(y_test, st)),
                          "test_accuracy": r["accuracy"], "test_precision": r["precision"],
                          "test_recall": r["recall"], "test_f1": r["f1"], "test_fpr": r["fpr"],
                          "heldout_portscan_recall": float((sp >= ts).mean())})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out_dir / "if_seed_runs.csv", index=False)
    metric_cols = [c for c in seed_df.columns if c not in ("seed", "threshold")]
    seed_summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [seed_df[c].mean() for c in metric_cols],
        "std": [seed_df[c].std(ddof=1) if len(seed_df) > 1 else 0.0 for c in metric_cols],
        "min": [seed_df[c].min() for c in metric_cols],
        "max": [seed_df[c].max() for c in metric_cols],
        "n_seeds": len(seed_df)})
    seed_summary.to_csv(out_dir / "if_seed_variance_summary.csv", index=False)

    #plots
    save_plots(plots_dir, title, y_test, s_test, thr, cm, tuning_df)

    #combined json
    combined = {
        "meta": {"tag": tag, "data_dir": str(data_dir), "seed": SEED, "sklearn": sklearn.__version__,
                 "rows": {"train": len(X_train), "val": len(X_val),
                          "test": len(X_test), "portscan": len(X_port)},
                 "feature_names": list(X_train.columns)},
        "best_params": params,
        "test_metrics": m,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "academic_table": academic.to_dict(orient="records")[0],
        "threshold_sensitivity": sens_df.to_dict(orient="records"),
        "seed_variance_summary": seed_summary.to_dict(orient="records"),
        "per_attack_recall": None if per_attack is None else per_attack.to_dict(orient="records"),
        "hyperparameter_tuning": tuning_df.to_dict(orient="records"),
    }
    with open(out_dir / "if_all_combined_metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\n=== Isolation Forest (same basis as the Autoencoder) ===")
    print(pd.DataFrame([m]).T.to_string(header=False))
    print("\n" + academic.to_string(index=False))
    print("\nSeed variance:\n" + seed_summary.to_string(index=False))
    print(f"\n[✓] Outputs -> {out_dir}\n[✓] Model   -> {model_path}")


if __name__ == "__main__":
    main()
