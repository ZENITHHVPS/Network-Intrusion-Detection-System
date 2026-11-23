#!/usr/bin/env python
"""
Hybrid NIDS: Autoencoder (frozen) -> feature fusion -> Random Forest.
Fusion = 68 original + 16 latent + 1 reconstruction error = 85 features.

Self-contained: needs no other project script.
Place this file in the PROJECT ROOT (next to data/, models/, results/).

Run from project root:
  python train_hybrid_ae_rf.py --stage check              # smoke test + leakage check
  python train_hybrid_ae_rf.py --stage fit --trees 50     # quick trial run
  python train_hybrid_ae_rf.py --stage fit                # full run (500 trees)
  python train_hybrid_ae_rf.py --stage test               # testing
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score
                            )

# 1. CONFIGURATION
SEED = 42
ROOT = Path(__file__).resolve().parent            # script lives in the project root

#The AE is the LOG1P variant, so every input (AE and RF) comes from the log1p folder.
DATA_BASE = ROOT / "data" / "processed"
DATA = DATA_BASE / "log1p"

AE_PATH = ROOT / "models" / "autoencoder" / "best_autoencoder_log1p.keras"
AE_META = ROOT / "models" / "autoencoder" / "autoencoder_metadata_log1p.json"

RF_PARAMS_FILE = ROOT / "results" / "random_forest" / "best_params.json"
RES = ROOT / "results" / "hybrid_ae_rf"
MOD = ROOT / "models" / "hybrid_ae_rf"
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

LATENT_LAYER = "latent_bottleneck"
FINAL_TREES = 500

#Used only if results/random_forest/best_params.json is missing
#(these are the tuned values from the RF tuning stage).
FALLBACK_CONFIG = {
    "params": {"max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1,
               "max_features": 0.5, "max_samples": None},
    "weighting": "sqrt",
}


def log(msg):
    print(f"[HYBRID] {msg}", flush=True)


# 2. HELPERS (same definitions as the RF training script)
def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def class_weights(y, scheme):
    if scheme == "none":
        return None
    counts = y.value_counts()
    weights = len(y) / (len(counts) * counts)
    if scheme == "sqrt":
        weights = np.sqrt(weights)
    weights = weights / (weights * counts).sum() * len(y)
    return {int(k): float(v) for k, v in weights.items()}


def evaluate(model, X, y, names, split):
    t0 = time.perf_counter()
    pred = model.predict(X)
    inference = time.perf_counter() - t0
    yt = y.to_numpy()
    labels = list(range(len(names)))
    benign = yt == 0
    metrics = {
        "split": split,
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_precision": float(precision_score(yt, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(yt, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, pred, labels=labels, average="weighted", zero_division=0)),
        "false_positive_rate": float((pred[benign] != 0).mean()) if benign.any() else 0.0,
        "attack_detection_rate": float((pred[~benign] != 0).mean()) if (~benign).any() else 0.0,
        "inference_seconds": float(inference),
        "inference_us_per_flow": float(1e6 * inference / len(X)),
        "n_rows": int(len(X)),
    }
    report = pd.DataFrame(classification_report(
        yt, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(yt, pred, labels=labels), index=names, columns=names)
    return metrics, report, cm, pred


def save_confusion_png(cm, path, title):
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=90)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if norm.values[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def binary_metrics(y_true, flagged):
    attack = y_true != 0
    return {
        "attack_detection_rate": float(flagged[attack].mean()),
        "false_positive_rate": float(flagged[~attack].mean()),
        "missed_attacks": int((~flagged[attack]).sum()),
        "false_alarms": int(flagged[~attack].sum()),
    }


def load_rf_config():
    if RF_PARAMS_FILE.is_file():
        cfg = json.loads(RF_PARAMS_FILE.read_text(encoding="utf-8"))
        log(f"RF settings from {RF_PARAMS_FILE}")
    else:
        cfg = FALLBACK_CONFIG
        log("best_params.json not found: using FALLBACK_CONFIG from this script")
    return cfg["params"], cfg["weighting"]


# 3. DATA LOADING (log1p folder)
def load_split(split):
    X = pd.read_parquet(DATA / f"X_{split}_supervised.parquet").reset_index(drop=True)
    y = pd.read_parquet(DATA / f"y_{split}_supervised.parquet")["Label"].astype("int32").reset_index(drop=True)
    return X, y


def load_names():
    for folder in (DATA, DATA_BASE):
        f = folder / "label_mapping.json"
        if f.is_file():
            mapping = json.loads(f.read_text(encoding="utf-8"))
            return [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    raise FileNotFoundError("label_mapping.json not found in data/processed/log1p or data/processed")


# 4. AUTOENCODER -> FEATURE FUSION
def load_feature_model():
    ae = tf.keras.models.load_model(AE_PATH, compile=False)   # frozen, no retraining
    ae.trainable = False
    return tf.keras.Model(
        inputs=ae.input,
        outputs=[ae.get_layer(LATENT_LAYER).output, ae.output],
    )


def fuse(feature_model, X, batch_size=4096):
    """Returns (fused 85-feature float32 matrix, reconstruction errors). One AE pass."""
    Xn = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
    latent, recon = feature_model.predict(Xn, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(Xn - recon), axis=1)
    fused = np.concatenate([Xn, latent, errors.reshape(-1, 1)], axis=1).astype(np.float32)
    return fused, errors


def fused_feature_names(original_cols, latent_dim):
    return list(original_cols) + [f"ae_latent_{i}" for i in range(latent_dim)] + ["ae_recon_error"]


# 5. CHECK: smoke test + leakage check (no training, no test set)
def stage_check():
    log(f"Project root: {ROOT}")
    for p in (AE_PATH, AE_META, DATA / "X_train_supervised.parquet", DATA / "y_train_supervised.parquet",
              DATA / "X_val_supervised.parquet", DATA / "X_test_supervised.parquet"):
        log(f"{'OK     ' if p.is_file() else 'MISSING'} {p}")

    ae = tf.keras.models.load_model(AE_PATH, compile=False)
    ae.summary()
    log(f"Layer names: {[l.name for l in ae.layers]}")
    fm = load_feature_model()

    X_va, y_va = load_split("val")
    F, err = fuse(fm, X_va.iloc[:5000])
    log(f"Fused shape for 5,000 rows: {F.shape} (expect (5000, 85))")
    yy = y_va.iloc[:5000].to_numpy()
    log(f"Mean error  benign: {err[yy == 0].mean():.6f} | attack: {err[yy != 0].mean():.6f} (attack should be higher)")

    #Leakage check: AE saw the benign TRAIN rows, so their error may be lower than on benign VAL rows.
    X_tr, y_tr = load_split("train")
    sample = X_tr[y_tr == 0].sample(200_000, random_state=SEED)
    _, e_tr = fuse(fm, sample)
    _, e_va = fuse(fm, X_va[y_va == 0])
    m_tr, m_va = e_tr.mean(), e_va.mean()
    log(f"Benign TRAIN error mean {m_tr:.6f}, p99 {np.percentile(e_tr, 99):.6f}")
    log(f"Benign VAL   error mean {m_va:.6f}, p99 {np.percentile(e_va, 99):.6f}")
    gap = (m_va - m_tr) / m_va * 100
    log(f"Train error is {gap:.1f}% lower than val. "
        + ("Above ~10-15%: consider training the RF on rows the AE never saw." if gap > 12
           else "Small gap: leakage through the error feature is unlikely to matter."))

    names = load_names()
    log(f"Classes ({len(names)}): {names}")
    log("PortScan in supervised classes? " + str(any("portscan" in n.lower() for n in names))
        + "  (should be False for a held-out experiment)")


# 6. FIT: train RF on fused features, validation only
def stage_fit(trees):
    params, scheme = load_rf_config()
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    feature_model = load_feature_model()
    class_names = load_names()

    log("Fusing training data (AE forward pass)...")
    Xf_tr, _ = fuse(feature_model, X_tr)
    names = fused_feature_names(X_tr.columns, Xf_tr.shape[1] - X_tr.shape[1] - 1)
    log(f"Fused matrix: {Xf_tr.shape} (expected 85 features)")

    rf = RandomForestClassifier(
        **params, n_estimators=trees,
        class_weight=class_weights(y_tr, scheme),
        random_state=SEED, n_jobs=-1,
    )
    log(f"Training Random Forest: {trees} trees, params={params}, weighting={scheme}")
    t0 = time.perf_counter()
    rf.fit(Xf_tr, y_tr)
    seconds = time.perf_counter() - t0
    log(f"Trained in {seconds / 60:.1f} min")

    joblib.dump(rf, MOD / "hybrid_rf.joblib", compress=3)
    save_json(MOD / "fused_feature_names.json", names)
    imp = pd.Series(rf.feature_importances_, index=names, name="importance").sort_values(ascending=False)
    imp.to_csv(RES / "hybrid_feature_importance.csv")
    log("Top 10 features:\n" + imp.head(10).to_string())
    log(f"ae_recon_error rank: {list(imp.index).index('ae_recon_error') + 1} of {len(imp)}")

    #validation evaluation (test set untouched)
    Xf_va, err_va = fuse(feature_model, X_va)
    metrics, report, cm, pred = evaluate(rf, Xf_va, y_va, class_names, "val")
    metrics.update(train_seconds=seconds, n_estimators=trees, weighting=scheme, **params)
    save_json(RES / "hybrid_val_metrics.json", metrics)
    report.to_csv(RES / "hybrid_val_report.csv")
    cm.to_csv(RES / "hybrid_val_confusion.csv")

    #thresholds for the decision rule, both chosen on VALIDATION
    tau_f1 = json.loads(AE_META.read_text(encoding="utf-8"))["calibrated_threshold"]
    yv = y_va.to_numpy()
    tau_p99 = float(np.percentile(err_va[yv == 0], 99))
    save_json(RES / "hybrid_thresholds.json", {"tau_best_f1": tau_f1, "tau_p99_benign_val": tau_p99})

    flag_stacked = pred != 0
    variants = {
        "stacked": flag_stacked,
        "stacked_plus_rule_tau_f1": flag_stacked | (err_va >= tau_f1),
        "stacked_plus_rule_tau_p99": flag_stacked | (err_va >= tau_p99),
    }
    summary = {k: binary_metrics(yv, v) for k, v in variants.items()}
    summary["thresholds"] = {"tau_best_f1": tau_f1, "tau_p99_benign_val": tau_p99}
    save_json(RES / "hybrid_val_summary.json", summary)

    log(f"VAL macro-F1: {metrics['macro_f1']:.4f} | accuracy: {metrics['accuracy']:.4f} | "
        f"FPR: {metrics['false_positive_rate']:.4%} | detection: {metrics['attack_detection_rate']:.4%}")
    log("VAL variants (choose the one you will report from THESE, before running test):\n"
        + json.dumps(summary, indent=2))


# 7. TEST (once) + held-out PortScan
def stage_test(force=False):
    lock = RES / "TEST_SET_USED.txt"
    if lock.exists() and not force:
        raise SystemExit("Test set already evaluated. Use --force only if intentional.")

    rf = joblib.load(MOD / "hybrid_rf.joblib")
    th = json.loads((RES / "hybrid_thresholds.json").read_text(encoding="utf-8"))
    taus = {"tau_f1": th["tau_best_f1"], "tau_p99": th["tau_p99_benign_val"]}
    feature_model = load_feature_model()
    class_names = load_names()

    X_te, y_te = load_split("test")
    Xf_te, err_te = fuse(feature_model, X_te)
    yt = y_te.to_numpy()

    metrics, report, cm, pred = evaluate(rf, Xf_te, y_te, class_names, "test")
    save_json(RES / "hybrid_test_metrics.json", metrics)
    report.to_csv(RES / "hybrid_test_report.csv")
    cm.to_csv(RES / "hybrid_test_confusion.csv")
    save_confusion_png(cm, RES / "hybrid_test_confusion.png", "Hybrid AE+RF (stacked) - test")

    flag_stacked = pred != 0
    summary = {"thresholds": th, "stacked": binary_metrics(yt, flag_stacked)}
    for name, tau in taus.items():
        summary[f"stacked_plus_rule_{name}"] = binary_metrics(yt, flag_stacked | (err_te >= tau))
        summary[f"ae_alone_{name}"] = binary_metrics(yt, err_te >= tau)

    ps_file = DATA / "X_heldout_portscan.parquet"
    if ps_file.exists():
        X_ps = pd.read_parquet(ps_file).reset_index(drop=True)
        Xf_ps, err_ps = fuse(feature_model, X_ps)
        ps_pred = rf.predict(Xf_ps) != 0
        ps = {"n": int(len(X_ps)), "caught_by_stacked_rf": float(ps_pred.mean())}
        for name, tau in taus.items():
            ps[f"caught_by_rule_{name}"] = float((ps_pred | (err_ps >= tau)).mean())
            ps[f"caught_by_ae_alone_{name}"] = float((err_ps >= tau).mean())
        summary["portscan"] = ps

    save_json(RES / "hybrid_test_summary.json", summary)
    lock.write_text(f"evaluated {time.strftime('%Y-%m-%d %H:%M')}", encoding="utf-8")
    log("TEST SUMMARY:\n" + json.dumps(summary, indent=2))
    log(f"Stacked macro-F1: {metrics['macro_f1']:.4f} | accuracy: {metrics['accuracy']:.4f} | "
        f"latency (RF only): {metrics['inference_us_per_flow']:.1f} us/flow")


# 8. MAIN
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid AE + Random Forest")
    ap.add_argument("--stage", required=True, choices=["check", "fit", "test"])
    ap.add_argument("--trees", type=int, default=FINAL_TREES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "check":
        stage_check()
    elif a.stage == "fit":
        stage_fit(a.trees)
    else:
        stage_test(a.force)
