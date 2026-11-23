#!/usr/bin/env python
"""
Member 2 - XGBoost for known-attack classification (CICIDS2017, Hybrid NIDS project).

Reads the shared supervised splits produced by src/preprocessing/data_pipeline.py:
    data/processed/X_{train,val,test}_supervised.parquet
    data/processed/y_{train,val,test}_supervised.parquet   (column "Label", BENIGN = 0)
    data/processed/label_mapping.json, feature_names.json

Team rules followed here:
    * random_state = 42 everywhere
    * every choice (early stopping, class weighting, hyper-parameters) is made on VALIDATION only
    * the TEST set is touched exactly once, by `--stage test`, and the script refuses to do it twice

Stages (run in order, from the project root):
    python src/models/train_xgboost.py --stage baseline     # baseline + class-imbalance analysis
    python src/models/train_xgboost.py --stage tune         # random search on validation
    python src/models/train_xgboost.py --stage final        # refit best config on full train, save model
    python src/models/train_xgboost.py --stage test         # ONE-TIME test evaluation (baseline + tuned)

    add --device cuda to train on an NVIDIA GPU (Colab / Kaggle) - roughly 10x faster.
    add --quick to run everything on a 10% stratified sample (for checking the code end to end).

Outputs:
    results/xgboost/*.csv|json|png    metrics, per-class reports, confusion matrices, tuning log
    Every evaluation prints five sections:
      A. Overall classification   accuracy, macro precision/recall/F1, weighted F1, MCC
      B. Per-class                precision, recall, F1, support for each of the 12 classes
      C. Intrusion detection      attack detection rate, FPR, attack-type accuracy,
                                  false alarms, missed attacks
      D. Computational            training time, inference latency, throughput, model size
      E. Confusion matrix
    models/xgboost/*.json|joblib      saved baseline + tuned models
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

SEED = 42
# project root if the script sits in src/models/, otherwise the folder it is run from (Colab)
ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "data").exists() \
    else Path.cwd()
DATA = ROOT / "data" / "processed"
RES = ROOT / "results" / "xgboost"
MOD = ROOT / "models" / "xgboost"
RARE_MIN = 10        # classes with fewer rows than this in a split are "too small to judge"

# Baseline config.
# Close to XGBoost defaults on purpose: the baseline is the reference point the tuning must beat.
BASELINE_PARAMS = dict(
    n_estimators=1000,          # upper bound; early stopping picks the real number of rounds
    learning_rate=0.3,          # XGBoost default
    max_depth=6,                # XGBoost default
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
)

# Fixed settings shared by every model in this script (not tuned).
FIXED = dict(
    objective="multi:softprob",  # multiclass, outputs class probabilities (needed by the hybrid NIDS)
    eval_metric="mlogloss",      # smooth, probability-aware metric for early stopping
    tree_method="hist",          # histogram algorithm: the only practical choice for ~1.6M rows
    max_bin=256,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=-1,
)

# Random search space (20 configs are sampled from this by default).
SEARCH_SPACE = dict(
    max_depth=[4, 6, 8, 10],
    learning_rate=[0.05, 0.1, 0.2],
    min_child_weight=[1, 3, 5],
    subsample=[0.7, 0.85, 1.0],
    colsample_bytree=[0.6, 0.8, 1.0],
    gamma=[0.0, 0.1, 1.0],
    reg_lambda=[1.0, 5.0],
)


# Data loading

def load(quick=False):
    X = {s: pd.read_parquet(DATA / f"X_{s}_supervised.parquet") for s in ("train", "val", "test")}
    y = {s: pd.read_parquet(DATA / f"y_{s}_supervised.parquet")["Label"].astype("int32")
         for s in ("train", "val", "test")}
    mapping = json.loads((DATA / "label_mapping.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    # XGBoost needs labels 0..K-1 with no gaps
    assert sorted(mapping.values()) == list(range(len(mapping))), "class ids must be 0..K-1"
    if quick:
        for s in X:
            idx = stratified_sample(y[s], 0.10, min_per_class=50)
            X[s], y[s] = X[s].loc[idx], y[s].loc[idx]
    return X, y, names


def stratified_sample(y, frac, min_per_class=2000):
    """Index of a per-class sample: frac of each class, but keep at least min_per_class rows
    (or the whole class if it is smaller) so rare attacks are never thinned out."""
    rng = np.random.RandomState(SEED)
    keep = []
    for c, idx in y.groupby(y).groups.items():
        idx = np.asarray(idx)
        n = min(len(idx), max(int(np.ceil(frac * len(idx))), min_per_class))
        keep.append(rng.choice(idx, n, replace=False))
    return pd.Index(np.concatenate(keep))


def class_weights(y, scheme):
    """Per-row sample weights. XGBoost's scale_pos_weight is binary-only, so multiclass
    imbalance is handled with sample_weight.
        none     - every row weight 1
        balanced - n / (K * n_c)  (sklearn 'balanced'; extreme for Heartbleed/Infiltration)
        sqrt     - sqrt of balanced: boosts rare classes without letting ~10 rows dominate"""
    if scheme == "none":
        return None
    counts = y.value_counts()
    w = len(y) / (len(counts) * counts)
    if scheme == "sqrt":
        w = np.sqrt(w)
    w = w / (w * counts).sum() * len(y)            # normalise: mean weight = 1
    return y.map(w).to_numpy(dtype="float32")


# Training and evaluation

def fit(params, Xtr, ytr, Xval, yval, weights, device):
    # validation set is passed in only for early stopping
    model = XGBClassifier(**{**FIXED, **params}, device=device)
    t = time.perf_counter()
    model.fit(Xtr, ytr, sample_weight=weights, eval_set=[(Xval, yval)], verbose=False)
    return model, time.perf_counter() - t


def evaluate(model, X, y, names, split, fast=False):
    """All supervised metrics. fast=True skips ROC-AUC (used inside the tuning loop)."""
    t = time.perf_counter()
    proba = model.predict_proba(X)
    infer = time.perf_counter() - t
    pred = proba.argmax(1)
    yt = y.to_numpy()
    labels = list(range(len(names)))
    benign, attack = yt == 0, yt != 0
    detected = attack & (pred != 0)

    p, r, f, s = precision_recall_fscore_support(yt, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=names)
    # Heartbleed and Infiltration have only a handful of rows, a single flow swings their F1
    reliable = per_class["support"] >= RARE_MIN

    auc = float("nan")
    if not fast:
        try:
            auc = roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            pass

    m = {
        "split": split,
        "n_rows": len(X),
        # overall classification
        "accuracy": accuracy_score(yt, pred),
        "macro_precision": precision_score(yt, pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(yt, pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(yt, pred, labels=labels, average="macro", zero_division=0),
        "macro_f1_excl_small_classes": float(per_class.loc[reliable, "f1"].mean()),
        "small_classes_excluded": per_class.index[~reliable].tolist(),
        "weighted_f1": f1_score(yt, pred, labels=labels, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(yt, pred),
        "macro_roc_auc_ovr": auc,
        # NIDS view: any attack class counts as "alert"
        "attack_detection_rate": float((pred[attack] != 0).mean()),    # attacks flagged as attack
        "false_positive_rate": float((pred[benign] != 0).mean()),      # benign flagged as attack
        "attack_type_accuracy": float((pred[detected] == yt[detected]).mean()) if detected.any() else float("nan"),
        "false_alarms": int((pred[benign] != 0).sum()),
        "missed_attacks": int((pred[attack] == 0).sum()),
        # efficiency
        "inference_seconds": infer,
        "inference_us_per_flow": 1e6 * infer / len(X),
        "throughput_flows_per_s": len(X) / infer,
    }
    report = pd.DataFrame(classification_report(
        yt, pred, labels=labels, target_names=names, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(yt, pred, labels=labels), index=names, columns=names)
    return m, report, cm, per_class


def print_benchmark(title, m, per_class, cm, model_path=None, train_seconds=None):
    """Prints sections A-E (same layout for baseline, tuned and Random Forest)."""
    if model_path is not None and Path(model_path).exists():
        m["model_size_mb"] = Path(model_path).stat().st_size / 1e6
    if train_seconds is not None:
        m["train_seconds"] = train_seconds
    kind = "TEST METRICS (OFFICIAL BENCHMARK)" if m["split"] == "test" else "VALIDATION METRICS"
    line, thin = "=" * 60, "-" * 60
    na = "n/a (not recorded)"

    def rows(items):
        return [f"  {k:<34}{v}" for k, v in items]

    out = [f"\n{line}", f"{title.upper()} - {kind}  ({m['n_rows']:,} flows)", line]

    out += ["\nA. OVERALL CLASSIFICATION PERFORMANCE", thin] + rows([
        ("Accuracy:", f"{m['accuracy']:.4f}"),
        ("Macro Precision:", f"{m['macro_precision']:.4f}"),
        ("Macro Recall:", f"{m['macro_recall']:.4f}"),
        ("Macro F1:", f"{m['macro_f1']:.4f}"),
        ("Macro F1 (excl. tiny classes):", f"{m['macro_f1_excl_small_classes']:.4f}"),
        ("Weighted F1:", f"{m['weighted_f1']:.4f}"),
        ("MCC:", f"{m['mcc']:.4f}"),
    ])
    if m["small_classes_excluded"]:
        out.append(f"  (tiny classes = fewer than {RARE_MIN} flows: {', '.join(m['small_classes_excluded'])})")

    out += ["\nB. PER-CLASS PERFORMANCE", thin]
    table = per_class.rename(columns={"precision": "Precision", "recall": "Recall",
                                      "f1": "F1", "support": "Support"})
    out += ["  " + r for r in table.round({"Precision": 4, "Recall": 4, "F1": 4}).to_string().splitlines()]

    out += ["\nC. INTRUSION DETECTION PERFORMANCE", thin] + rows([
        ("Attack Detection Rate:", f"{m['attack_detection_rate']:.4%}"),
        ("False Positive Rate (FPR):", f"{m['false_positive_rate']:.4%}"),
        ("Attack-Type Accuracy:", f"{m['attack_type_accuracy']:.4%}"),
        ("False Alarms:", f"{m['false_alarms']:,}  (benign flows flagged as an attack)"),
        ("Missed Attacks:", f"{m['missed_attacks']:,}  (attack flows labelled BENIGN)"),
    ])

    ts = m.get("train_seconds")
    out += ["\nD. COMPUTATIONAL PERFORMANCE", thin] + rows([
        ("Training Time:", f"{ts:,.0f} s ({ts / 60:.1f} min)" if ts is not None else na),
        ("Inference Latency:", f"{m['inference_us_per_flow']:.2f} microseconds per flow"),
        ("Throughput:", f"{m['throughput_flows_per_s']:,.0f} flows/s"),
        ("Model Size:", f"{m['model_size_mb']:.1f} MB" if "model_size_mb" in m else na),
    ])
    out += ["\nE. CONFUSION MATRIX  (rows = true class, columns = predicted class, counts)", thin]
    short = [short_name(c) for c in cm.columns]
    out += ["  " + r for r in cm.set_axis(short, axis=1).to_string().splitlines()]
    out.append("  Diagonal = correct. BENIGN column = missed attacks; BENIGN row = false alarms.")
    out.append(line + "\n")
    print("\n".join(out), flush=True)


def short_name(c):
    """Compact column labels so the 12x12 matrix fits in a terminal."""
    c = c.replace("DoS ", "").replace("Web Attack", "WebAttack")
    return c if len(c) <= 9 else c[:9]


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=2, default=float)


def save_cm_png(cm, path, title):
    """Heatmap coloured by row % (per-class recall), each cell labelled with count and row %."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[XGB] matplotlib/seaborn not installed - confusion matrix saved as CSV only")
        return
    # colour by row share, raw counts would make every small class look empty
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    labels = np.array([[f"{c:,}\n{p:.1%}" if c else "0" for c, p in zip(cr, pr)]
                       for cr, pr in zip(cm.to_numpy(), norm.to_numpy())])
    plt.figure(figsize=(14, 11))
    sns.heatmap(norm, annot=labels, fmt="", cmap="Blues", cbar=True, vmin=0, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"label": "share of true class (row %)"})
    plt.title(f"{title}\n(cell = count / % of the true class)")
    plt.ylabel("True class"); plt.xlabel("Predicted class")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def log(msg):
    print(f"[XGB] {msg}", flush=True)


# Stages

def stage_baseline(X, y, names, device):
    # class imbalance analysis (train distribution)
    dist = pd.DataFrame({s: y[s].value_counts() for s in y}).fillna(0).astype(int)
    dist.index = [names[i] for i in dist.index]
    dist["train_%"] = (100 * dist["train"] / dist["train"].sum()).round(4)
    dist["imbalance_vs_benign"] = (dist["train"].max() / dist["train"]).round(1)
    for scheme in ("balanced", "sqrt"):
        w = class_weights(y["train"], scheme)
        dist[f"weight_{scheme}"] = (pd.Series(w, index=y["train"].index)
                                    .groupby(y["train"].map(dict(enumerate(names)))).first().round(3))
    dist = dist.sort_values("train", ascending=False)
    dist.to_csv(RES / "class_distribution.csv")
    log("class distribution:\n" + dist.to_string())
    # test counts are only printed here, nothing is fitted or scored on test
    small = dist.index[(dist["val"] < 10) | (dist["test"] < 10)].tolist()
    if small:
        log(f"WARNING: <10 rows in val/test for {small}: their F1 is very noisy; report them separately")

    # baseline + three weighting schemes, all scored on validation
    rows, best = [], None
    for scheme in ("none", "sqrt", "balanced"):
        log(f"baseline, class weighting = {scheme} ...")
        model, secs = fit(BASELINE_PARAMS, X["train"], y["train"], X["val"], y["val"],
                          class_weights(y["train"], scheme), device)
        m, rep, cm, per_class = evaluate(model, X["val"], y["val"], names, "val")
        m.update(weighting=scheme, train_seconds=secs, best_iteration=model.best_iteration)
        rows.append({k: v for k, v in m.items() if k != "small_classes_excluded"})
        rep.to_csv(RES / f"baseline_{scheme}_val_report.csv")
        log(f"  macro-F1 {m['macro_f1']:.4f} | FPR {m['false_positive_rate']:.4%} | "
            f"rounds {model.best_iteration + 1} | {secs:.0f}s")
        if scheme == "none":                               # the true baseline (no imbalance handling)
            model.save_model(MOD / "xgb_baseline.json")
            joblib.dump(model, MOD / "xgb_baseline.joblib")
            cm.to_csv(RES / "baseline_val_confusion.csv")
            save_cm_png(cm, RES / "baseline_val_confusion.png", "Baseline XGBoost - validation")
            print_benchmark("XGBoost (baseline)", m, per_class, cm, MOD / "xgb_baseline.joblib", secs)
            save_json(m, RES / "baseline_val_metrics.json")
            per_class.to_csv(RES / "baseline_val_per_class.csv")
            save_json({"baseline": secs}, RES / "train_times.json")
        if best is None or m["macro_f1"] > best[1]:
            best = (scheme, m["macro_f1"])

    table = pd.DataFrame(rows).set_index("weighting")
    table.to_csv(RES / "imbalance_comparison_val.csv")
    recalls = pd.DataFrame({s: pd.read_csv(RES / f"baseline_{s}_val_report.csv", index_col=0)["recall"]
                            for s in ("none", "sqrt", "balanced")}).loc[names]
    recalls.to_csv(RES / "imbalance_per_class_recall_val.csv")
    log("per-class recall by weighting scheme (val):\n" + recalls.round(3).to_string())
    # tune stage reads the winning scheme from this file
    json.dump({"weighting": best[0]}, open(RES / "chosen_weighting.json", "w"))
    log(f"chosen weighting for tuning: {best[0]} (best val macro-F1)")


def stage_tune(X, y, names, device, n_iter, frac):
    # RandomizedSearchCV isn't used: it cross-validates on train folds and can't early-stop
    scheme = json.load(open(RES / "chosen_weighting.json"))["weighting"]
    idx = stratified_sample(y["train"], frac)
    Xs, ys = X["train"].loc[idx], y["train"].loc[idx]
    w = class_weights(ys, scheme)
    log(f"tuning on {len(Xs):,} train rows ({frac:.0%} stratified, rare classes kept whole), "
        f"weighting={scheme}, {n_iter} configs, scored on validation")

    out = RES / "tuning_results.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    configs = list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=SEED))
    rows = done.to_dict("records")
    for i, p in enumerate(configs):
        if i < len(done):                                   # resume after an interruption
            continue
        model, secs = fit({**p, "n_estimators": 2000}, Xs, ys, X["val"], y["val"], w, device)
        m, _, _, _ = evaluate(model, X["val"], y["val"], names, "val", fast=True)
        rows.append({"config": i, **p, "best_iteration": model.best_iteration,
                     "val_macro_f1": m["macro_f1"], "val_macro_f1_excl_small": m["macro_f1_excl_small_classes"],
                     "val_mcc": m["mcc"], "val_weighted_f1": m["weighted_f1"],
                     "val_fpr": m["false_positive_rate"], "val_detection": m["attack_detection_rate"],
                     "train_seconds": secs, "inference_us_per_flow": m["inference_us_per_flow"]})
        pd.DataFrame(rows).to_csv(out, index=False)          # save after every config
        log(f"[{i + 1}/{n_iter}] macro-F1 {m['macro_f1']:.4f} | {secs:.0f}s | {p}")

    # best macro-F1 wins, lower FPR breaks ties
    res = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_fpr"], ascending=[False, True])
    log("top 5 configurations (validation):\n" + res.head(5).to_string(index=False))
    best = res.iloc[0]
    best_params = {k: (int(best[k]) if k in ("max_depth", "min_child_weight") else float(best[k]))
                   for k in SEARCH_SPACE}
    json.dump({"params": best_params, "weighting": scheme}, open(RES / "best_params.json", "w"), indent=2)
    log(f"best params saved: {best_params}")


def stage_final(X, y, names, device):
    cfg = json.load(open(RES / "best_params.json"))
    log(f"refitting best config on the FULL train set: {cfg}")
    # more rounds allowed here since the full train set needs longer to converge
    model, secs = fit({**cfg["params"], "n_estimators": 3000}, X["train"], y["train"],
                      X["val"], y["val"], class_weights(y["train"], cfg["weighting"]), device)
    m, rep, cm, per_class = evaluate(model, X["val"], y["val"], names, "val")
    m.update(best_iteration=model.best_iteration, **cfg["params"], weighting=cfg["weighting"])
    rep.to_csv(RES / "tuned_val_report.csv")
    cm.to_csv(RES / "tuned_val_confusion.csv")
    save_cm_png(cm, RES / "tuned_val_confusion.png", "Tuned XGBoost - validation")
    model.save_model(MOD / "xgb_tuned.json")
    joblib.dump(model, MOD / "xgb_tuned.joblib")
    print_benchmark("XGBoost (tuned)", m, per_class, cm, MOD / "xgb_tuned.joblib", secs)
    save_json(m, RES / "tuned_val_metrics.json")
    per_class.to_csv(RES / "tuned_val_per_class.csv")
    times_path = RES / "train_times.json"
    times = json.load(open(times_path)) if times_path.exists() else {}
    save_json({**times, "tuned": secs}, times_path)
    # gain importance; features never used in a split get 0
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(
        X["train"].columns).fillna(0).sort_values(ascending=False)
    imp.to_csv(RES / "tuned_feature_importance_gain.csv", header=["gain"])
    log(f"tuned model saved: {model.best_iteration + 1} rounds, train {secs:.0f}s")


def stage_test(X, y, names, force):
    # lock file stops the test set from being used a second time
    lock = RES / "TEST_SET_USED.txt"
    if lock.exists() and not force:
        raise SystemExit(f"Test set already evaluated ({lock.read_text().strip()}). "
                         "Team rule: touch the test set once. Use --force only if you really must.")
    times_path = RES / "train_times.json"
    times = json.load(open(times_path)) if times_path.exists() else {}
    rows = []
    for tag in ("baseline", "tuned"):
        path = MOD / f"xgb_{tag}.joblib"
        model = joblib.load(path)
        m, rep, cm, per_class = evaluate(model, X["test"], y["test"], names, "test")
        print_benchmark(f"XGBoost ({tag})", m, per_class, cm, path, times.get(tag))
        m["model"] = tag
        rows.append({k: v for k, v in m.items() if k != "small_classes_excluded"})
        save_json(m, RES / f"{tag}_test_metrics.json")
        per_class.to_csv(RES / f"{tag}_test_per_class.csv")
        rep.to_csv(RES / f"{tag}_test_report.csv")
        cm.to_csv(RES / f"{tag}_test_confusion.csv")
        save_cm_png(cm, RES / f"{tag}_test_confusion.png", f"{tag.title()} XGBoost - test")
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(RES / "final_test_metrics.csv")
    lock.write_text(f"evaluated {time.strftime('%Y-%m-%d %H:%M')}")
    log("FINAL TEST RESULTS - baseline vs tuned (report these, do not tune further):\n"
        + table.drop(columns=["split"]).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["baseline", "tune", "final", "test"])
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--n-iter", type=int, default=20, help="random-search configurations")
    ap.add_argument("--tune-frac", type=float, default=0.3, help="train fraction used while tuning")
    ap.add_argument("--quick", action="store_true", help="10%% sample of everything (code check)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    MOD.mkdir(parents=True, exist_ok=True)
    X, y, names = load(a.quick)
    log(f"train {len(X['train']):,} | val {len(X['val']):,} | test {len(X['test']):,} | "
        f"{X['train'].shape[1]} features | {len(names)} classes")

    if a.stage == "baseline":
        stage_baseline(X, y, names, a.device)
    elif a.stage == "tune":
        stage_tune(X, y, names, a.device, a.n_iter, a.tune_frac)
    elif a.stage == "final":
        stage_final(X, y, names, a.device)
    else:
        stage_test(X, y, names, a.force)