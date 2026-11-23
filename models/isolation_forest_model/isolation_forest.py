"""
src/models/isolation_forest.py

Run from the project root:
    python -m src.models.isolation_forest            # headline run -> data/processed
    python -m src.models.isolation_forest --log1p    # ablation run -> data/processed/log1p
    python -m src.models.isolation_forest --quick    # fast pilot (small sample, tiny grid)

Useful options:
    --seeds N        total number of seeds for the robustness check (default 5, 1 = off)
    --bootstrap N    bootstrap resamples for test confidence intervals (default 200, 0 = off)
    --fresh          ignore saved tuning progress and start over

Resume: tuning progress is saved after every fit. If the run is interrupted,
just run the same command again and it continues where it stopped.

NOTE ON METHODOLOGY: hyperparameters and thresholds are chosen on the validation
set using ATTACK LABELS (PR-AUC / best-F1). The model itself is trained on benign
data only, but model selection is therefore semi-supervised, not purely
unsupervised. State this in the write-up.

Outputs:
    models/isolation_forest_<tag>.joblib
    results/isolation_forest/<tag>/results.json
    results/isolation_forest/<tag>/scores_{val,test,heldout_portscan}.parquet
        (score = higher is more anomalous; row index matches the data files)
"""
import argparse
import itertools
import json
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_NAMES = ["best_f1", "p95", "p99"]


#HELPERS
def anomaly_score(model, X):
    """Higher = more anomalous (sklearn's score_samples is the opposite)."""
    return -model.score_samples(X)


def load_labels(path):
    """Read a one-column label parquet as a 1-D int array (safe for 1-row files)."""
    y = pd.read_parquet(path).iloc[:, 0].to_numpy().astype(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{path.name}: labels must be binary 0/1, got {np.unique(y)}")
    if y.min() == y.max():
        raise ValueError(f"{path.name}: only one class present; metrics are undefined")
    return y


def best_f1_threshold(y_true, scores):
    """Threshold that maximises F1 on the validation set."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def choose_thresholds(y_val, val_scores):
    return {
        "best_f1": best_f1_threshold(y_val, val_scores),
        "p95": float(np.percentile(val_scores[y_val == 0], 95)),  # 5% FPR on benign val
        "p99": float(np.percentile(val_scores[y_val == 0], 99)),  # 1% FPR on benign val
    }


def report(name, y_true, scores, thr, verbose=True):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    res = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if verbose:
        print(f"\n[{name}] threshold={thr:.5f}")
        for k, v in res.items():
            print(f"   {k:20s} {v:.4f}" if isinstance(v, float) else f"   {k:20s} {v}")
    return res


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


#LEAKAGE/ DATA SANITY CHECKS (warnings only; they never stop the run)
def row_hashes(df):
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def sanity_checks(frames):
    """frames: dict name -> DataFrame. Prints warnings and returns a summary dict."""
    print("\n[*] Data sanity checks")
    summary = {}
    ref_cols = list(frames["train"].columns)
    for name, df in frames.items():
        if list(df.columns) != ref_cols:
            raise ValueError(f"Feature columns of '{name}' differ from train "
                             f"(names or order). Fix the preprocessing before running.")

    for name, df in frames.items():
        num = df.select_dtypes(include=[np.number])
        n_nan = int(num.isna().sum().sum())
        n_inf = int(np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)).sum())
        dup_names = int(pd.Index(df.columns).duplicated().sum())
        summary[name] = {"rows": len(df), "nan_cells": n_nan, "inf_cells": n_inf,
                         "duplicate_column_names": dup_names}
        if n_nan or n_inf or dup_names:
            print(f"   [WARN] {name}: NaN={n_nan} inf={n_inf} duplicate col names={dup_names}")

    #identical-valued columns (e.g. CICIDS2017's duplicated 'Fwd Header Length'), checked on a sample
    sample = frames["train"].sample(n=min(20_000, len(frames["train"])), random_state=SEED)
    same = sample.T.duplicated()
    if same.any():
        cols = [c for c, d in zip(sample.columns, same) if d]
        print(f"   [WARN] train has columns identical to an earlier column: {cols}")
        summary["identical_columns"] = cols

    #exact-row overlap between splits (near-duplicate flows inflate every metric)
    h_train = row_hashes(frames["train"])
    summary["train_internal_duplicate_frac"] = float(pd.Series(h_train).duplicated().mean())
    print(f"   train: {summary['train_internal_duplicate_frac']:.2%} rows duplicated within the split")
    overlap = {}
    for name in [n for n in frames if n != "train"]:
        h = row_hashes(frames[name])
        frac = float(np.isin(h, h_train).mean())
        overlap[name] = frac
        flag = "  [WARN: check for leakage / dedupe before splitting]" if frac > 0.05 else ""
        print(f"   {name}: {frac:.2%} of rows have an exact copy in train{flag}")
    if "test" in frames and "val" in frames:
        frac = float(np.isin(row_hashes(frames["test"]), row_hashes(frames["val"])).mean())
        overlap["test_in_val"] = frac
        print(f"   test: {frac:.2%} of rows have an exact copy in val")
    if "portscan" in frames and "test" in frames:
        frac = float(np.isin(row_hashes(frames["portscan"]), row_hashes(frames["test"])).mean())
        overlap["portscan_in_test"] = frac
        if frac > 0:
            print(f"   [WARN] {frac:.2%} of held-out PortScan rows also appear in test; "
                  f"PortScan may not be fully held out")
    summary["exact_row_overlap"] = overlap
    print("   (Reminder: confirm PortScan is absent from val/test, and prefer a temporal split by day.)")
    return summary


# UNCERTAINITY: bootstrap CIs and multi-seed robustness
def bootstrap_ci(y, scores, thresholds, n_boot, seed=SEED):
    """Stratified bootstrap 95% CIs for ranking metrics and threshold metrics."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = {"roc_auc": [], "pr_auc": []}
    for name in thresholds:
        for m in ("precision", "recall", "f1", "false_positive_rate"):
            draws[f"{m}@{name}"] = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, sb = y[idx], scores[idx]
        draws["roc_auc"].append(roc_auc_score(yb, sb))
        draws["pr_auc"].append(average_precision_score(yb, sb))
        for name, thr in thresholds.items():
            pred = sb >= thr
            tp = int(((pred == 1) & (yb == 1)).sum())
            fp = int(((pred == 1) & (yb == 0)).sum())
            fn = int(((pred == 0) & (yb == 1)).sum())
            tn = int(((pred == 0) & (yb == 0)).sum())
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            draws[f"precision@{name}"].append(p)
            draws[f"recall@{name}"].append(r)
            draws[f"f1@{name}"].append(2 * p * r / max(p + r, 1e-12))
            draws[f"false_positive_rate@{name}"].append(fp / max(fp + tn, 1))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


def seed_robustness(params, n_seeds, X_train, X_val, y_val, X_test, y_test, primary_model):
    """Refit the best config with several seeds; thresholds are re-derived on val each time."""
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1", "false_positive_rate"]
    per_seed = []
    seeds = [SEED + k for k in range(n_seeds)]
    for s in seeds:
        if s == SEED:
            m = primary_model
        else:
            t0 = time.time()
            m = IsolationForest(contamination="auto", n_jobs=-1, random_state=s, **params)
            m.fit(X_train)
            print(f"   seed {s}: refit in {time.time() - t0:.0f}s")
        vs, ts = anomaly_score(m, X_val), anomaly_score(m, X_test)
        thr = choose_thresholds(y_val, vs)
        row = {"seed": s, "val_pr_auc": float(average_precision_score(y_val, vs))}
        for name in THRESHOLD_NAMES:
            r = report(f"seed {s} @ {name}", y_test, ts, thr[name], verbose=False)
            row[name] = {k: r[k] for k in keys}
        per_seed.append(row)
    summary = {}
    for name in THRESHOLD_NAMES:
        summary[name] = {k: {"mean": float(np.mean([r[name][k] for r in per_seed])),
                             "std": float(np.std([r[name][k] for r in per_seed], ddof=1))
                             if len(per_seed) > 1 else 0.0}
                         for k in keys}
    return {"seeds": seeds, "per_seed": per_seed, "summary": summary}


def zero_day_matched(port_scores, benign_scores, target_prev, n_rep=20, seed=SEED):
    """PR-AUC with the PortScan:benign ratio forced to the main test prevalence."""
    if not 0 < target_prev < 1:
        return None
    rng = np.random.RandomState(seed)
    n_pos = int(min(len(port_scores), round(target_prev / (1 - target_prev) * len(benign_scores))))
    n_neg = int(min(len(benign_scores), round(n_pos * (1 - target_prev) / target_prev)))
    if n_pos < 1 or n_neg < 1:
        return None
    vals = []
    for _ in range(n_rep):
        p = rng.choice(len(port_scores), n_pos, replace=False)
        b = rng.choice(len(benign_scores), n_neg, replace=False)
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        s = np.r_[port_scores[p], benign_scores[b]]
        vals.append(average_precision_score(y, s))
    return {"pr_auc_mean": float(np.mean(vals)), "pr_auc_std": float(np.std(vals)),
            "n_portscan": n_pos, "n_benign": n_neg, "prevalence": n_pos / (n_pos + n_neg)}


######################################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log1p", action="store_true", help="use data/processed/log1p")
    ap.add_argument("--quick", action="store_true",
                    help="pilot run: subsample 50k rows per split, tiny grid")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved tuning progress and start over")
    ap.add_argument("--seeds", type=int, default=5,
                    help="total seeds for the robustness check (1 = skip)")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="bootstrap resamples for test CIs (0 = skip)")
    args = ap.parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.bootstrap = min(args.bootstrap, 20)

    data_dir = ROOT / "data" / "processed" / ("log1p" if args.log1p else "")
    tag = ("log1p" if args.log1p else "main") + ("_quick" if args.quick else "")
    out_dir = ROOT / "results" / "isolation_forest" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    #load
    print(f"[*] Loading data from {data_dir}")
    X_train = pd.read_parquet(data_dir / "X_train_anomaly_benign.parquet")
    X_val = pd.read_parquet(data_dir / "X_val_anomaly.parquet")
    y_val = load_labels(data_dir / "y_val_anomaly_binary.parquet")
    X_test = pd.read_parquet(data_dir / "X_test_anomaly.parquet")
    y_test = load_labels(data_dir / "y_test_anomaly_binary.parquet")
    X_port = pd.read_parquet(data_dir / "X_heldout_portscan.parquet")
    if len(X_val) != len(y_val) or len(X_test) != len(y_test):
        raise ValueError("Feature/label length mismatch in val or test")
    if args.quick:
        rng = np.random.RandomState(SEED)
        n = 50_000
        X_train = X_train.sample(n=min(n, len(X_train)), random_state=SEED)
        vi = np.sort(rng.choice(len(X_val), min(n, len(X_val)), replace=False))
        X_val, y_val = X_val.iloc[vi], y_val[vi]
        ti = np.sort(rng.choice(len(X_test), min(n, len(X_test)), replace=False))
        X_test, y_test = X_test.iloc[ti], y_test[ti]
        X_port = X_port.sample(n=min(n, len(X_port)), random_state=SEED)
        print("[QUICK MODE] using 50k-row subsamples; results are only a sanity check")
    print(f"    train(benign)={len(X_train):,}  val={len(X_val):,}  "
          f"test={len(X_test):,}  portscan={len(X_port):,}")
    print(f"    attack prevalence: val={y_val.mean():.3%}  test={y_test.mean():.3%}")

    sanity = sanity_checks({"train": X_train, "val": X_val, "test": X_test, "portscan": X_port})

    #tune on VALIDATION only (semi-supervised model selection)
    grid = {
        "n_estimators": [100, 200, 300],
        "max_samples": [256, 512, 1024],
        "max_features": [0.5, 1.0],
    }
    if args.quick:
        grid = {"n_estimators": [100], "max_samples": [256], "max_features": [1.0]}
    #resumable tuning
    progress_file = out_dir / "tuning_progress.json"
    best_file = out_dir / "best_so_far.joblib"
    signature = {"grid": grid, "quick": args.quick, "seed": SEED,
                 "n_train": len(X_train), "n_val": len(X_val),
                 "val_attack_rate": round(float(y_val.mean()), 8),
                 "sklearn": sklearn.__version__}
    done = {}   # "n_est|max_s|max_f" -> val PR-AUC
    best = None
    if progress_file.exists() and not args.fresh:
        try:
            saved = json.loads(progress_file.read_text(encoding="utf-8"))
            if saved.get("signature") == signature and best_file.exists():
                done = saved["done"]
                bp = saved["best_params"]
                best = {"pr": saved["best_pr"], "params": bp, "model": joblib.load(best_file)}
                print(f"[*] Resuming: {len(done)} fit(s) already finished.")
            else:
                print("[!] Saved progress is for a different setup; starting fresh.")
        except Exception as e:
            print(f"[!] Could not read saved progress ({e}); starting fresh.")
            done, best = {}, None

    print("\n[*] Tuning on validation set (PR-AUC)...")
    combos = list(itertools.product(*grid.values()))
    for i, (n_est, max_s, max_f) in enumerate(combos, 1):
        key = f"{n_est}|{max_s}|{max_f}"
        if key in done:
            print(f"   [{i}/{len(combos)}] n_est={n_est:3d} max_samples={max_s:4d} "
                  f"max_feat={max_f}  already done (PR-AUC={done[key]:.4f})")
            continue
        t0 = time.time()
        m = IsolationForest(n_estimators=n_est, max_samples=max_s,
                            max_features=max_f, contamination="auto",
                            n_jobs=-1, random_state=SEED)
        m.fit(X_train)
        pr = float(average_precision_score(y_val, anomaly_score(m, X_val)))
        print(f"   [{i}/{len(combos)}] n_est={n_est:3d} max_samples={max_s:4d} "
              f"max_feat={max_f}  val PR-AUC={pr:.4f}  ({time.time() - t0:.0f}s)")
        done[key] = pr
        if best is None or pr > best["pr"]:
            best = {"pr": pr, "model": m,
                    "params": {"n_estimators": n_est, "max_samples": max_s,
                               "max_features": max_f}}
            joblib.dump(m, best_file)
        #save progress (write temp file, then replace, so a shutdown can't corrupt it)
        tmp = progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": signature, "done": done,
                                   "best_pr": best["pr"], "best_params": best["params"]},
                                  indent=2), encoding="utf-8")
        tmp.replace(progress_file)

    model, params = best["model"], best["params"]
    print(f"\n[+] Best params: {params}  (val PR-AUC={best['pr']:.4f})")
    ranked = sorted(done.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1:
        print(f"    gap between best and 2nd-best config: {ranked[0][1] - ranked[1][1]:.4f} "
              f"(compare with the seed std below before reading anything into it)")

    #thresholds from VALIDATION
    val_scores = anomaly_score(model, X_val)
    thresholds = choose_thresholds(y_val, val_scores)

    #evaluate on TEST (once)
    test_scores = anomaly_score(model, X_test)
    results = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_hash": git_hash(),
            "args": vars(args),
            "seed": SEED,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "data_dir": str(data_dir),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "rows": {"train": len(X_train), "val": len(X_val),
                     "test": len(X_test), "portscan": len(X_port)},
            "attack_prevalence": {"val": float(y_val.mean()), "test": float(y_test.mean())},
            "model_selection": "semi-supervised: params/thresholds chosen on validation "
                               "using attack labels; model trained on benign data only",
        },
        "sanity_checks": sanity,
        "tuning": {"val_pr_auc_by_config": dict(ranked)},
        "params": params,
        "thresholds": thresholds,
        "test": {}, "portscan_zero_day": {},
    }
    for name in THRESHOLD_NAMES:
        results["test"][name] = report(f"TEST @ {name}", y_test, test_scores, thresholds[name])

    if args.bootstrap > 0:
        print(f"\n[*] Bootstrap 95% CIs on test ({args.bootstrap} stratified resamples)...")
        results["test_bootstrap_ci95"] = bootstrap_ci(y_test, test_scores, thresholds, args.bootstrap)
        ci = results["test_bootstrap_ci95"]
        print(f"   roc_auc  [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]")
        print(f"   pr_auc   [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}]")
        print(f"   f1@best_f1 [{ci['f1@best_f1'][0]:.4f}, {ci['f1@best_f1'][1]:.4f}]")

    #zero-day PortScan: PortScan (1) + benign test (0)
    port_scores = anomaly_score(model, X_port)
    benign_test_scores = test_scores[y_test == 0]
    y_zd = np.r_[np.ones(len(port_scores)), np.zeros(len(benign_test_scores))]
    s_zd = np.r_[port_scores, benign_test_scores]
    zd = results["portscan_zero_day"]
    zd["n_portscan"] = int(len(port_scores))
    zd["n_benign"] = int(len(benign_test_scores))
    zd["prevalence"] = float(len(port_scores) / (len(port_scores) + len(benign_test_scores)))
    zd["roc_auc"] = float(roc_auc_score(y_zd, s_zd))
    zd["pr_auc"] = float(average_precision_score(y_zd, s_zd))
    zd["pr_auc_random_baseline"] = zd["prevalence"]   # a random scorer's PR-AUC
    for name in THRESHOLD_NAMES:
        zd[f"recall@{name}"] = float((port_scores >= thresholds[name]).mean())
    matched = zero_day_matched(port_scores, benign_test_scores, float(y_test.mean()))
    if matched is not None:
        zd["prevalence_matched_to_test"] = matched
    print("\n[PortScan zero-day]")
    for k, v in zd.items():
        if isinstance(v, dict):
            print(f"   {k:28s} PR-AUC={v['pr_auc_mean']:.4f} +/- {v['pr_auc_std']:.4f} "
                  f"(prevalence {v['prevalence']:.3%})")
        elif isinstance(v, float):
            print(f"   {k:28s} {v:.4f}")
        else:
            print(f"   {k:28s} {v}")

    #seed robustness
    if args.seeds > 1:
        print(f"\n[*] Seed robustness: best config over {args.seeds} seeds...")
        results["seed_robustness"] = seed_robustness(
            params, args.seeds, X_train, X_val, y_val, X_test, y_test, model)
        s = results["seed_robustness"]["summary"]["best_f1"]
        print("   test @ best_f1 (mean +/- std over seeds):")
        for k in ("pr_auc", "roc_auc", "f1", "recall", "false_positive_rate"):
            print(f"      {k:20s} {s[k]['mean']:.4f} +/- {s[k]['std']:.4f}")

    #save per-row scores / predictions
    def scores_frame(index, score, y=None):
        df = pd.DataFrame({"score": score.astype("float32")}, index=index)
        if y is not None:
            df["y_true"] = y.astype("int8")
        for name in THRESHOLD_NAMES:
            df[f"pred_{name}"] = (score >= thresholds[name]).astype("int8")
        return df

    scores_frame(X_val.index, val_scores, y_val).to_parquet(out_dir / "scores_val.parquet")
    scores_frame(X_test.index, test_scores, y_test).to_parquet(out_dir / "scores_test.parquet")
    scores_frame(X_port.index, port_scores).to_parquet(out_dir / "scores_heldout_portscan.parquet")

    #save
    joblib.dump(model, model_dir / f"isolation_forest_{tag}.joblib")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[✓] Model -> {model_dir / f'isolation_forest_{tag}.joblib'}\n[✓] Results -> {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    main()
