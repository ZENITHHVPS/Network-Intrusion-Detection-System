#!/usr/bin/env python
"""
Member 1 - Random Forest known-attack classification.
CICIDS2017 Hybrid NIDS project.


Run from project root:
  python src/models/train_random_forest.py --stage baseline
  python src/models/train_random_forest.py --stage tune
  python src/models/train_random_forest.py --stage final
  python src/models/train_random_forest.py --stage test
"""


import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


from sklearn.model_selection import ParameterSampler




# ============================================================
# 1. CONFIGURATION
# ============================================================


SEED = 42


ROOT = Path(__file__).resolve().parents[2]


DATA = ROOT / "data" / "processed"


RES = ROOT / "results" / "random_forest"


MOD = ROOT / "models" / "random_forest"




# Random Forest baseline configuration


BASELINE_PARAMS = {
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
}


TUNE_TREES = 200
FINAL_TREES = 500




# Random Forest hyperparameter search space


SEARCH_SPACE = {
    "max_depth": [20, 30, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 0.8],
    "max_samples": [None, 0.8],
}




def log(message):
    print(f"[RF] {message}", flush=True)




def save_json(path, data):


    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8"
    )




# ============================================================
# 2. DATA LOADING
# ============================================================


def stratified_sample(y, fraction, min_per_class=2000):


    rng = np.random.RandomState(SEED)


    chosen = []


    for _, indexes in y.groupby(y).groups.items():


        indexes = np.asarray(
            list(indexes),
            dtype=np.int64
        )


        count = min(
            len(indexes),
            max(
                int(np.ceil(fraction * len(indexes))),
                min_per_class
            )
        )


        chosen.append(
            rng.choice(
                indexes,
                count,
                replace=False
            )
        )


    return pd.Index(np.concatenate(chosen))




def load(quick=False):


    X = {}
    y = {}


    for split in ("train", "val", "test"):


        xp = DATA / f"X_{split}_supervised.parquet"


        yp = DATA / f"y_{split}_supervised.parquet"


        if not xp.is_file() or not yp.is_file():


            raise FileNotFoundError(
                f"Missing supervised files: {xp} or {yp}"
            )


        X[split] = pd.read_parquet(
            xp
        ).reset_index(drop=True)


        yf = pd.read_parquet(yp)


        if "Label" not in yf.columns:


            raise ValueError(
                f"{yp.name} must contain a Label column"
            )


        y[split] = (
            yf["Label"]
            .astype("int32")
            .reset_index(drop=True)
        )


        if len(X[split]) != len(y[split]) or X[split].empty:


            raise ValueError(
                f"Empty data or feature/label mismatch in {split}"
            )


        if not X[split].columns.is_unique:


            raise ValueError(
                f"Duplicate feature columns in {split}"
            )


        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X[split].dtypes
        ):


            raise TypeError(
                f"Non-numeric features in {split}"
            )


        for column in X[split].columns:


            if not np.isfinite(
                X[split][column].to_numpy()
            ).all():


                raise ValueError(
                    f"Invalid values in {split}.{column}"
                )


    features = list(X["train"].columns)


    for split in ("val", "test"):


        if list(X[split].columns) != features:


            raise ValueError(
                f"Feature mismatch in {split}"
            )


    mapping_file = DATA / "label_mapping.json"


    mapping = json.loads(
        mapping_file.read_text(encoding="utf-8")
    )


    if not isinstance(mapping, dict) or not mapping:


        raise ValueError(
            "Invalid label_mapping.json"
        )


    if sorted(mapping.values()) != list(range(len(mapping))):


        raise ValueError(
            "Class IDs must be consecutive integers starting at 0"
        )


    names = [
        key
        for key, value in sorted(
            mapping.items(),
            key=lambda item: item[1]
        )
    ]


    valid = set(mapping.values())


    for split in X:


        found = set(
            int(value)
            for value in y[split].unique()
        )


        if not found.issubset(valid):


            raise ValueError(
                f"Unmapped classes in {split}"
            )


    if set(y["train"].unique()) != valid:


        raise ValueError(
            "Not every mapped class is present in training data"
        )


    feature_file = DATA / "feature_names.json"


    if feature_file.exists():


        saved = json.loads(
            feature_file.read_text(encoding="utf-8")
        )


        if isinstance(saved, list) and saved != features:


            raise ValueError(
                "feature_names.json differs from training columns"
            )


    if quick:


        for split in X:


            indexes = stratified_sample(
                y[split],
                0.10,
                min_per_class=50
            )


            X[split] = X[split].loc[indexes].reset_index(drop=True)


            y[split] = y[split].loc[indexes].reset_index(drop=True)


    return X, y, names




# ============================================================
# 3. CLASS IMBALANCE HANDLING
# ============================================================


def class_weights(y, scheme):


    if scheme == "none":
        return None


    counts = y.value_counts()


    weights = len(y) / (len(counts) * counts)


    if scheme == "sqrt":


        weights = np.sqrt(weights)


    weights = (
        weights / (weights * counts).sum() * len(y)
    )


    return {
        int(key): float(value)
        for key, value in weights.items()
    }




# ============================================================
# 4. MODEL TRAINING
# ============================================================


def fit(params, Xtr, ytr, scheme, n_trees):


    model = RandomForestClassifier(


        **params,


        n_estimators=n_trees,


        class_weight=class_weights(ytr, scheme),


        random_state=SEED,


        n_jobs=-1


    )


    start = time.perf_counter()


    model.fit(
        Xtr,
        ytr
    )


    elapsed = time.perf_counter() - start


    return model, elapsed




# ============================================================
# 5. EVALUATION METRICS
# ============================================================


def evaluate(model, X, y, names, split):


    start = time.perf_counter()


    predictions = model.predict(X)


    inference = time.perf_counter() - start


    yt = y.to_numpy()


    labels = list(range(len(names)))


    benign = yt == 0


    metrics = {


        "split": split,


        "accuracy": float(
            accuracy_score(yt, predictions)
        ),


        "macro_precision": float(
            precision_score(
                yt,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),


        "macro_recall": float(
            recall_score(
                yt,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),


        "macro_f1": float(
            f1_score(
                yt,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),


        "weighted_f1": float(
            f1_score(
                yt,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0
            )
        ),


        "false_positive_rate": (
            float((predictions[benign] != 0).mean())
            if benign.any() else 0.0
        ),


        "attack_detection_rate": (
            float((predictions[~benign] != 0).mean())
            if (~benign).any() else 0.0
        ),


        "inference_seconds": float(inference),


        "inference_us_per_flow": float(
            1e6 * inference / len(X)
        ),


        "n_rows": int(len(X))


    }


    report = pd.DataFrame(
        classification_report(
            yt,
            predictions,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0
        )
    ).T


    cm = pd.DataFrame(
        confusion_matrix(
            yt,
            predictions,
            labels=labels
        ),
        index=names,
        columns=names
    )


    return metrics, report, cm




# ============================================================
# 6. CONFUSION MATRIX IMAGE
# ============================================================


def save_confusion_png(cm, path, title):


    try:


        import matplotlib.pyplot as plt


        import seaborn as sns


    except ImportError:


        return


    normalized = cm.div(
        cm.sum(axis=1).replace(0, 1),
        axis=0
    )


    plt.figure(figsize=(11, 9))


    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        vmin=0,
        vmax=1
    )


    plt.title(title)


    plt.ylabel("True")


    plt.xlabel("Predicted")


    plt.tight_layout()


    plt.savefig(path, dpi=150)


    plt.close()




# ============================================================
# 7. BASELINE AND CLASS IMBALANCE COMPARISON
# ============================================================


def stage_baseline(X, y, names):


    rows = []


    best = None


    distribution = pd.DataFrame(
        {
            split: y[split].value_counts()
            for split in y
        }
    ).fillna(0).astype(int)


    distribution.index = [
        names[int(i)]
        for i in distribution.index
    ]


    distribution["train_percent"] = (
        100 * distribution["train"] /
        distribution["train"].sum()
    ).round(4)


    distribution.to_csv(
        RES / "class_distribution.csv"
    )


    for scheme in ("none", "sqrt", "balanced"):


        log(
            f"Baseline: class weighting = {scheme}"
        )


        model, seconds = fit(
            BASELINE_PARAMS,
            X["train"],
            y["train"],
            scheme,
            300
        )


        metrics, report, cm = evaluate(
            model,
            X["val"],
            y["val"],
            names,
            "val"
        )


        metrics.update(
            weighting=scheme,
            train_seconds=seconds,
            n_estimators=300
        )


        rows.append(metrics)


        report.to_csv(
            RES / f"baseline_{scheme}_val_report.csv"
        )


        log(
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"FPR: {metrics['false_positive_rate']:.4%} | "
            f"Time: {seconds:.1f}s"
        )


        if scheme == "none":


            joblib.dump(
                model,
                MOD / "rf_baseline.joblib",
                compress=3
            )


            cm.to_csv(
                RES / "baseline_val_confusion.csv"
            )


            save_confusion_png(
                cm,
                RES / "baseline_val_confusion.png",
                "Baseline Random Forest - validation"
            )


        if best is None or metrics["macro_f1"] > best[1]:


            best = (
                scheme,
                metrics["macro_f1"]
            )


    pd.DataFrame(rows).set_index(
        "weighting"
    ).to_csv(
        RES / "imbalance_comparison_val.csv"
    )


    save_json(
        RES / "chosen_weighting.json",
        {"weighting": best[0]}
    )


    log(
        f"Chosen weighting: {best[0]} "
        "(best validation Macro-F1)"
    )




# ============================================================
# 8. HYPERPARAMETER TUNING
# ============================================================


def stage_tune(X, y, names, n_iter, fraction):


    choice_file = RES / "chosen_weighting.json"


    if not choice_file.exists():


        raise FileNotFoundError(
            "Run --stage baseline first"
        )


    scheme = json.loads(
        choice_file.read_text(encoding="utf-8")
    )["weighting"]


    indexes = stratified_sample(
        y["train"],
        fraction
    )


    X_sample = X["train"].loc[indexes]


    y_sample = y["train"].loc[indexes]


    log(
        f"Tuning rows: {len(X_sample):,} | "
        f"Weighting: {scheme} | "
        f"Configurations: {n_iter}"
    )


    output = RES / "tuning_results.csv"


    previous = (
        pd.read_csv(output)
        if output.exists()
        else pd.DataFrame()
    )


    configs = list(
        ParameterSampler(
            SEARCH_SPACE,
            n_iter=n_iter,
            random_state=SEED
        )
    )


    rows = previous.to_dict("records")


    for i, params in enumerate(configs):


        if i < len(previous):
            continue


        model, seconds = fit(
            params,
            X_sample,
            y_sample,
            scheme,
            TUNE_TREES
        )


        metrics, _, _ = evaluate(
            model,
            X["val"],
            y["val"],
            names,
            "val"
        )


        rows.append({


            "config": i,


            **params,


            "val_macro_f1": metrics["macro_f1"],


            "val_weighted_f1": metrics["weighted_f1"],


            "val_fpr": metrics["false_positive_rate"],


            "val_detection": metrics["attack_detection_rate"],


            "train_seconds": seconds,


            "inference_us_per_flow": metrics["inference_us_per_flow"]


        })


        pd.DataFrame(rows).to_csv(
            output,
            index=False
        )


        log(
            f"[{i + 1}/{n_iter}] "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Training: {seconds:.1f}s"
        )


    ranked = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_fpr"],
        ascending=[False, True]
    )


    best = ranked.iloc[0]


    params = {}


    for key in SEARCH_SPACE:


        value = best[key]


        if pd.isna(value):
            value = None


        elif key in (
            "max_depth",
            "min_samples_split",
            "min_samples_leaf"
        ):
            value = int(value)


        elif key == "max_features" and value not in ("sqrt", "log2"):
            value = float(value)


        elif key == "max_samples" and value is not None:
            value = float(value)


        params[key] = value


    save_json(
        RES / "best_params.json",
        {
            "params": params,
            "weighting": scheme
        }
    )


    log(
        f"Best validation configuration saved: {params}"
    )




# ============================================================
# 9. FINAL TRAINING ON COMPLETE TRAINING DATASET
# ============================================================


def stage_final(X, y, names):


    config_file = RES / "best_params.json"


    if not config_file.exists():


        raise FileNotFoundError(
            "Run --stage tune first"
        )


    config = json.loads(
        config_file.read_text(encoding="utf-8")
    )


    params = config["params"]


    scheme = config["weighting"]


    log(
        f"FINAL TRAINING USING FULL TRAINING SPLIT: "
        f"{len(X['train']):,} rows"
    )


    model, seconds = fit(
        params,
        X["train"],
        y["train"],
        scheme,
        FINAL_TREES
    )


    metrics, report, cm = evaluate(
        model,
        X["val"],
        y["val"],
        names,
        "val"
    )


    metrics.update(
        train_seconds=seconds,
        n_estimators=FINAL_TREES,
        weighting=scheme,
        **params
    )


    save_json(
        RES / "tuned_val_metrics.json",
        metrics
    )


    report.to_csv(
        RES / "tuned_val_report.csv"
    )


    cm.to_csv(
        RES / "tuned_val_confusion.csv"
    )


    save_confusion_png(
        cm,
        RES / "tuned_val_confusion.png",
        "Tuned Random Forest - validation"
    )


    joblib.dump(
        model,
        MOD / "rf_tuned.joblib",
        compress=3
    )


    importance = pd.Series(
        model.feature_importances_,
        index=X["train"].columns,
        name="importance"
    ).sort_values(
        ascending=False
    )


    importance.to_csv(
        RES / "tuned_feature_importance.csv"
    )


    save_json(
        RES / "final_training_info.json",
        {
            "algorithm": "RandomForestClassifier",
            "training_rows": int(len(X["train"])),
            "features": int(X["train"].shape[1]),
            "classes": int(len(names)),
            "n_estimators": FINAL_TREES,
            "training_seconds": seconds,
            "training_minutes": seconds / 60,
            "weighting": scheme,
            "parameters": params
        }
    )


    log(
        f"FINAL TRAINING COMPLETED: {seconds / 60:.2f} minutes"
    )


    log(
        f"VALIDATION MACRO-F1: {metrics['macro_f1']:.4f}"
    )




# ============================================================
# 10. ONE-TIME FINAL TEST EVALUATION
# ============================================================


def stage_test(X, y, names, force=False):


    lock = RES / "TEST_SET_USED.txt"


    if lock.exists() and not force:


        raise SystemExit(
            "Test set already evaluated. "
            "Use --force only if you intentionally accept "
            "a repeat test evaluation."
        )


    rows = []


    for tag in ("baseline", "tuned"):


        model_path = MOD / f"rf_{tag}.joblib"


        if not model_path.exists():


            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )


        model = joblib.load(model_path)


        metrics, report, cm = evaluate(
            model,
            X["test"],
            y["test"],
            names,
            "test"
        )


        metrics["model"] = tag


        rows.append(metrics)


        report.to_csv(
            RES / f"{tag}_test_report.csv"
        )


        cm.to_csv(
            RES / f"{tag}_test_confusion.csv"
        )


        save_confusion_png(
            cm,
            RES / f"{tag}_test_confusion.png",
            f"{tag.title()} Random Forest - test"
        )


    table = pd.DataFrame(rows).set_index("model")


    table.to_csv(
        RES / "final_test_metrics.csv"
    )


    lock.write_text(
        f"evaluated {time.strftime('%Y-%m-%d %H:%M')}",
        encoding="utf-8"
    )


    log(
        "FINAL TEST RESULTS:\n" + table.T.to_string()
    )




# ============================================================
# 11. MAIN FUNCTION
# ============================================================


def main():


    parser = argparse.ArgumentParser(
        description="CICIDS2017 Random Forest training"
    )


    parser.add_argument(
        "--stage",
        required=True,
        choices=["baseline", "tune", "final", "test"]
    )


    parser.add_argument(
        "--n-iter",
        type=int,
        default=20
    )


    parser.add_argument(
        "--tune-frac",
        type=float,
        default=0.30
    )


    parser.add_argument(
        "--quick",
        action="store_true"
    )


    parser.add_argument(
        "--force",
        action="store_true"
    )


    args = parser.parse_args()


    if args.n_iter < 1:
        parser.error("--n-iter must be at least 1")


    if not 0 < args.tune_frac <= 1:
        parser.error("--tune-frac must be in (0, 1]")


    RES.mkdir(parents=True, exist_ok=True)


    MOD.mkdir(parents=True, exist_ok=True)


    X, y, names = load(args.quick)


    log(
        f"TRAIN: {len(X['train']):,} | "
        f"VAL: {len(X['val']):,} | "
        f"TEST: {len(X['test']):,} | "
        f"FEATURES: {X['train'].shape[1]} | "
        f"CLASSES: {len(names)}"
    )


    if args.stage == "baseline":


        stage_baseline(X, y, names)


    elif args.stage == "tune":


        stage_tune(
            X,
            y,
            names,
            args.n_iter,
            args.tune_frac
        )


    elif args.stage == "final":


        stage_final(X, y, names)


    else:


        stage_test(
            X,
            y,
            names,
            args.force
        )




if __name__ == "__main__":


    main()
