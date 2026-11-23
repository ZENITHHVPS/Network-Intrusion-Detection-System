from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ----------------------------- Project Paths ----------------------------------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
LOG1P_DATA = DATA / "log1p"
OUT_DIR = ROOT / "results" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(candidates):
    for path in candidates:
        if path and Path(path).is_file():
            return Path(path)
    return None


def read_json(path):
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_label_map():
    for path in (LOG1P_DATA / "label_mapping.json", DATA / "label_mapping.json"):
        mapping = read_json(path)
        if isinstance(mapping, dict) and mapping:
            return {int(v): str(k) for k, v in mapping.items()}
    return {}


ID_TO_LABEL = load_label_map()


def save_beeswarm(shap_vals_2d, X_sample, title, out_path, max_display=15):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_2d,
        X_sample,
        max_display=max_display,
        show=False,
    )
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"[SAVED] {out_path}")


def extract_2d_shap(raw_shap, class_idx=0):
    """Normalize SHAP output across SHAP versions (list of 2D arrays vs 3D ndarray vs 2D array)."""
    if isinstance(raw_shap, list):
        return np.asarray(raw_shap[class_idx])
    arr = np.asarray(raw_shap)
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP array shape: {arr.shape}")


# ===================== 1. RANDOM FOREST & 2. XGBOOST ==========================
def run_supervised_shap(model_name, model_path, n_samples=1000):
    if not model_path:
        print(f"[SKIP] {model_name}: model file not found.")
        return

    X_path = first_existing([
        LOG1P_DATA / "X_test_supervised.parquet",
        DATA / "X_test_supervised.parquet",
    ])
    if not X_path:
        print(f"[SKIP] {model_name}: X_test_supervised.parquet not found.")
        return

    print(f"\n=== [SHAP] {model_name} ({model_path.name}) ===")
    model = joblib.load(model_path)
    X = pd.read_parquet(X_path)

    feature_names = getattr(model, "feature_names_in_", None)
    feature_names = list(feature_names) if feature_names is not None else list(X.columns)
    X_aligned = X.loc[:, feature_names]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_aligned), size=min(n_samples, len(X_aligned)), replace=False)
    X_sample = X_aligned.iloc[idx].copy()

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_sample)

    classes = list(getattr(model, "classes_", []))
    class_labels = [ID_TO_LABEL.get(int(c), str(c)) for c in classes]
    slug = model_name.lower().replace(" ", "_")

    # BENIGN beeswarm plot
    benign_idx = next(
        (i for i, lbl in enumerate(class_labels) if lbl.strip().upper() in {"BENIGN", "NORMAL"}),
        0,
    )
    sv_benign = extract_2d_shap(raw_shap, benign_idx)
    save_beeswarm(
        sv_benign,
        X_sample,
        f"{model_name} — SHAP Summary ({class_labels[benign_idx]})",
        OUT_DIR / f"{slug}_shap_beeswarm_benign.png",
    )

    # Top attack class beeswarm plot
    attack_indices = [i for i in range(len(class_labels)) if i != benign_idx]
    if attack_indices:
        top_attack_idx = max(
            attack_indices,
            key=lambda i: float(np.mean(np.abs(extract_2d_shap(raw_shap, i)))),
        )
        attack_lbl = class_labels[top_attack_idx]
        safe_lbl = "".join(c if c.isalnum() else "_" for c in attack_lbl).strip("_")
        sv_attack = extract_2d_shap(raw_shap, top_attack_idx)
        save_beeswarm(
            sv_attack,
            X_sample,
            f"{model_name} — SHAP Summary ({attack_lbl})",
            OUT_DIR / f"{slug}_shap_beeswarm_{safe_lbl}.png",
        )

    # Global mean |SHAP| CSV
    if isinstance(raw_shap, list):
        mean_abs = np.mean([np.mean(np.abs(s), axis=0) for s in raw_shap], axis=0)
    else:
        arr = np.asarray(raw_shap)
        mean_abs = np.mean(np.abs(arr), axis=(0, 2)) if arr.ndim == 3 else np.mean(np.abs(arr), axis=0)

    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(OUT_DIR / f"{slug}_shap_importance.csv", index=False)


# ======================== 3. ISOLATION FOREST =================================
def run_isolation_forest_shap(n_samples=1000):
    if_path = first_existing([
        MODELS / "isolation_forest_model" / "isolation_forest" / "isolation_forest_log1p.joblib",
        MODELS / "isolation_forest_model" / "isolation_forest" / "isolation_forest_log1p_ae_basis.joblib",
        MODELS / "isolation_forest" / "isolation_forest_log1p.joblib",
    ])
    X_path = first_existing([
        LOG1P_DATA / "X_test_anomaly.parquet",
        DATA / "X_test_anomaly.parquet",
    ])
    if not if_path or not X_path:
        print("[SKIP] Isolation Forest: model or X_test_anomaly.parquet not found.")
        return

    print(f"\n=== [3/5 SHAP] Isolation Forest ({if_path.name}) ===")
    model = joblib.load(if_path)
    X = pd.read_parquet(X_path)

    feature_names = getattr(model, "feature_names_in_", None)
    feature_names = list(feature_names) if feature_names is not None else list(X.columns)
    X_aligned = X.loc[:, feature_names]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_aligned), size=min(n_samples, len(X_aligned)), replace=False)
    X_sample = X_aligned.iloc[idx].copy()

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_sample)
    sv = extract_2d_shap(raw_shap, 0)

    save_beeswarm(
        sv,
        X_sample,
        "Isolation Forest — SHAP Anomaly Summary",
        OUT_DIR / "isolation_forest_shap_beeswarm.png",
    )


# ======================== 4. HYBRID AE + RF ===================================
def run_hybrid_shap(n_samples=1000):
    bundle_path = first_existing([
        MODELS / "hybrid_ae_rf_model" / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
        MODELS / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
    ])
    hybrid_rf_path = first_existing([
        MODELS / "hybrid_ae_rf_model" / "hybrid_ae_rf" / "hybrid_rf.joblib",
        MODELS / "hybrid_ae_rf" / "hybrid_rf.joblib",
    ])
    fused_names_path = first_existing([
        MODELS / "hybrid_ae_rf_model" / "hybrid_ae_rf" / "fused_feature_names.json",
        MODELS / "hybrid_ae_rf" / "fused_feature_names.json",
    ])
    ae_path = first_existing([
        MODELS / "autoencoder_model" / "autoencoder" / "best_autoencoder_log1p.keras",
        MODELS / "autoencoder" / "best_autoencoder_log1p.keras",
    ])
    X_path = first_existing([
        LOG1P_DATA / "X_test_anomaly.parquet",
        DATA / "X_test_anomaly.parquet",
    ])

    if not X_path or (not bundle_path and not hybrid_rf_path):
        print("[SKIP] Hybrid AE + RF: bundle/model or X_test_anomaly.parquet not found.")
        return

    print("\n=== [4/5 SHAP] Hybrid AE + RF ===")
    X = pd.read_parquet(X_path)
    bundle = joblib.load(bundle_path) if bundle_path else {}

    rf_model = None
    for key in ("rf_model", "rf", "classifier", "random_forest", "model"):
        if key in bundle and hasattr(bundle[key], "predict_proba"):
            rf_model = bundle[key]
            break
    if rf_model is None and hybrid_rf_path:
        rf_model = joblib.load(hybrid_rf_path)

    if rf_model is None:
        print("[SKIP] Hybrid AE + RF: could not extract RF estimator.")
        return

    base_features = list(bundle.get("feature_names") or read_json(LOG1P_DATA / "feature_names.json") or X.columns)
    X_base = X.loc[:, base_features]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_base), size=min(n_samples, len(X_base)), replace=False)
    X_sample = X_base.iloc[idx].copy()

    fused_names = read_json(fused_names_path) if fused_names_path else list(getattr(rf_model, "feature_names_in_", []))
    n_expected = rf_model.n_features_in_

    if n_expected == len(base_features):
        X_for_rf = X_sample
    elif ae_path:
        from tensorflow.keras.models import load_model, Model
        ae = load_model(ae_path, compile=False)
        vals = X_sample.to_numpy(dtype=np.float32)
        recon = ae.predict(vals, verbose=0, batch_size=512)
        ae_mse = np.mean(np.square(vals - recon), axis=1, keepdims=True)
        diff = n_expected - len(base_features)

        if diff == 1:
            extra = ae_mse
        else:
            # Ensure Sequential/Functional model is built and inspect layer.output.shape (Keras 3 safe)
            _ = ae(vals[:1])
            candidate_layers = [
                layer for layer in ae.layers[:-1]
                if hasattr(layer, "output")
                and len(layer.output.shape) == 2
                and layer.output.shape[1] is not None
            ]
            bottleneck_layer = min(candidate_layers, key=lambda layer: int(layer.output.shape[1]))
            encoder = Model(inputs=ae.inputs, outputs=bottleneck_layer.output)
            latent = encoder.predict(vals, verbose=0, batch_size=512)
            if latent.shape[1] + 1 == diff:
                extra = np.hstack([latent, ae_mse])
            elif latent.shape[1] == diff:
                extra = latent
            else:
                print(f"[INFO] Hybrid RF expects {n_expected} features (latent={latent.shape[1]}); skipping.")
                return

        col_names = fused_names if len(fused_names) == n_expected else (
            base_features + [f"ae_feat_{i}" for i in range(diff)]
        )
        X_for_rf = pd.DataFrame(np.hstack([vals, extra]), columns=col_names)
    else:
        print("[SKIP] Hybrid AE + RF: Autoencoder model needed to construct fused features.")
        return

    explainer = shap.TreeExplainer(rf_model)
    raw_shap = explainer.shap_values(X_for_rf)
    sv = extract_2d_shap(raw_shap, 0)

    save_beeswarm(
        sv,
        X_for_rf,
        "Hybrid AE + RF — SHAP Summary",
        OUT_DIR / "hybrid_ae_rf_shap_beeswarm.png",
    )


# ======================== 5. DEEP AUTOENCODER =================================
def run_autoencoder_shap(n_samples=60):
    ae_path = first_existing([
        MODELS / "autoencoder_model" / "autoencoder" / "best_autoencoder_log1p.keras",
        MODELS / "autoencoder" / "best_autoencoder_log1p.keras",
    ])
    X_path = first_existing([
        LOG1P_DATA / "X_test_anomaly.parquet",
        DATA / "X_test_anomaly.parquet",
    ])
    feat_path = first_existing([
        LOG1P_DATA / "feature_names.json",
        DATA / "feature_names.json",
    ])

    if not ae_path or not X_path or not feat_path:
        print("[SKIP] Deep Autoencoder: model, data, or feature_names.json not found.")
        return

    print(f"\n=== [5/5 SHAP] Deep Autoencoder ({ae_path.name}) ===")
    from tensorflow.keras.models import load_model
    ae = load_model(ae_path, compile=False)
    feature_names = read_json(feat_path)
    X = pd.read_parquet(X_path).loc[:, feature_names]

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=min(40, len(X)), replace=False)
    test_idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    X_bg = X.iloc[bg_idx].to_numpy(dtype=np.float32)
    X_test_sub = X.iloc[test_idx]

    def ae_mse_score(data_np):
        arr = np.asarray(data_np, dtype=np.float32)
        recon = ae.predict(arr, verbose=0, batch_size=256)
        return np.mean(np.square(arr - recon), axis=1)

    explainer = shap.KernelExplainer(ae_mse_score, X_bg)
    sv = explainer.shap_values(X_test_sub.to_numpy(dtype=np.float32), nsamples=100)

    save_beeswarm(
        np.asarray(sv),
        X_test_sub,
        "Deep Autoencoder — SHAP Reconstruction Error Summary",
        OUT_DIR / "autoencoder_shap_beeswarm.png",
    )


def main():
    rf_path = first_existing([
        MODELS / "random_forest_model" / "random_forest" / "random_forest_final.joblib",
        MODELS / "random_forest_model" / "random_forest" / "random_forest_selected.joblib",
        MODELS / "random_forest_model" / "random_forest" / "rf_tuned.joblib",
        MODELS / "random_forest_model" / "random_forest" / "rf_baseline.joblib",
        MODELS / "random_forest" / "rf_tuned.joblib",
        MODELS / "random_forest" / "rf_baseline.joblib",
    ])

    xgb_path = first_existing([
        MODELS / "xgboost_model" / "xgboost" / "xgb_tuned.joblib",
        MODELS / "xgboost_model" / "xgboost" / "xgb_baseline.joblib",
        MODELS / "xgboost" / "xgb_tuned.joblib",
        MODELS / "xgboost" / "xgb_baseline.joblib",
    ])

    run_supervised_shap("Random Forest", rf_path, n_samples=1000)
    run_supervised_shap("XGBoost", xgb_path, n_samples=1000)
    run_isolation_forest_shap(n_samples=1000)
    run_hybrid_shap(n_samples=1000)
    run_autoencoder_shap(n_samples=60)

    print(f"\nAll done! Check your generated SHAP plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()