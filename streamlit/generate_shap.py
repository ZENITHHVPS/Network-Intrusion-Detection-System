"""
generate_shap.py - Precompute SHAP artifacts for the bundled 85-feature hybrid model.
Configured for: models/hybrid_ae_rf/
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from models.hybrid_ae_rf_model.hybrid_predict import build_fused


def resolve_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main():
    root = Path(__file__).resolve().parent

    # 1. Locate bundle and metadata
    bundle_path = resolve_path([
        root / "models" / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
        root / "hybrid_nids_bundle.joblib"
    ])
    fused_json_path = resolve_path([
        root / "models" / "hybrid_ae_rf" / "fused_feature_names.json",
        root / "fused_feature_names.json"
    ])

    if not bundle_path.exists():
        raise FileNotFoundError(f"Could not locate hybrid_nids_bundle.joblib at: {bundle_path}")

    print(f"[*] Loading hybrid bundle from: {bundle_path}")
    bundle = joblib.load(bundle_path)
    rf_model = bundle["rf"]
    feature_names_68 = bundle["feature_names"]

    # Load 85 fused feature names
    if fused_json_path.exists():
        print(f"[*] Loading fused feature names from: {fused_json_path}")
        with open(fused_json_path, "r") as f:
            fused_feature_names = json.load(f)
    else:
        print("[!] fused_feature_names.json not found; falling back to default naming.")
        fused_feature_names = (
            list(feature_names_68)
            + [f"ae_latent_{i}" for i in range(16)]
            + ["ae_reconstruction_error"]
        )

    # 2. Locate test dataset split
    test_file = resolve_path([
        root / "data" / "processed" / "log1p" / "X_test_supervised.parquet",
        root / "data" / "processed" / "X_test_supervised.parquet",
    ])
    if not test_file.exists():
        raise FileNotFoundError(f"Could not locate X_test_supervised.parquet in data/processed/. Looked at: {test_file}")

    print(f"[*] Reading test data from: {test_file}")
    X_test_raw = pd.read_parquet(test_file)

    # 3. Compute 85 fused features using Autoencoder numpy forward pass
    print("[*] Performing Autoencoder forward pass (68 -> 85 fused features)...")
    X_fused, _ = build_fused(bundle, X_test_raw)
    df_fused = pd.DataFrame(X_fused, columns=fused_feature_names)

    # 4. Sampling for TreeSHAP to optimize RAM and speed
    np.random.seed(42)
    bg_size = min(100, len(df_fused))
    eval_size = min(500, len(df_fused))
    print(f"[*] Sampling {bg_size} background rows and {eval_size} evaluation rows...")

    bg_idx = np.random.choice(len(df_fused), size=bg_size, replace=False)
    eval_idx = np.random.choice(len(df_fused), size=eval_size, replace=False)

    X_bg = df_fused.iloc[bg_idx]
    X_eval = df_fused.iloc[eval_idx]

    # 5. Fit TreeExplainer
    print("[*] Computing TreeSHAP values for Random Forest...")
    explainer = shap.TreeExplainer(rf_model, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer(X_eval, check_additivity=False)

    out_dir = root / "results" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6. Global Summary Plot & CSV
    print("[*] Exporting summary plot and feature rankings...")
    plt.figure(figsize=(10, 6))

    if len(shap_values.shape) == 3:  # Multiclass: (samples, features, classes)
        mean_abs = np.mean(np.abs(shap_values.values), axis=(0, 2))
        top_indices = np.argsort(mean_abs)[::-1][:20]
        top_df = pd.DataFrame({
            "Feature": [fused_feature_names[i] for i in top_indices],
            "Mean_Absolute_SHAP": mean_abs[top_indices]
        })
        top_df.to_csv(out_dir / "top_shap_features.csv", index=False)
        shap.summary_plot(shap_values.values[:, :, 0], X_eval, feature_names=fused_feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_eval, feature_names=fused_feature_names, show=False)

    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_plot.png", dpi=300)
    plt.close()

    # 7. Persist explainer and data subsets
    print("[*] Serializing explainer and sample parquets for Streamlit...")
    joblib.dump(explainer, out_dir / "tree_explainer.joblib")
    X_bg.to_parquet(out_dir / "shap_bg_sample.parquet")
    X_eval.to_parquet(out_dir / "shap_eval_sample.parquet")

    print(f"\n[✓] Done! All SHAP artifacts generated and saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()