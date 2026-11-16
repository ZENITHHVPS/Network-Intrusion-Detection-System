"""
app.py - Hardened Master Streamlit Dashboard for Hybrid NIDS and SHAP XAI
"""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Safe import of hybrid_predict functions
try:
    from models.hybrid_ae_rf_model.hybrid_predict import build_fused, predict
except ImportError:
    st.error("Error: 'hybrid_predict.py' not found in current working directory.")
    st.stop()

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="Hybrid NIDS & SHAP XAI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

def find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

# ----------------- Cached Loaders -----------------
@st.cache_resource
def load_bundle():
    bundle_path = find_file([
        ROOT / "models" / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
        ROOT / "hybrid_nids_bundle.joblib"
    ])
    if not bundle_path.exists():
        st.error(f"Cannot find hybrid bundle file. Checked: {bundle_path}")
        st.stop()
    return joblib.load(bundle_path)

@st.cache_resource
def load_shap_artifacts():
    out_dir = ROOT / "results" / "shap"
    explainer_path = out_dir / "tree_explainer.joblib"
    eval_path = out_dir / "shap_eval_sample.parquet"

    if not explainer_path.exists() or not eval_path.exists():
        return None, None

    explainer = joblib.load(explainer_path)
    eval_df = pd.read_parquet(eval_path)
    return explainer, eval_df

@st.cache_data
def load_feature_names(feature_names_bundle):
    fused_json = find_file([
        ROOT / "models" / "hybrid_ae_rf" / "fused_feature_names.json",
        ROOT / "fused_feature_names.json"
    ])
    if fused_json.exists():
        with open(fused_json, "r") as f:
            return json.load(f)
    return list(feature_names_bundle) + [f"ae_latent_{i}" for i in range(16)] + ["ae_reconstruction_error"]

# Load core artifacts
bundle = load_bundle()
fused_names = load_feature_names(bundle["feature_names"])

# ----------------- Navigation -----------------
st.sidebar.title("🛡️ Hybrid NIDS")
page = st.sidebar.radio(
    "Navigation Menu",
    ["1. System Overview", "2. Live Hybrid Inspection", "3. SHAP Explainability"]
)

# ----------------- 1. System Overview -----------------
if page == "1. System Overview":
    st.title("🛡️ Hybrid Network Intrusion Detection System")
    st.markdown("""
    ### System Architecture
    This prototype fuses **Random Forest** supervised classification with **Deep Autoencoder** reconstruction error:
    * **Feature Fusion:** 68 traffic features pass through an Autoencoder forward pass in NumPy, producing 16 latent representations and 1 reconstruction MSE feature (85 fused features total).
    * **Supervised Layer:** Evaluates the flow against learned attack signatures.
    * **Dual-Engine Gate Rule:** When the supervised model predicts normal traffic ($p_{\\text{benign}} < p_{\\text{cut}}$) but the Autoencoder reconstruction error exceeds threshold $\\tau$, the decision is overridden to **`ANOMALY`**.
    """)

    st.subheader("Configured Gate Settings")
    st.json(bundle.get("gate_configs", {}))

# ----------------- 2. Live Hybrid Inspection -----------------
elif page == "2. Live Hybrid Inspection":
    st.title("⚡ Live Flow Classification & Gating")

    available_modes = list(bundle.get("gate_configs", {}).keys()) + ["stacked"]
    selected_mode = st.sidebar.selectbox("Gating Mode", available_modes, index=0)

    st.write("Evaluate individual network flows using the fused dual-engine model:")

    test_file = find_file([
        ROOT / "data" / "processed" / "log1p" / "X_test_supervised.parquet",
        ROOT / "data" / "processed" / "X_test_supervised.parquet"
    ])

    if test_file.exists():
        df_test = pd.read_parquet(test_file)
        row_id = st.slider("Select Sample Flow Index", 0, len(df_test) - 1, 10)
        flow_input = df_test.iloc[[row_id]]
    else:
        st.warning("Processed test parquet not found. Using empty dummy record.")
        flow_input = pd.DataFrame(np.zeros((1, len(bundle["feature_names"]))), columns=bundle["feature_names"])

    st.dataframe(flow_input.iloc[:, :12], use_container_width=True)

    if st.button("Classify Flow", type="primary"):
        out = predict(bundle, flow_input, mode=selected_mode)
        decision = out["decision"][0]
        ae_err = float(out["ae_error"][0])
        p_benign = float(out["p_benign"][0])

        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", str(decision))
        col2.metric("P(Benign)", f"{p_benign * 100:.2f}%")
        col3.metric("AE Reconstruction MSE", f"{ae_err:.6f}")

        if decision == "BENIGN":
            st.success("✅ **BENIGN:** Both engines cleared the flow as normal traffic.")
        elif decision == "ANOMALY":
            st.warning("⚠️ **ANOMALY:** Random Forest marked benign, but Autoencoder reconstruction error spiked (potential zero-day).")
        else:
            st.error(f"🚨 **KNOWN ATTACK:** Classified as **{decision}**.")

# ----------------- 3. SHAP Explainability -----------------
elif page == "3. SHAP Explainability":
    st.title("🔍 Model Explainability & Interpretability (SHAP)")
    st.markdown("Interpret global feature impact and inspect local prediction drivers via Shapley attribution.")

    tab_global, tab_local = st.tabs(["Global Feature Importance", "Local Flow Waterfall"])

    with tab_global:
        st.subheader("Global Feature Attribution (85 Fused Features)")
        summary_img = ROOT / "results" / "shap" / "shap_summary_plot.png"
        if summary_img.exists():
            st.image(str(summary_img), caption="TreeSHAP Summary Distribution", use_container_width=True)
        else:
            st.info("Global summary plot not found in results/shap/")

        top_csv = ROOT / "results" / "shap" / "top_shap_features.csv"
        if top_csv.exists():
            st.write("Top Ranking Features:")
            st.dataframe(pd.read_csv(top_csv), use_container_width=True)

    with tab_local:
        st.subheader("Local Instance Explanation (Waterfall)")
        explainer, eval_df = load_shap_artifacts()

        if explainer is None or eval_df is None:
            st.warning("SHAP evaluation artifacts missing. Ensure generate_shap.py has finished.")
        else:
            instance_idx = st.number_input("Select Background Sample Index", min_value=0, max_value=len(eval_df) - 1, value=0)
            target_row = eval_df.iloc[[instance_idx]]

            if st.button("Generate Waterfall Plot"):
                with st.spinner("Rendering waterfall plot..."):
                    shap_obj = explainer(target_row, check_additivity=False)
                    probs = bundle["rf"].predict_proba(target_row.to_numpy())[0]
                    target_class_idx = int(np.argmax(probs))
                    class_name = bundle["class_names"][target_class_idx]

                    st.markdown(f"**Explained Prediction:** `{class_name}` (Confidence: **{probs[target_class_idx] * 100:.2f}%**)")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    if len(shap_obj.shape) == 3:
                        shap.plots.waterfall(shap_obj[0, :, target_class_idx], max_display=12, show=False)
                    else:
                        shap.plots.waterfall(shap_obj[0], max_display=12, show=False)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()