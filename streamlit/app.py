from pathlib import Path
import json
import importlib.util
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- Project paths ----------------------------------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
LOG1P_DATA = DATA / "log1p"

st.set_page_config(
    page_title="Hybrid NIDS | Network Intrusion Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Hybrid Network Intrusion Detection System")
st.caption(
    "CICIDS2017 • Supervised Classification • Anomaly Detection • Hybrid AE + RF"
)

# Helpers
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


def read_parquet(path):
    if not path or not Path(path).is_file():
        return None
    return pd.read_parquet(path)


def read_labels(path):
    frame = read_parquet(path)
    if frame is None or frame.empty:
        return None
    return frame.iloc[:, 0].tolist()


def label_mapping():
    for path in (DATA / "label_mapping.json", LOG1P_DATA / "label_mapping.json"):
        mapping = read_json(path)
        if isinstance(mapping, dict) and mapping:
            return mapping
    return {}


LABEL_MAP = label_mapping()
ID_TO_LABEL = {}
for label, value in LABEL_MAP.items():
    try:
        ID_TO_LABEL[int(value)] = str(label)
    except (TypeError, ValueError):
        pass


def display_class(value):
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    ):
        return ID_TO_LABEL.get(int(value), str(int(value)))
    text = str(value)
    try:
        numeric = int(float(text))
        return ID_TO_LABEL.get(numeric, text)
    except (TypeError, ValueError):
        return text


def is_benign(value):
    return display_class(value).strip().upper() in {"BENIGN", "NORMAL"}


def anomaly_truth(value):
    text = str(value).strip().upper()
    if text in {"BENIGN", "NORMAL", "0", "0.0", "FALSE"}:
        return "BENIGN"
    if text in {"ANOMALY", "ATTACK", "MALICIOUS", "1", "1.0", "TRUE"}:
        return "ANOMALY"
    return text


def align_features(frame, feature_names, model_name):
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{model_name}: dataset is missing required features: {missing[:10]}"
        )
    return frame.loc[:, feature_names].copy()


def load_feature_names(path):
    names = read_json(path)
    if not isinstance(names, list) or not names:
        raise ValueError(f"Feature-name file is missing or invalid: {path}")
    return names


def load_joblib(path):
    return joblib.load(str(path))


def load_keras(path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)


def load_hybrid_predict_module(folder):
    module_path = folder / "hybrid_predict.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"hybrid_predict.py not found beside the hybrid bundle: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("nids_hybrid_predict", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_indices(total, limit=20000):
    """Deterministic, evenly spaced subset of held-out rows for demo curation."""
    count = min(total, limit)
    if count <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, total - 1, num=count, dtype=int))


def make_supervised_examples(model, X, y, max_examples=3):
    """
    Curate actual held-out rows for:
    1) correct BENIGN prediction,
    2) correct attack-class prediction,
    3) a genuine misclassification, if found.
    """
    if y is None or len(X) != len(y):
        raise ValueError(
            "Supervised test features and labels are missing or have different lengths."
        )

    indices = candidate_indices(len(X))
    candidates = X.iloc[indices]
    predicted_values = model.predict(candidates)

    correct_benign = None
    correct_attack = None
    misclassified = None

    for idx, predicted_value in zip(indices, predicted_values):
        actual = display_class(y[idx])
        predicted = display_class(predicted_value)
        is_correct = actual.strip().upper() == predicted.strip().upper()

        record = {
            "row_index": int(idx),
            "actual": actual,
            "predicted_at_curation": predicted,
            "features": X.iloc[int(idx)].copy(),
        }

        if not is_correct and misclassified is None:
            record["purpose"] = "Misclassification example"
            record["name"] = f"⚠ Misclassified — actual {actual} (test row {idx})"
            misclassified = record
        elif is_correct and is_benign(actual) and correct_benign is None:
            record["purpose"] = "Correct BENIGN example"
            record["name"] = f"✓ Correct BENIGN (test row {idx})"
            correct_benign = record
        elif is_correct and not is_benign(actual) and correct_attack is None:
            record["purpose"] = "Correct attack-class example"
            record["name"] = f"✓ Correct {actual} (test row {idx})"
            correct_attack = record

        if correct_benign and correct_attack and misclassified:
            break

    examples = [
        item for item in (correct_benign, correct_attack, misclassified)
        if item is not None
    ]

    # If the search subset has no example of one category, show available real rows.
    if len(examples) < max_examples:
        used = {item["row_index"] for item in examples}
        for idx in indices:
            idx = int(idx)
            if idx in used:
                continue
            actual = display_class(y[idx])
            predicted = display_class(
                model.predict(X.iloc[[idx]])[0]
            )
            is_correct = actual.strip().upper() == predicted.strip().upper()
            examples.append({
                "name": f"{'✓' if is_correct else '⚠'} {actual} — test row {idx}",
                "row_index": idx,
                "actual": actual,
                "predicted_at_curation": predicted,
                "purpose": "Additional real test example",
                "features": X.iloc[idx].copy(),
            })
            used.add(idx)
            if len(examples) >= max_examples:
                break

    return examples[:max_examples]


def make_anomaly_demo_examples(X, y, predict_labels, max_examples=3):
    """
    Curate real held-out rows for:
    1) correctly identified BENIGN,
    2) correctly identified ANOMALY,
    3) a genuine binary false positive/false negative, if found.
    predict_labels accepts a feature DataFrame and returns BENIGN/ANOMALY labels.
    """
    if y is None or len(X) != len(y):
        raise ValueError(
            "Anomaly test features and binary labels are missing or have different lengths."
        )

    indices = candidate_indices(len(X))
    candidate_X = X.iloc[indices]
    predictions = predict_labels(candidate_X)

    correct_benign = None
    correct_attack = None
    misclassified = None

    for idx, predicted in zip(indices, predictions):
        actual = anomaly_truth(y[int(idx)])
        predicted = str(predicted).upper()
        is_correct = actual == predicted

        record = {
            "row_index": int(idx),
            "actual": actual,
            "predicted_at_curation": predicted,
            "features": X.iloc[int(idx)].copy(),
        }

        if not is_correct and misclassified is None:
            record["purpose"] = "Misclassification example"
            record["name"] = (
                f"⚠ Misclassified — actual {actual} (test row {idx})"
            )
            misclassified = record
        elif is_correct and actual == "BENIGN" and correct_benign is None:
            record["purpose"] = "Correct BENIGN example"
            record["name"] = f"✓ Correct BENIGN (test row {idx})"
            correct_benign = record
        elif is_correct and actual == "ANOMALY" and correct_attack is None:
            record["purpose"] = "Correct anomaly example"
            record["name"] = f"✓ Correct ANOMALY (test row {idx})"
            correct_attack = record

        if correct_benign and correct_attack and misclassified:
            break

    examples = [
        item for item in (correct_benign, correct_attack, misclassified)
        if item is not None
    ]

    if len(examples) < max_examples:
        used = {item["row_index"] for item in examples}
        for idx, predicted in zip(indices, predictions):
            idx = int(idx)
            if idx in used:
                continue
            actual = anomaly_truth(y[idx])
            predicted = str(predicted).upper()
            is_correct = actual == predicted
            examples.append({
                "name": f"{'✓' if is_correct else '⚠'} {actual} — test row {idx}",
                "row_index": idx,
                "actual": actual,
                "predicted_at_curation": predicted,
                "purpose": "Additional real test example",
                "features": X.iloc[idx].copy(),
            })
            used.add(idx)
            if len(examples) >= max_examples:
                break

    return examples[:max_examples]


def render_example_selector(model_key, examples):
    """Choose a real example and return its prefilled feature row."""
    if not examples:
        st.error("No usable test examples were found.")
        st.stop()

    names = [example["name"] for example in examples]
    selected_name = st.selectbox(
        "Choose an example network flow",
        names,
        key=f"{model_key}_example_choice",
    )
    selected = examples[names.index(selected_name)]

    st.markdown(f"**Dataset's actual label:** `{selected['actual']}`")
    purpose = selected.get("purpose", "Real held-out test example")
    if "Misclassification" in purpose:
        st.warning(
            "This real test flow was selected because the model misclassified it "
            "during demo-example selection. Press Predict to reproduce the result."
        )
    else:
        st.caption(f"Demo category: {purpose}")
    if not any("Misclassification" in item.get("purpose", "") for item in examples):
        st.info(
            "No misclassified example was found in the candidate rows searched for "
            "this model. The dashboard will not invent an incorrect prediction."
        )
    st.caption(
        "The selected test flow is loaded into the fields below. You can inspect or "
        "edit its feature values before pressing Predict. If you edit the values, "
        "the original dataset label may no longer describe the modified flow."
    )

    with st.expander("Inspect the exact selected network-flow values", expanded=False):
        values_df = selected["features"].rename("Value").to_frame()
        st.dataframe(values_df, use_container_width=True)

    return selected


def render_predict_form(model_key, feature_names, selected_example):
    """Render editable fields prefilled with the chosen example and a Predict button."""
    sample = selected_example["features"].reindex(feature_names)
    form_key = f"{model_key}_predict_form_{selected_example.get('row_index', selected_example['name'])}"

    st.markdown("#### Edit flow values (optional)")
    st.caption(
        f"{len(feature_names)} features • Values are prefilled from the selected "
        "real test flow. Change any values if you want to test a modified flow."
    )

    values = {}
    with st.form(form_key):
        left, right = st.columns(2)
        for i, name in enumerate(feature_names):
            raw = sample[name]
            try:
                default = float(raw)
                if not np.isfinite(default):
                    default = 0.0
            except (TypeError, ValueError):
                default = 0.0

            container = left if i % 2 == 0 else right
            with container:
                values[name] = st.number_input(
                    name,
                    value=default,
                    step=0.01,
                    format="%.8f",
                    key=f"{form_key}_{i}",
                )

        submitted = st.form_submit_button(
            "Predict this network flow",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        return pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names,
            dtype=np.float32,
        )
    return None


def show_classification_result(label):
    if str(label).strip().upper() in {"BENIGN", "NORMAL"}:
        st.success(f"Prediction: {label}")
    else:
        st.error(f"Prediction: {label}")


def show_anomaly_result(prediction):
    if prediction == "BENIGN":
        st.success("Prediction: BENIGN-LIKE TRAFFIC")
    elif prediction == "ANOMALY":
        st.error("Prediction: ANOMALY DETECTED")
    else:
        st.warning(f"Prediction: {prediction}")


# ----------------------------- Model paths ------------------------------------
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

if_path = first_existing([
    MODELS / "isolation_forest_model" / "isolation_forest" / "isolation_forest_log1p.joblib",
    MODELS / "isolation_forest_model" / "isolation_forest" / "isolation_forest_log1p_ae_basis.joblib",
    MODELS / "isolation_forest" / "isolation_forest_log1p.joblib",
    MODELS / "isolation_forest_log1p.joblib",
])

ae_path = first_existing([
    MODELS / "autoencoder_model" / "autoencoder" / "best_autoencoder_log1p.keras",
    MODELS / "autoencoder" / "best_autoencoder_log1p.keras",
])

hybrid_bundle_path = first_existing([
    MODELS / "hybrid_ae_rf_model" / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
    MODELS / "hybrid_ae_rf" / "hybrid_nids_bundle.joblib",
])

# Dataset paths
X_supervised_path = DATA / "X_test_supervised.parquet"
y_supervised_path = DATA / "y_test_supervised.parquet"
X_anomaly_path = LOG1P_DATA / "X_test_anomaly.parquet"
y_anomaly_path = LOG1P_DATA / "y_test_anomaly_binary.parquet"
X_heldout_path = LOG1P_DATA / "X_heldout_portscan.parquet"
y_heldout_path = LOG1P_DATA / "y_heldout_portscan.parquet"

# Sidebar navigation
pages = [
    "Overview",
    "Random Forest",
    "XGBoost",
    "Isolation Forest",
    "Deep Autoencoder",
    "Hybrid AE + RF",
]
selected = st.sidebar.radio("Dashboard navigation", pages)
st.sidebar.divider()
st.sidebar.caption("Model availability")
for name, path in [
    ("Random Forest", rf_path),
    ("XGBoost", xgb_path),
    ("Isolation Forest", if_path),
    ("Autoencoder", ae_path),
    ("Hybrid AE + RF bundle", hybrid_bundle_path),
]:
    st.sidebar.write(("🟢 " if path else "🔴 ") + name)

# ----------------------------- Overview ---------------------------------------
if selected == "Overview":
    st.subheader("Interactive network-flow demonstration")
    st.write(
        "Choose a model, select one of the provided real test flows, inspect or edit "
        "its feature values, and press Predict to see the model's output."
    )
    cards = st.columns(5)
    for col, (name, path) in zip(cards, [
        ("Random Forest", rf_path),
        ("XGBoost", xgb_path),
        ("Isolation Forest", if_path),
        ("Deep Autoencoder", ae_path),
        ("Hybrid AE + RF", hybrid_bundle_path),
    ]):
        col.markdown(f"**{name}**")
        if path:
            col.success("Available")
        else:
            col.error("Not found")

    st.markdown("### What each model tells us")
    st.markdown(
        """
        - **Random Forest / XGBoost:** predict a known traffic class (for example, BENIGN or DDoS).
        - **Isolation Forest / Deep Autoencoder:** predict BENIGN-like or ANOMALY; they do not identify the exact attack family.
        - **Hybrid AE + RF:** combines the supervised classification result with the Autoencoder anomaly gate.
        """
    )
    st.info(
        "Each model uses real held-out test rows and its own preprocessing representation. "
        "The example selector deliberately includes correct predictions and a genuine "
        "misclassification when one is found. These are curated teaching examples, not "
        "a substitute for reporting full-test-set performance."
    )

# ----------------------------- Random Forest ----------------------------------
elif selected == "Random Forest":
    st.subheader("Random Forest - Known-attack classification")
    if not rf_path:
        st.error("Random Forest model file not found.")
        st.stop()

    try:
        model = load_joblib(rf_path)
        X = read_parquet(X_supervised_path)
        y = read_labels(y_supervised_path)
        if X is None or y is None:
            raise FileNotFoundError(
                "X_test_supervised.parquet or y_test_supervised.parquet is missing."
            )

        feature_names = getattr(model, "feature_names_in_", None)
        feature_names = list(feature_names) if feature_names is not None else list(X.columns)
        X = align_features(X, feature_names, "Random Forest")
        examples = make_supervised_examples(model, X, y)
        selected_example = render_example_selector("rf", examples)
        row = render_predict_form("rf", feature_names, selected_example)

        if row is not None:
            predicted = display_class(model.predict(row)[0])
            st.markdown("### Model output")
            show_classification_result(predicted)
            st.write(f"Dataset's actual label: **{selected_example['actual']}**")
            st.write(
                "**Comparison:** "
                + ("Correct prediction" if predicted == selected_example["actual"] else "Prediction differs from dataset label")
            )
    except Exception as exc:
        st.error("Could not run the Random Forest example.")
        st.exception(exc)

# ----------------------------- XGBoost ----------------------------------------
elif selected == "XGBoost":
    st.subheader("XGBoost - Known-attack classification")
    if not xgb_path:
        st.error("XGBoost model file not found.")
        st.stop()

    try:
        model = load_joblib(xgb_path)
        X = read_parquet(X_supervised_path)
        y = read_labels(y_supervised_path)
        if X is None or y is None:
            raise FileNotFoundError(
                "X_test_supervised.parquet or y_test_supervised.parquet is missing."
            )

        feature_names = getattr(model, "feature_names_in_", None)
        feature_names = list(feature_names) if feature_names is not None else list(X.columns)
        X = align_features(X, feature_names, "XGBoost")
        examples = make_supervised_examples(model, X, y)
        selected_example = render_example_selector("xgb", examples)
        row = render_predict_form("xgb", feature_names, selected_example)

        if row is not None:
            predicted = display_class(model.predict(row)[0])
            st.markdown("### Model output")
            show_classification_result(predicted)
            st.write(f"Dataset's actual label: **{selected_example['actual']}**")
            st.write(
                "**Comparison:** "
                + ("Correct prediction" if predicted == selected_example["actual"] else "Prediction differs from dataset label")
            )
    except Exception as exc:
        st.error("Could not run the XGBoost example.")
        st.exception(exc)

# ----------------------------- Isolation Forest -------------------------------
elif selected == "Isolation Forest":
    st.subheader("Isolation Forest - Anomaly detection")
    if not if_path:
        st.error("Isolation Forest model file not found.")
        st.stop()

    try:
        model = load_joblib(if_path)
        X = read_parquet(X_anomaly_path)
        y = read_labels(y_anomaly_path)
        if X is None:
            raise FileNotFoundError("X_test_anomaly.parquet is missing.")

        feature_names = getattr(model, "feature_names_in_", None)
        feature_names = list(feature_names) if feature_names is not None else list(X.columns)
        X = align_features(X, feature_names, "Isolation Forest")

        heldout_X = read_parquet(X_heldout_path)
        heldout_y = read_labels(y_heldout_path)
        if heldout_X is not None:
            heldout_X = align_features(heldout_X, feature_names, "Held-out PortScan")

        result_json = read_json(RESULTS / "isolation_forest" / "log1p" / "results.json")
        thresholds = result_json.get("thresholds", {})
        threshold_key = next(
            (key for key in ("best_f1", "p95", "p99") if key in thresholds),
            None,
        )
        threshold = float(thresholds[threshold_key]) if threshold_key else None

        if threshold is None:
            raise ValueError(
                "No calibrated Isolation Forest threshold was found; "
                "cannot curate correct/incorrect binary demo examples."
            )

        def if_predict_labels(batch):
            scores = -model.score_samples(batch)
            return np.where(scores >= threshold, "ANOMALY", "BENIGN")

        examples = make_anomaly_demo_examples(X, y, if_predict_labels)
        selected_example = render_example_selector("iforest", examples)
        row = render_predict_form("iforest", feature_names, selected_example)

        if row is not None:
            score = float(-model.score_samples(row)[0])
            st.markdown("### Model output")
            st.metric("Anomaly score (higher = more unusual)", f"{score:.8f}")
            if threshold is not None:
                prediction = "ANOMALY" if score >= threshold else "BENIGN"
                show_anomaly_result(prediction)
                st.caption(f"Decision threshold ({threshold_key}): {threshold:.8f}")
                st.write(f"Dataset's actual label: **{selected_example['actual']}**")
                expected = "BENIGN" if selected_example["actual"] == "BENIGN" else "ANOMALY"
                st.write(
                    "**Comparison:** "
                    + ("Correct anomaly decision" if prediction == expected else "Prediction differs from dataset label")
                )
            else:
                st.warning(
                    "No calibrated threshold was found in the Isolation Forest results. "
                    "The anomaly score is shown, but a BENIGN/ANOMALY decision is not assigned."
                )
    except Exception as exc:
        st.error("Could not run the Isolation Forest example.")
        st.exception(exc)

# ----------------------------- Autoencoder ------------------------------------
elif selected == "Deep Autoencoder":
    st.subheader("Deep Autoencoder - Reconstruction based anomaly detection")
    if not ae_path:
        st.error("Autoencoder .keras model not found.")
        st.stop()

    try:
        metadata_path = ae_path.parent / "autoencoder_metadata_log1p.json"
        metadata = read_json(metadata_path)
        if not metadata:
            raise FileNotFoundError(f"Autoencoder metadata not found: {metadata_path}")

        model = load_keras(ae_path)
        feature_names = load_feature_names(LOG1P_DATA / "feature_names.json")
        model_dim = int(model.input_shape[-1])
        metadata_dim = int(metadata.get("input_dim", -1))
        if model_dim != metadata_dim or len(feature_names) != model_dim:
            raise ValueError(
                f"Autoencoder dimensions mismatch: model={model_dim}, "
                f"metadata={metadata_dim}, features={len(feature_names)}."
            )

        X = read_parquet(X_anomaly_path)
        y = read_labels(y_anomaly_path)
        if X is None:
            raise FileNotFoundError("X_test_anomaly.parquet is missing.")
        X = align_features(X, feature_names, "Autoencoder")

        heldout_X = read_parquet(X_heldout_path)
        heldout_y = read_labels(y_heldout_path)
        if heldout_X is not None:
            heldout_X = align_features(heldout_X, feature_names, "Held-out PortScan")

        threshold = metadata.get("calibrated_threshold")
        threshold = float(threshold) if threshold is not None else None
        if threshold is None:
            raise ValueError(
                "Autoencoder metadata has no calibrated_threshold; "
                "cannot curate correct/incorrect binary demo examples."
            )

        def ae_predict_labels(batch):
            values = batch.to_numpy(dtype=np.float32)
            recon = model.predict(values, verbose=0, batch_size=1024)
            errors = np.mean(np.square(values - recon), axis=1)
            return np.where(errors >= threshold, "ANOMALY", "BENIGN")

        examples = make_anomaly_demo_examples(X, y, ae_predict_labels)
        selected_example = render_example_selector("ae", examples)
        row = render_predict_form("ae", feature_names, selected_example)

        if row is not None:
            values = row.to_numpy(dtype=np.float32)
            reconstruction = model.predict(values, verbose=0)
            error = float(np.mean(np.square(values - reconstruction), axis=1)[0])

            st.markdown("### Model output")
            st.metric("Reconstruction error", f"{error:.8f}")
            if threshold is not None:
                prediction = "ANOMALY" if error >= threshold else "BENIGN"
                show_anomaly_result(prediction)
                st.caption(f"Calibrated threshold: {threshold:.8f}")
                st.write(f"Dataset's actual label: **{selected_example['actual']}**")
                expected = "BENIGN" if selected_example["actual"] == "BENIGN" else "ANOMALY"
                st.write(
                    "**Comparison:** "
                    + ("Correct anomaly decision" if prediction == expected else "Prediction differs from dataset label")
                )
            else:
                st.warning(
                    "The metadata does not contain calibrated_threshold. "
                    "Reconstruction error is shown without assigning a class."
                )
    except Exception as exc:
        st.error("Could not run the Autoencoder example.")
        st.exception(exc)

# ----------------------------- Hybrid AE + RF ---------------------------------
elif selected == "Hybrid AE + RF":
    st.subheader("Hybrid NIDS - Autoencoder + Random Forest")
    if not hybrid_bundle_path:
        st.error("hybrid_nids_bundle.joblib not found.")
        st.stop()

    try:
        bundle = joblib.load(str(hybrid_bundle_path))
        hybrid_predict = load_hybrid_predict_module(hybrid_bundle_path.parent)
        feature_names = list(bundle["feature_names"])

        X = read_parquet(X_anomaly_path)
        y = read_labels(y_anomaly_path)
        if X is None:
            raise FileNotFoundError("X_test_anomaly.parquet is missing.")
        X = align_features(X, feature_names, "Hybrid AE + RF")

        heldout_X = read_parquet(X_heldout_path)
        heldout_y = read_labels(y_heldout_path)
        if heldout_X is not None:
            heldout_X = align_features(heldout_X, feature_names, "Held-out PortScan")

        mode_options = list(bundle.get("gate_configs", {}).keys())
        default_mode = bundle.get("default_mode")
        if not mode_options:
            raise ValueError("The hybrid bundle contains no gate_configs.")
        mode_index = mode_options.index(default_mode) if default_mode in mode_options else 0
        mode = st.selectbox(
            "Hybrid decision strategy",
            mode_options,
            index=mode_index,
            key="hybrid_gate_mode",
        )

        def hybrid_predict_labels(batch):
            output = hybrid_predict.predict(bundle, batch, mode=mode)
            decisions = [str(value).upper() for value in output["decision"]]
            return np.array([
                "BENIGN" if value == "BENIGN" else "ANOMALY"
                for value in decisions
            ])

        examples = make_anomaly_demo_examples(X, y, hybrid_predict_labels)
        selected_example = render_example_selector("hybrid", examples)
        row = render_predict_form("hybrid", feature_names, selected_example)

        if row is not None:
            output = hybrid_predict.predict(bundle, row, mode=mode)
            decision = str(output["decision"][0])
            p_benign = float(output["p_benign"][0])
            ae_error = float(output["ae_error"][0])
            flagged = bool(output["flagged"][0])

            st.markdown("### Model output")
            show_classification_result(decision)
            col1, col2 = st.columns(2)
            col1.metric("RF probability of BENIGN", f"{p_benign:.2%}")
            col2.metric("AE reconstruction error", f"{ae_error:.8f}")
            st.write(f"Dataset's actual label: **{selected_example['actual']}**")
            st.write(f"Hybrid alert: **{'YES' if flagged else 'NO'}**")
    except Exception as exc:
        st.error("Could not run the Hybrid AE + RF example.")
        st.exception(exc)

st.sidebar.divider()

