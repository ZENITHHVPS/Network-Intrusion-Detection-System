#!/usr/bin/env python
"""
Pack the hybrid NIDS into ONE file: models/hybrid_ae_rf/hybrid_nids_bundle.joblib

Contents: Random Forest, Autoencoder weights (plain numpy, no TensorFlow needed to use it),
gate settings (p_cut, tau), class names, feature order, library versions.

Place in the PROJECT ROOT next to train_hybrid_ae_rf.py, confidence_gate.py and hybrid_predict.py.
Run once (needs TensorFlow, because it reads the .keras file):
    python build_bundle.py
It then checks the bundle against your existing pipeline on 5,000 validation flows.
"""
import json
import platform

from anyio import Path
import joblib
import numpy as np
import sklearn
import tensorflow as tf

from models.hybrid_ae_rf_model.confidence_gate import apply_rule
from models.hybrid_ae_rf_model.hybrid_ae_rf.hybrid_predict import build_fused, predict
from models.hybrid_ae_rf_model.train_hybrid_ae_rf import (AE_PATH, MOD, RES, fuse, load_feature_model, load_names,
                                load_split, log)

OUT = MOD / "hybrid_nids_bundle.joblib"
LATENT_LAYER = "latent_bottleneck"
DEFAULT_MODE = "gate_p0.99_fpr0.5pct"          # balanced operating point from the report


def extract_ae(ae):
    """Keras layers -> list of plain numpy dicts (Dense and BatchNormalization only)."""
    layers = []
    for l in ae.layers:
        kind = l.__class__.__name__
        if kind in ("InputLayer", "Dropout"):
            continue                                          # no effect at inference time
        if kind == "Dense":
            W, b = l.get_weights()
            layers.append({"type": "dense", "name": l.name, "W": W.astype(np.float32),
                           "b": b.astype(np.float32), "act": l.get_config()["activation"]})
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = l.get_weights()
            layers.append({"type": "bn", "name": l.name, "gamma": gamma, "beta": beta,
                           "mean": mean, "var": var, "eps": float(l.epsilon)})
        else:
            raise ValueError(f"Layer type {kind} ({l.name}) is not supported by the numpy forward pass")
    return layers


def main():
    rf = joblib.load(MOD / "hybrid_rf.joblib")
    ae = tf.keras.models.load_model(AE_PATH, compile=False)
    X_va, y_va = load_split("val")
    gate_cfgs = {c["name"]: {"p_cut": c["p_cut"], "tau": c["tau"], "budget": c["budget"]}
                 for c in json.loads((RES / "gate_config.json").read_text(encoding="utf-8"))}
    if DEFAULT_MODE not in gate_cfgs:
        raise SystemExit(f"{DEFAULT_MODE} not in gate_config.json: {list(gate_cfgs)}")

    bundle = {
        "format_version": 1,
        "rf": rf,
        "ae_layers": extract_ae(ae),
        "latent_layer": LATENT_LAYER,
        "gate_configs": gate_cfgs,
        "default_mode": DEFAULT_MODE,
        "class_names": load_names(),
        "feature_names": list(X_va.columns),
        "input_note": "Input must be log1p-transformed and MinMax-scaled, in feature_names order.",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__, "tensorflow": tf.__version__},
    }

    # ---- verification against the existing (Keras) pipeline ----
    X = X_va.iloc[:5000]
    fused_k, err_k = fuse(load_feature_model(), X)
    fused_b, err_b = build_fused(bundle, X)
    diff = float(np.max(np.abs(fused_k - fused_b)))
    log(f"Max |difference| in the 85 fused features, Keras vs numpy: {diff:.2e}")
    assert diff < 1e-3, "Numpy autoencoder does not match Keras; do not use this bundle"

    proba = rf.predict_proba(fused_k)
    rf_flag = rf.classes_[proba.argmax(axis=1)] != 0
    out = predict(bundle, X, "stacked")
    assert (out["class_id"] == rf.classes_[proba.argmax(axis=1)]).mean() > 0.999, "Class predictions differ"
    for name, cfg in gate_cfgs.items():
        expected = apply_rule(proba[:, 0], err_k, rf_flag, cfg["p_cut"], cfg["tau"])
        got = predict(bundle, X, name)["flagged"]
        agree = float((expected == got).mean())
        log(f"{name}: flags agree with the existing pipeline on {agree:.4%} of flows")
        assert agree > 0.999, f"Gate {name} disagrees with the existing pipeline"

    from pathlib import Path
    bundle["predict_source"] = Path("hybrid_predict.py").read_text(encoding="utf-8")
    joblib.dump(bundle, OUT, compress=3)
    
    log(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB). Versions: {bundle['versions']}")


if __name__ == "__main__":
    main()
