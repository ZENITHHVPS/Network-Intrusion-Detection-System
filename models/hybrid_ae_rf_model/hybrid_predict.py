"""
hybrid_predict.py - use the bundled hybrid NIDS WITHOUT TensorFlow.

Needs only: numpy, scikit-learn (same version as the bundle was built with), joblib.

    import joblib
    from hybrid_predict import predict
    bundle = joblib.load("hybrid_nids_bundle.joblib")
    out = predict(bundle, X)                       # X: log1p + MinMax-scaled flows, 68 features
    out["decision"]                                # 'BENIGN', an attack name, or 'ANOMALY'
"""
import numpy as np


def ae_forward(layers, X, latent_name):
    """Autoencoder forward pass in numpy. Returns (latent code, reconstruction)."""
    h = np.asarray(X, dtype=np.float32)
    latent = None
    for L in layers:
        if L["type"] == "dense":
            h = h @ L["W"] + L["b"]
            if L["act"] == "relu":
                h = np.maximum(h, 0.0)
            elif L["act"] == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -50.0, 50.0)))
            elif L["act"] != "linear":
                raise ValueError(f"Unsupported activation: {L['act']}")
        else:  # batch normalisation, inference mode
            h = (h - L["mean"]) / np.sqrt(L["var"] + L["eps"]) * L["gamma"] + L["beta"]
        if L["name"] == latent_name:
            latent = h
    return latent, h


def build_fused(bundle, X):
    """68 input features -> (85 fused features, reconstruction error)."""
    if hasattr(X, "columns"):                              # DataFrame: enforce the training column order
        X = X[bundle["feature_names"]].to_numpy(dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(bundle["feature_names"]):
        raise ValueError(f"Expected {len(bundle['feature_names'])} features, got shape {X.shape}")
    latent, recon = ae_forward(bundle["ae_layers"], X, bundle["latent_layer"])
    err = np.mean(np.square(X - recon), axis=1)
    fused = np.concatenate([X, latent, err.reshape(-1, 1)], axis=1).astype(np.float32)
    return fused, err


def predict(bundle, X, mode=None):
    """
    mode: 'stacked' (RF only) or a gate configuration name stored in the bundle,
          e.g. 'gate_p0.99_fpr0.5pct' (default) or 'gate_p0.99_fpr1.0pct'.
    Returns a dict of arrays: decision, class_id, flagged, p_benign, ae_error.
    """
    mode = mode or bundle["default_mode"]
    fused, err = build_fused(bundle, X)
    proba = bundle["rf"].predict_proba(fused)
    classes = np.asarray(bundle["rf"].classes_)
    class_id = classes[proba.argmax(axis=1)]
    p_benign = proba[:, 0]
    rf_flag = class_id != 0

    if mode == "stacked":
        flagged = rf_flag
    else:
        cfg = bundle["gate_configs"][mode]
        mask = np.ones(len(err), dtype=bool) if cfg["p_cut"] is None else p_benign < cfg["p_cut"]
        flagged = rf_flag | (mask & (err > cfg["tau"]))

    names = np.asarray(bundle["class_names"], dtype=object)
    decision = names[class_id].astype(object)
    decision[flagged & ~rf_flag] = "ANOMALY"               # RF said benign, gate overruled it
    return {"decision": decision, "class_id": class_id, "flagged": flagged,
            "p_benign": p_benign, "ae_error": err}
