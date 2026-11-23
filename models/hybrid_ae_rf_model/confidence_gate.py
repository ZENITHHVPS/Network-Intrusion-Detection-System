#!/usr/bin/env python
"""
Confidence-gated anomaly rule for the hybrid AE + Random Forest.

Plain rule      : flag = RF says attack  OR  AE error > tau
Confidence gate : flag = RF says attack  OR  (RF is unsure AND AE error > tau)
                  "RF is unsure"  <=>  P(BENIGN) from rf.predict_proba  <  p_cut

Nothing is retrained. It reuses models/hybrid_ae_rf/hybrid_rf.joblib and the frozen AE.
Place this file in the PROJECT ROOT next to train_hybrid_ae_rf.py.

  python confidence_gate.py --stage val     # choose tau on VALIDATION, print the table
  python confidence_gate.py --stage test    # apply the saved configs, ONCE

For each false-alarm budget, tau is the cutoff that makes the benign validation
false-positive rate equal that budget. So every row at the same budget has the same
false-alarm cost, and the comparison between gated and plain rules is fair.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from train_hybrid_ae_rf import (DATA, MOD, RES, binary_metrics, fuse, load_feature_model,
                                load_split, log, save_json)

#Fixed in advance. Do NOT add or change values after looking at test / PortScan results.
BUDGETS = [0.005, 0.01]            #total benign false-positive rate allowed: 0.5% and 1.0%
P_CUTS = [0.5, 0.9, 0.99, None]    #None = no gate (plain rule), the fair baseline

CONFIG_FILE = RES / "gate_config.json"


def score(rf, fm, X):
    """Returns (P(BENIGN), AE error, RF attack flag). Column 0 of predict_proba must be BENIGN."""
    Xf, err = fuse(fm, X)
    proba = rf.predict_proba(Xf)
    assert int(rf.classes_[0]) == 0, "Column 0 of predict_proba is not BENIGN"
    rf_flag = rf.classes_[proba.argmax(axis=1)] != 0
    return proba[:, 0], err, rf_flag


def gate_mask(p_benign, p_cut):
    return np.ones_like(p_benign, dtype=bool) if p_cut is None else p_benign < p_cut


def fit_tau(p_benign, err, rf_flag, y, p_cut, budget):
    """Largest-coverage tau whose benign false-positive rate stays within `budget` (validation)."""
    benign = y == 0
    already = int((rf_flag & benign).sum())                  #false alarms the RF makes by itself
    allowed = int(np.floor(budget * benign.sum())) - already
    if allowed < 0:
        return None                                          #budget below the RF's own FPR
    eligible = np.sort(err[benign & ~rf_flag & gate_mask(p_benign, p_cut)])[::-1]
    if allowed >= len(eligible):
        return -1.0                                          #gate never adds more than the budget
    return float(eligible[allowed])


def apply_rule(p_benign, err, rf_flag, p_cut, tau):
    return rf_flag | (gate_mask(p_benign, p_cut) & (err > tau))


def name_of(p_cut, budget):
    gate = "nogate" if p_cut is None else f"gate_p{p_cut}"
    return f"{gate}_fpr{budget * 100:.1f}pct"


#VALIDATION: choose tau, print table (test set untouched)
def stage_val():
    rf = joblib.load(MOD / "hybrid_rf.joblib")
    fm = load_feature_model()
    X_va, y_va = load_split("val")
    yv = y_va.to_numpy()
    p_benign, err, rf_flag = score(rf, fm, X_va)

    attack, benign = yv != 0, yv == 0
    log(f"RF alone on val: detection {rf_flag[attack].mean():.4%}, FPR {rf_flag[benign].mean():.4%}")
    for pc in [c for c in P_CUTS if c is not None]:
        log(f"p_benign < {pc}: {np.mean(p_benign[benign] < pc):.2%} of benign flows, "
            f"{np.mean(p_benign[attack] < pc):.2%} of known-attack flows")

    configs, rows = [], []
    for budget in BUDGETS:
        for p_cut in P_CUTS:
            tau = fit_tau(p_benign, err, rf_flag, yv, p_cut, budget)
            if tau is None:
                log(f"skip {name_of(p_cut, budget)}: budget below the RF's own FPR")
                continue
            m = binary_metrics(yv, apply_rule(p_benign, err, rf_flag, p_cut, tau))
            name = name_of(p_cut, budget)
            configs.append({"name": name, "budget": budget, "p_cut": p_cut, "tau": tau})
            rows.append({"config": name, "tau": tau, **m})

    table = pd.DataFrame(rows)
    table.to_csv(RES / "gate_val_grid.csv", index=False)
    save_json(CONFIG_FILE, configs)
    log("VALIDATION (known attacks only; unseen-attack benefit cannot be measured here):\n"
        + table.to_string(index=False))
    log(f"Saved {CONFIG_FILE}. Review, then run --stage test once.")


#TEST (once): all pre-registered configs + held-out PortScan
def stage_test(force=False):
    lock = RES / "GATE_TEST_USED.txt"
    if lock.exists() and not force:
        raise SystemExit("Gate test already run. Use --force only if intentional.")
    configs = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    rf = joblib.load(MOD / "hybrid_rf.joblib")
    fm = load_feature_model()
    X_te, y_te = load_split("test")
    yt = y_te.to_numpy()
    p_te, e_te, f_te = score(rf, fm, X_te)

    X_ps = pd.read_parquet(DATA / "X_heldout_portscan.parquet").reset_index(drop=True)
    p_ps, e_ps, f_ps = score(rf, fm, X_ps)

    rows = [{"config": "stacked_rf_only", "tau": None, **binary_metrics(yt, f_te),
             "portscan_caught": float(f_ps.mean())}]
    for c in configs:
        flag = apply_rule(p_te, e_te, f_te, c["p_cut"], c["tau"])
        ps = apply_rule(p_ps, e_ps, f_ps, c["p_cut"], c["tau"])
        rows.append({"config": c["name"], "tau": c["tau"], **binary_metrics(yt, flag),
                     "portscan_caught": float(ps.mean())})

    table = pd.DataFrame(rows)
    table.to_csv(RES / "gate_test_summary.csv", index=False)
    save_json(RES / "gate_test_summary.json", rows)
    lock.write_text(f"evaluated {time.strftime('%Y-%m-%d %H:%M')}", encoding="utf-8")
    log("TEST (all pre-set configs reported, none selected on test):\n" + table.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Confidence-gated anomaly rule")
    ap.add_argument("--stage", required=True, choices=["val", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stage_val() if a.stage == "val" else stage_test(a.force)
