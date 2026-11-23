"""
main.py - Complete Preprocessed Data Verification & Integrity Test Suite
Verifies data readiness for all 4 models:
  - Member 1 (Random Forest) & Member 2 (XGBoost) -> Supervised Multi-class Splits
  - Member 3 (Isolation Forest) & Member 4 (Deep Autoencoder) -> Unsupervised & Binary Anomaly Splits
Also verifies the 'data/processed/log1p/' dataset

Run from project root:
    python main.py
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
LOG1P_DIR = PROCESSED_DIR / "log1p"


class DataVerificationSuite:
    # one instance per data folder (main or log1p), collects fails and warnings
    def __init__(self, base_dir: Path, is_log1p: bool = False):
        self.dir = base_dir
        self.is_log1p = is_log1p
        self.tag = "LOG1P" if is_log1p else "MAIN"
        self.failures = []
        self.warnings = []

    def log_pass(self, msg: str):
        print(f"  [PASS] {msg}")

    def log_fail(self, msg: str):
        print(f"  [FAIL] {msg}")
        self.failures.append(msg)

    def log_warn(self, msg: str):
        print(f"  [WARN] {msg}")
        self.warnings.append(msg)

    def check_metadata(self):
        """Validates scaler, feature definitions, and label mappings."""
        print(f"\n--- [{self.tag}] 1. METADATA & ARTIFACT CHECKS ---")

        # Resolve artifact paths (checking local folder first, then falling back to parent processed dir)
        feature_path = self.dir / "feature_names.json" if (self.dir / "feature_names.json").exists() else PROCESSED_DIR / "feature_names.json"
        scaler_path = self.dir / "scaler.joblib" if (self.dir / "scaler.joblib").exists() else PROCESSED_DIR / "scaler.joblib"
        mapping_path = self.dir / "label_mapping.json" if (self.dir / "label_mapping.json").exists() else PROCESSED_DIR / "label_mapping.json"

        # 1. Feature Names Check
        if not feature_path.exists():
            self.log_fail(f"Missing feature_names.json (checked {feature_path})")
            return None, None, None

        with open(feature_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        self.log_pass(f"Loaded feature list: {len(features)} total features")

        # 2. Scaler Check
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            self.log_pass(f"Loaded scaler ({type(scaler).__name__}) with {scaler.n_features_in_} features")
            if scaler.n_features_in_ != len(features):
                self.log_fail(f"Scaler dimension ({scaler.n_features_in_}) != feature list count ({len(features)})")
        else:
            self.log_warn(f"Scaler artifact not found at {scaler_path}")

        # 3. Label Mapping Check (Always load for both MAIN and LOG1P)
        label_map = None
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            self.log_pass(f"Loaded label mapping ({len(label_map)} classes)")
            if "BENIGN" not in label_map and 0 not in label_map.values():
                self.log_fail("Label mapping missing normal 'BENIGN' baseline class (ID 0)")
        else:
            self.log_fail(f"Missing label_mapping.json at {mapping_path}")

        return features, scaler, label_map

    def check_matrix(self, name: str, df: pd.DataFrame, expected_cols: list, is_bounded_01: bool = True):
        """Verifies dimension, NaNs, infinities, variance, and [0, 1] scaling bounds."""
        if df is None:
            return

        # 1. Null and Infinity checks
        nan_count = int(df.isna().sum().sum())
        inf_count = int(np.isinf(df.to_numpy(dtype=np.float32)).sum())

        if nan_count > 0:
            self.log_fail(f"{name}: contains {nan_count:,} NaN values")
        if inf_count > 0:
            self.log_fail(f"{name}: contains {inf_count:,} infinite values")
        if nan_count == 0 and inf_count == 0:
            self.log_pass(f"{name}: 0 NaNs and 0 Infs across {df.shape[0]:,} rows")

        # 2. Column alignment
        if list(df.columns) != expected_cols:
            self.log_fail(f"{name}: columns do not match feature_names.json schema/order")
        else:
            self.log_pass(f"{name}: column count and header sequence match ({len(df.columns)} cols)")

        # 3. [0, 1] Boundary constraints
        # small tolerance for float32 rounding
        v_min = float(df.min().min())
        v_max = float(df.max().max())
        if is_bounded_01:
            if v_min < -1e-5 or v_max > 1.0 + 1e-5:
                self.log_fail(f"{name}: Values out of [0, 1] bounds! Range: [{v_min:.4f}, {v_max:.4f}]")
            else:
                self.log_pass(f"{name}: Values strictly bounded within [0, 1] (Min: {v_min:.4f}, Max: {v_max:.4f})")
        else:
            self.log_pass(f"{name}: Observed feature range [{v_min:.4f}, {v_max:.4f}]")

        # 4. Zero-variance check
        stds = df.std(axis=0)
        zero_var_cols = stds[stds == 0.0].index.tolist()
        if zero_var_cols:
            self.log_warn(f"{name}: Found {len(zero_var_cols)} constant (zero-variance) columns: {zero_var_cols[:5]}...")

    def run_supervised_checks(self, features: list, label_map: dict):
        # Checks readiness for RF & XGB
        print(f"\n--- [{self.tag}] 2. SUPERVISED DATA CHECKS (Random Forest & XGBoost) ---")
        splits = ["train", "val", "test"]

        for split in splits:
            x_file = self.dir / f"X_{split}_supervised.parquet"
            y_file = self.dir / f"y_{split}_supervised.parquet"

            if not x_file.exists():
                self.log_fail(f"Missing {x_file.name}")
                continue
            if not y_file.exists():
                self.log_fail(f"Missing {y_file.name}")
                continue

            X = pd.read_parquet(x_file)
            y = pd.read_parquet(y_file).squeeze()

            if len(X) != len(y):
                self.log_fail(f"{split} split length mismatch: X has {len(X)} rows, y has {len(y)} rows")
            else:
                self.log_pass(f"{split} split synchronized: {len(X):,} samples")

            self.check_matrix(f"X_{split}_supervised", X, features, is_bounded_01=True)

            # Robust label verification (Casting all IDs to standard int to prevent type mismatches)
            unique_labels = set(int(val) for val in y.unique())
            valid_ids = set(int(val) for val in label_map.values()) if label_map else set()

            if not valid_ids:
                self.log_fail(f"Validation failed: valid label ID list is empty for y_{split}_supervised")
            elif not unique_labels.issubset(valid_ids):
                self.log_fail(f"y_{split}_supervised contains unknown label IDs: {unique_labels - valid_ids}")
            else:
                self.log_pass(f"y_{split}_supervised contains {len(unique_labels)} classes aligned with label_mapping")

    def run_anomaly_checks(self, features: list):
       # Checks readiness for IF & AE
        print(f"\n--- [{self.tag}] 3. ANOMALY DATA CHECKS (Isolation Forest & Autoencoder) ---")

        # 1. Pure Benign Train Set (M3 & M4 fit exclusively on normal traffic)
        benign_file = self.dir / "X_train_anomaly_benign.parquet"
        if not benign_file.exists():
            self.log_fail(f"Missing {benign_file.name}")
        else:
            X_benign = pd.read_parquet(benign_file)
            self.log_pass(f"X_train_anomaly_benign: Loaded {len(X_benign):,} pure normal training flows")
            self.check_matrix("X_train_anomaly_benign", X_benign, features, is_bounded_01=True)

        # 2. Validation & Test splits with binary labels (0 = Normal, 1 = Attack)
        for split in ["val", "test"]:
            x_file = self.dir / f"X_{split}_anomaly.parquet"
            y_file = self.dir / f"y_{split}_anomaly_binary.parquet"

            if not x_file.exists():
                self.log_fail(f"Missing {x_file.name}")
                continue
            if not y_file.exists():
                self.log_fail(f"Missing {y_file.name}")
                continue

            X = pd.read_parquet(x_file)
            y = pd.read_parquet(y_file).squeeze()

            if len(X) != len(y):
                self.log_fail(f"Anomaly {split} split mismatch: X has {len(X)} rows, y has {len(y)} rows")
            else:
                self.log_pass(f"Anomaly {split} split synchronized: {len(X):,} samples")

            self.check_matrix(f"X_{split}_anomaly", X, features, is_bounded_01=True)

            unique_bins = set(int(val) for val in y.unique())
            if not unique_bins.issubset({0, 1}):
                self.log_fail(f"y_{split}_anomaly_binary contains non-binary targets: {unique_bins}")
            else:
                normal_cnt = int((y == 0).sum())
                attack_cnt = int((y == 1).sum())
                self.log_pass(f"y_{split}_anomaly_binary validated: {normal_cnt:,} Normal (0), {attack_cnt:,} Attacks (1)")

        # 3. Held-out Zero-day subset 
        heldout_x = self.dir / "X_heldout_portscan.parquet"
        heldout_y = self.dir / "y_heldout_portscan.parquet"
        if heldout_x.exists():
            X_held = pd.read_parquet(heldout_x)
            self.log_pass(f"X_heldout_portscan: Found {len(X_held):,} zero-day PortScan samples")
            self.check_matrix("X_heldout_portscan", X_held, features, is_bounded_01=True)

            if heldout_y.exists():
                y_held = pd.read_parquet(heldout_y).squeeze()
                if (y_held != 1).sum() > 0:
                    self.log_warn("y_heldout_portscan contains targets other than 1")
                else:
                    self.log_pass(f"y_heldout_portscan verified (all {len(y_held):,} marked as anomaly target 1)")
        else:
            self.log_warn("X_heldout_portscan.parquet not found in this folder")

    def run_all(self):
        print("=" * 70)
        print(f"VERIFYING DATA READINESS: {self.dir}")
        print("=" * 70)

        if not self.dir.exists():
            self.log_fail(f"Target directory does not exist: {self.dir}")
            return False

        features, scaler, label_map = self.check_metadata()
        if not features:
            return False

        # Run supervised tests if files exist
        # (log1p folder is allowed to have no supervised files)
        sup_exists = (self.dir / "X_train_supervised.parquet").exists()
        if sup_exists:
            self.run_supervised_checks(features, label_map)
        elif not self.is_log1p:
            self.log_fail("Supervised parquet files missing in main data/processed/ directory!")

        # Anomaly checks apply to both main data and log1p data
        self.run_anomaly_checks(features)

        # Verdict
        print(f"\n{'=' * 70}")
        print(f"SUMMARY VERDICT FOR [{self.tag}]: {len(self.failures)} FAILS, {len(self.warnings)} WARNINGS")
        print(f"{'=' * 70}")
        if self.failures:
            print(f"[!] FAILED CHECKS in {self.tag}:")
            for f in self.failures:
                print(f"    - {f}")
            return False
        else:
            print(f"[*] {self.tag} DATASET IS FULLY VALIDATED AND READY FOR MODEL TRAINING.")
            return True


def main():
    # 1. Test Base Processed Directory (used by all 4 models)
    main_suite = DataVerificationSuite(base_dir=PROCESSED_DIR, is_log1p=False)
    main_ok = main_suite.run_all()

    # 2. Test Log1p Processed Directory (Used for Ablation in IsoForest & Autoencoder)
    print("\n" * 2)
    if LOG1P_DIR.exists():
        log1p_suite = DataVerificationSuite(base_dir=LOG1P_DIR, is_log1p=True)
        log1p_ok = log1p_suite.run_all()
    else:
        print("=" * 70)
        print(f"[!] log1p directory not found at: {LOG1P_DIR}")
        print("    If you have not generated the log1p ablation set yet, run your")
        print("    data_pipeline.py with log1p transformation enabled.")
        print("=" * 70)
        # missing log1p is not treated as a failure
        log1p_ok = True

    # Global Status
    print("\n" + "#" * 70)
    if main_ok and log1p_ok:
        print("# OVERALL PIPELINE STATUS: READY FOR PHASE 4 MODEL TRAINING (ALL 4 MODELS) #")
    else:
        print("# OVERALL PIPELINE STATUS: ACTION REQUIRED - FIX PREPROCESSING ERRORS ABOVE #")
    print("#" * 70)


if __name__ == "__main__":
    main()