# XGBoost

**Member 2 · Harini Senthil Kumar (CS24B1011)**

This folder has the XGBoost classifier for our hybrid NIDS project on CICIDS2017. It labels each flow as BENIGN or one of 11 known attacks, and it was compared with the Random Forest for known-attack classification.

Full details are in the project report.

## Setup

- **Data:** shared pipeline output (`data_pipeline.py`), 68 features, 12 classes, seed 42
- **Model:** `max_depth` 10, `learning_rate` 0.2, balanced sample weights, early stopping (50 rounds)

## How to run

```bash
pip install -r requirements.txt
python data_pipeline.py
python train_xgboost.py --stage baseline
python train_xgboost.py --stage tune
python train_xgboost.py --stage final
python train_xgboost.py --stage test      # one-time test evaluation
```

Add `--device cuda` to train on a GPU, or `--quick` to test the code on a 10% sample.

**Saved models:** `xgb_baseline.joblib`, `xgb_tuned.joblib`

## Key results (test set)

| Metric | Tuned XGBoost |
|---|---|
| Accuracy | 99.90% |
| Macro F1 | 96.79% |
| Attack detection rate | 99.89% |
| False positive rate | 0.095% |
| Training time (final fit) | 96 min |

The tuned Random Forest scored slightly higher (macro F1 97.44%), so it was used in the hybrid.
