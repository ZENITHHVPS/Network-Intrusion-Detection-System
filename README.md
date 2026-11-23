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

Results at a glance (test set, 317,953 flows)
Metric	Baseline	Tuned
Macro F1	0.9209	0.9679
Macro F1 (excl. Heartbleed, Infiltration)	0.9479	0.9615
Macro Recall	0.9016	0.9873
Macro Precision	0.9655	0.9543
MCC	0.9940	0.9963
Attack Detection Rate	99.512%	99.892%
False Positive Rate	0.1001%	0.0949%
Missed attacks	245	54 (−78%)
False alarms	268	254
Bot recall	0.588	0.889

Team decision: The tuned Random Forest scored slightly higher (macro-F1 97.44% vs 96.79%, with higher macro-recall and a lower FPR), so the hybrid NIDS uses RF. XGBoost remains the second benchmarked supervised model.
