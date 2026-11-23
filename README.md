<<<<<<< HEAD
# XGBoost

**Member 2 · Harini Senthil Kumar (CS24B1011)**

This folder has the XGBoost classifier for our hybrid NIDS project on CICIDS2017. It labels each flow as BENIGN or one of 11 known attacks, and it was compared with the Random Forest for known-attack classification.
=======
# Random Forest and SHAP

**Member 1** · `<Priyanka / CS24B1045>`

This folder has the supervised Random Forest classifier and the SHAP explainability step for our hybrid NIDS project on CICIDS2017. The RF labels each flow as BENIGN or one of 11 known attacks. It was chosen over XGBoost as the supervised model for the hybrid. SHAP explains all five models.
>>>>>>> ceadd6b38bba39569737c8205a901df3c1423323

Full details are in the project report.

## Setup

<<<<<<< HEAD
- **Data:** shared pipeline output (`data_pipeline.py`), 68 features, 12 classes, seed 42
- **Model:** `max_depth` 10, `learning_rate` 0.2, balanced sample weights, early stopping (50 rounds)
=======
- Data: shared pipeline output (src/preprocessing/data_pipeline.py), 68 features, 12 classes, seed 42
- Model: 500 trees, max_depth 20, max_features 0.5, square-root class weights
>>>>>>> ceadd6b38bba39569737c8205a901df3c1423323

## How to run

```bash
pip install -r requirements.txt
<<<<<<< HEAD
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

Team decision: The tuned Random Forest scored slightly higher (macro-F1 97.44% vs 96.79%, with higher macro-recall and a lower FPR), so the hybrid NIDS uses RF. XGBoost remains the second benchmarked supervised model.
=======
python src/preprocessing/data_pipeline.py
python models/random_forest_model/train_random_forest.py
python generate_shap.py              # SHAP plots saved to results/shap/
```

## Key results (test set)

| Metric | Tuned RF |
|---|---|
| Accuracy | 99.91% |
| Macro F1 | 97.44% |
| Attack detection rate | 99.92% |
| False positive rate | 0.085% |
| Training time | 32.74 min |

SHAP shows that packet-length features (Bwd Packet Length Std/Mean) and TCP flag features drive the predictions. In the hybrid model, the Autoencoder's reconstruction error is the top feature.
>>>>>>> ceadd6b38bba39569737c8205a901df3c1423323
