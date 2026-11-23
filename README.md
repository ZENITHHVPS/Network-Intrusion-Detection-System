# Random Forest and SHAP

**Member 1** · `<Priyanka / CS24B1045>`

This folder has the supervised Random Forest classifier and the SHAP explainability step for our hybrid NIDS project on CICIDS2017. The RF labels each flow as BENIGN or one of 11 known attacks. It was chosen over XGBoost as the supervised model for the hybrid. SHAP explains all five models.

Full details are in the project report.

## Setup

-Data: shared pipeline output (src/preprocessing/data_pipeline.py), 68 features, 12 classes, seed 42
- Model: 500 trees, max_depth 20, max_features 0.5, square-root class weights

## How to run

```bash
pip install -r requirements.txt
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
