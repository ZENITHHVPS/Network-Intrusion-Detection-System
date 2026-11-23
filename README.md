# Network Intrusion Detection System (NIDS)

**CS3006 Introduction to Data Science for Engineers** · Professor: Dr. Jagadeesh Kakarla

*A Comparative Study and Implementation of Hybrid Framework for Network Intrusion Detection Using Machine Learning, Deep Learning, and Anomaly Detection*

This project benchmarks four models on the CICIDS2017 dataset and combines the best supervised model and the best anomaly detector into a hybrid NIDS. PortScan is held out of training to simulate an unseen attack. SHAP explains the models, and a Streamlit dashboard demonstrates them on real test flows.

Full details are in the project report.

## Team

| Roll no. | Name |
|---|---|
| CS24B1011 | Harini Senthil Kumar |
| CS24B1030 | Vennela V. |
| CS24B1045 | Priyanka HL |
| CS24B1060 | Maddi Sonali Reddy |

## Models

| Category | Model | Selected for hybrid |
|---|---|---|
| Supervised | Random Forest | ✅ |
| Supervised | XGBoost | |
| Anomaly detection | Isolation Forest | |
| Anomaly detection | Deep Autoencoder | ✅ |
| Hybrid | Autoencoder + Random Forest with Confidence Gate | |

## Setup

- Dataset: CICIDS2017 (2.8M flows), cleaned to 68 features and 12 classes
- Split: 70 / 15 / 15 train / validation / test, stratified, seed 42
- Held out: 1,958 PortScan flows, never used for training

## How to run

```bash
pip install -r requirements.txt
python data_pipeline.py      # builds the processed Parquet data
# train each model using the instructions in its folder's README
python generate_shap.py      # SHAP plots saved to results/shap/
streamlit run app.py         # dashboard
```

## Key results (test set)

| Model | Main metric | False alarms | PortScan recall |
|---|---|---|---|
| Random Forest (tuned) | Macro F1 97.44% | 0.085% | – |
| XGBoost (tuned) | Macro F1 96.79% | 0.095% | – |
| Isolation Forest (log1p) | ROC-AUC 93.99% | 6.05% | 22.06% |
| Autoencoder (log1p) | ROC-AUC 96.95% | 2.40% | 47.19% |
| Hybrid, stacked | Macro F1 97.30% | 0.086% | 6.23% |
| Hybrid + Confidence Gate (0.5% budget) | Macro F1 83.42% | 0.50% | **85.34%** |

The stacked hybrid matches the standalone Random Forest on known attacks. Adding the Confidence Gate raises detection of the unseen PortScan attack from 6.23% to 85.34% at a 0.5% false-alarm budget.

## Contributions

| Member | Contribution | branch |
|---|---|---|
| Harini Senthil Kumar | XG Boost, Hybrid NIDS| m2 |
| Vennela V. | Isolation forest, Hybrid NIDS| mem3 |
| Priyanka HL | Data cleaning, Random forest, SHAP| member1 |
| Maddi Sonali Reddy | Data preprocessing, Autoencoder, dashboard| member4|
