# Isolation Forest and Hybrid Model

**Member 3** · Vennela V / CS24B1030

This folder has two parts of our NIDS project on CICIDS2017:

- **Isolation Forest:** an anomaly detector trained on benign traffic only, compared with the Autoencoder.
- **Hybrid AE + RF:** the frozen Autoencoder adds 16 latent features and its reconstruction error to the 68 flow features (85 in total), and a Random Forest classifies the result. A **Confidence Gate** also flags a flow when the RF is unsure it is benign (P(BENIGN) < p_cut) and the AE error is above τ.

Full details are in the project report.

## How to run

```bash
pip install -r requirements.txt
python data_pipeline.py
python isolation_forest.py            # Isolation Forest (Script A)
python evaluate_isolation_forest.py   # Isolation Forest (Script B)
python train_hybrid_ae_rf.py          # saves hybrid_rf.joblib
streamlit run app.py                  # dashboard
```

## Key results (test set)

**Isolation Forest (log1p):** ROC-AUC 93.99%, false alarm rate 6.05%, PortScan recall 22.06%. The Autoencoder did better, so it was used in the hybrid.

**Hybrid:**

| Model | Macro F1 | False alarms | PortScan recall |
|---|---|---|---|
| Stacked AE + RF | 97.30% | 230 | 6.23% |
| Confidence Gate (p_cut 0.99, 0.5% budget) | 83.42% | 1,352 | **85.34%** |

The stacked model matches the standalone RF on known attacks. The Confidence Gate catches far more of the unseen PortScan attack at a 0.5% false-alarm budget. Training the hybrid RF on 85 features took about 40 min (estimated).
