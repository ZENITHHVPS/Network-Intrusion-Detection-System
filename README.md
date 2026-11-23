# Deep Autoencoder

**Member 4** · `<Sonali / CS24B1060>`

This folder has the deep Autoencoder anomaly detector for our hybrid NIDS project on CICIDS2017. It is trained on benign traffic only and flags a flow as an attack when its reconstruction error is above a threshold τ. It was chosen over Isolation Forest as the anomaly model for the hybrid.

Full details are in the project report.

## Setup

- Data: 1,304,306 benign training flows, 68 log1p-scaled features
- Architecture: 68 → 64 → 32 → 16 (latent) → 32 → 64 → 68
- Training: Adam, batch size 512, up to 50 epochs; τ set by best F1 on validation

## Key results (test set, log1p model)

| Metric | Value |
|---|---|
| ROC-AUC | 96.95% |
| F1 | 82.83% |
| False alarm rate | 2.40% |
| Held-out PortScan recall | 47.19% |
| Latency | 2.85 µs per flow |
