# Ensemble comparison (cross-validation, development patients)

Same unseen development patients for all three models. Neural-model weight used for the locked test, per horizon: 30: 0.30, 60: 0.25, 90: 0.20, 120: 0.20, spike: 0.20.

RMSE in mg/dL (lower is better):

| Model | 30 min | 60 min | 90 min | 120 min |
|---|---|---|---|---|
| Ensemble | 17.3 | 24.3 | 27.9 | 29.9 |
| LightGBM | 17.4 | 24.4 | 28.0 | 30.1 |
| P-MRGF (neural) | 18.1 | 26.0 | 30.9 | 33.5 |

Spike AUROC: LightGBM 0.799, P-MRGF (neural) 0.785, Ensemble 0.815
