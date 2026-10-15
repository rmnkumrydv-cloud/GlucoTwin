# Ablation: what each data stream adds (cross-validation, development patients)

| Inputs | RMSE 30 min | RMSE 60 min | RMSE 90 min | RMSE 120 min | Spike AUROC |
|---|---|---|---|---|---|
| CGM only | 18.3 | 25.7 | 29.4 | 31.2 | 0.774 |
| CGM + EHR | 18.3 | 25.6 | 29.2 | 31.0 | 0.770 |
| CGM + EHR + meals | 17.4 | 24.5 | 28.1 | 30.1 | 0.802 |
| CGM + EHR + meals + activity | 17.5 | 24.6 | 28.1 | 30.1 | 0.800 |
| all streams except EHR | 17.2 | 24.4 | 28.2 | 30.3 | 0.793 |
| all streams (main model) | 17.4 | 24.4 | 28.0 | 30.1 | 0.799 |
