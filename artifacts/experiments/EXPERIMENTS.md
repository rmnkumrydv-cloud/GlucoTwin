# Experiments (cross-validation, 36 development patients)

RMSE in mg/dL (lower is better). Personalization gain: how much the personalized twin lowers RMSE (positive = it helps). Spike AUROC/precision/recall depend on the spike definition, so compare D with the others using the event columns: share of crossings above 180 mg/dL warned in advance, false alerts per patient-day and the share of time an alert is showing.

| Experiment | RMSE 30 | RMSE 60 | RMSE 120 | Personal. gain 60 | Personal. gain 120 | Spike AUROC | Precision | Recall | Spikes caught % | False alerts/day | Alert on % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A current (spec90 alerts, residual personalization) | 17.4 | 24.4 | 30.1 | 0.08 | -0.83 | 0.773 | 0.54 | 0.32 | 63.0 | 2.09 | 17.1 |
| B personal sensitivity (gain) | 17.4 | 24.4 | 30.1 | 0.40 | 0.17 | 0.773 | 0.54 | 0.32 | 63.0 | 2.09 | 17.1 |
| C bias-only personalization | 17.4 | 24.4 | 30.1 | 0.19 | 0.25 | 0.773 | 0.54 | 0.32 | 63.0 | 2.09 | 17.1 |
| D spike = glucose reaches 180 only | 17.4 | 24.4 | 30.1 | 0.08 | -0.83 | 0.799 | 0.52 | 0.37 | 66.8 | 1.62 | 19.2 |
