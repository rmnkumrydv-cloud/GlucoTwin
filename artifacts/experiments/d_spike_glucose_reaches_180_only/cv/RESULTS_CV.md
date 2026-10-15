# GlucoTwin development results (cross-validation)

Leave-patients-out cross-validation over 36 development patients: each one is scored by a model that never saw them. Use these numbers to choose settings. The 9 locked test patients were not used; their results come from `python -m glucotwin.test` (artifacts/test/RESULTS_TEST.md).

## Glucose forecast error (RMSE, mg/dL; lower is better)

| Model | 30 min | 60 min | 90 min | 120 min |
|---|---|---|---|---|
| Persistence (no change) | 20.6 | 30.0 | 36.3 | 40.7 |
| Linear trend (last 30 min) | 28.0 | 48.3 | 63.3 | 74.7 |
| Ridge regression | 18.3 | 26.3 | 30.6 | 32.8 |
| **GlucoTwin (honest: future meals and activity unknown)** | **17.4** | **24.4** | **28.0** | **30.1** |
| GlucoTwin personalized (scored after the calibration days) | 17.4 | 23.6 | 27.5 | 29.9 |
| same rows, population model (for comparison) | 16.9 | 23.7 | 27.1 | 29.1 |
| Oracle: real future meals and activity given (what-if accuracy, not a forecast) | 17.3 | 24.3 | 27.7 | 29.2 |

## Clinical accuracy (Clarke error grid, honest mode)

| Horizon | MAE | MARD % | Zone A % | Zone A+B % | Zone D+E % | 80% band coverage % |
|---|---|---|---|---|---|---|
| 30 min | 11.5 | 8.1 | 91.4 | 99.6 | 0.39 | 81.7 |
| 60 min | 16.2 | 11.3 | 83.2 | 99.1 | 0.84 | 82.5 |
| 90 min | 18.7 | 13.0 | 78.4 | 98.8 | 1.15 | 83.2 |
| 120 min | 20.2 | 14.0 | 75.8 | 98.5 | 1.41 | 83.4 |

## R2: read this before quoting it

R2 on absolute glucose is inflated by autocorrelation: doing nothing (persistence) already scores high. R2 of the predicted *change* is the honest skill measure.

| Horizon | R2 absolute (GlucoTwin) | R2 absolute (persistence) | R2 of change (GlucoTwin) | Pearson r |
|---|---|---|---|---|
| 30 min | 0.840 | 0.777 | 0.281 | 0.918 |
| 60 min | 0.686 | 0.526 | 0.338 | 0.830 |
| 90 min | 0.586 | 0.306 | 0.404 | 0.767 |
| 120 min | 0.522 | 0.130 | 0.451 | 0.724 |

## By metabolic group

| Group | Patients | RMSE 30 min | RMSE 60 min | RMSE 90 min | RMSE 120 min | Spike AUROC |
|---|---|---|---|---|---|---|
| healthy | 12 | 14.9 | 18.9 | 20.8 | 21.6 | 0.798 |
| prediabetes | 13 | 17.1 | 23.4 | 25.9 | 27.3 | 0.758 |
| t2d | 11 | 20.2 | 30.2 | 36.0 | 39.6 | 0.749 |

## Adverse-event prediction: glucose excursion within 2 hours

- Rows evaluated: 85,388 (positive rate 15.4%)
- AUROC 0.799, AUPRC 0.455, Brier score 0.111 (accuracy is deliberately not reported: with 15% positives it rewards a model that never alerts)
- At the threshold chosen on development patients: precision 0.52, recall (sensitivity) 0.37, specificity 0.94, F1 0.43
- Event level: 614 crossings above 180 mg/dL; 67% caught in advance, median warning 120.0 min (longest measurable: 120 min), 1.62 false alerts per patient-day, alert active 19% of the time

## Figures

![clarke_120min](figures/clarke_120min.png)
![clarke_60min](figures/clarke_60min.png)
![example_forecast](figures/example_forecast.png)
![feature_groups](figures/feature_groups.png)
![personalization](figures/personalization.png)
![rmse_by_horizon](figures/rmse_by_horizon.png)
![spike_calibration](figures/spike_calibration.png)

## Run time

- cross-validation: 212 s
- development model: 42 s
- total: 259 s
