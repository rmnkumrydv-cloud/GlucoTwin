# GlucoTwin development results (cross-validation)

Leave-patients-out cross-validation over 36 development patients: each one is scored by a model that never saw them. Use these numbers to choose settings. The 9 locked test patients were not used; their results come from `python -m glucotwin.test` (artifacts/test/RESULTS_TEST.md).

## Glucose forecast error (RMSE, mg/dL; lower is better)

| Model | 30 min | 60 min | 90 min | 120 min |
|---|---|---|---|---|
| Persistence (no change) | 20.6 | 30.0 | 36.3 | 40.7 |
| Linear trend (last 30 min) | 28.0 | 48.3 | 63.3 | 74.7 |
| Ridge regression | 18.3 | 26.3 | 30.4 | 32.6 |
| **GlucoTwin (honest: future meals and activity unknown)** | **17.4** | **24.5** | **28.1** | **30.1** |
| GlucoTwin personalized (scored after the calibration days) | 16.5 | 23.4 | 27.0 | 29.0 |
| same rows, population model (for comparison) | 16.9 | 23.7 | 27.2 | 29.0 |
| Oracle: real future meals and activity given (what-if accuracy, not a forecast) | 17.3 | 24.3 | 27.8 | 29.2 |

## Clinical accuracy (Clarke error grid, honest mode)

| Horizon | MAE | MARD % | Zone A % | Zone A+B % | Zone D+E % | 80% band coverage % |
|---|---|---|---|---|---|---|
| 30 min | 11.4 | 8.1 | 91.5 | 99.6 | 0.39 | 81.6 |
| 60 min | 16.2 | 11.3 | 83.4 | 99.1 | 0.85 | 82.6 |
| 90 min | 18.8 | 13.1 | 78.1 | 98.8 | 1.14 | 82.9 |
| 120 min | 20.3 | 14.1 | 75.7 | 98.5 | 1.40 | 83.3 |

## R2: read this before quoting it

R2 on absolute glucose is inflated by autocorrelation: doing nothing (persistence) already scores high. R2 of the predicted *change* is the honest skill measure.

| Horizon | R2 absolute (GlucoTwin) | R2 absolute (persistence) | R2 of change (GlucoTwin) | Pearson r |
|---|---|---|---|---|
| 30 min | 0.841 | 0.777 | 0.286 | 0.918 |
| 60 min | 0.685 | 0.526 | 0.336 | 0.828 |
| 90 min | 0.584 | 0.306 | 0.400 | 0.765 |
| 120 min | 0.522 | 0.130 | 0.451 | 0.723 |

## By metabolic group

| Group | Patients | RMSE 30 min | RMSE 60 min | RMSE 90 min | RMSE 120 min | Spike AUROC |
|---|---|---|---|---|---|---|
| healthy | 12 | 14.8 | 18.9 | 20.6 | 21.4 | 0.796 |
| prediabetes | 13 | 17.1 | 23.4 | 26.1 | 27.3 | 0.763 |
| t2d | 11 | 20.0 | 30.3 | 36.2 | 39.7 | 0.761 |

## Adverse-event prediction: glucose excursion within 2 hours

- Rows evaluated: 85,388 (positive rate 15.4%)
- AUROC 0.802, AUPRC 0.463, Brier score 0.113 (accuracy is deliberately not reported: with 15% positives it rewards a model that never alerts)
- At the threshold chosen on development patients: precision 0.50, recall (sensitivity) 0.39, specificity 0.93, F1 0.44
- Event level: 614 crossings above 180 mg/dL; 70% caught in advance, median warning 120.0 min (longest measurable: 120 min), 1.51 false alerts per patient-day, alert active 21% of the time

## Figures

![clarke_120min](figures/clarke_120min.png)
![clarke_60min](figures/clarke_60min.png)
![example_forecast](figures/example_forecast.png)
![feature_groups](figures/feature_groups.png)
![personalization](figures/personalization.png)
![rmse_by_horizon](figures/rmse_by_horizon.png)
![spike_calibration](figures/spike_calibration.png)

## Run time

- cross-validation: 185 s
- development model: 49 s
- total: 239 s
