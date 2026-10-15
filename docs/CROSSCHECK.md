# Cross-check: "GlucoTwin-X" and "Locked project definition" vs this repository

Verdict per proposal. **Adopted** = implemented and tested. **Already had** = the repository
already did this. **Corrected** = the idea is right but a detail was wrong for CGMacros.
**Deferred** = good idea, not worth the time before Phase 1.

## Data and leakage (the most important section)

| Proposal | Verdict | What we did |
|---|---|---|
| CGMacros CGM values are linearly interpolated to 1 minute, so interpolated points contain the next reading | **Adopted (verified)** | The CGMacros paper in *Scientific Data* confirms Libre (15 min) and Dexcom (5 min) were linearly interpolated to 1 minute. `data.py` now recovers the ORIGINAL readings (phase search per 12 h block) and uses only those. |
| Work at 5-minute resolution, Dexcom as primary | Already had | Sensor chosen per patient; Dexcom preferred. |
| Do not treat the two CGMs as independent samples; use the second for agreement checks | **Adopted** | One sensor per patient; `patient_info.csv` now reports Dexcom-vs-Libre MARD for quality control. |
| (Found while checking) our 5-minute bins were labelled at their START, so the "12:00" row contained data up to 12:04 | **Fixed** | All bins are now right-labelled (the 12:00 row covers 11:55 to 12:00); meals appear at the first grid time after they are logged; short gaps are filled by carrying the last reading forward (causal), never by interpolation. A new test perturbs raw 1-minute data after time T and proves no stream row at or before T changes. |
| No random-window splitting | Already had | Leave-patients-out 5-fold CV throughout. |
| Missingness as information | **Adopted** | `glucose_age_min`, `glucose_missing_60`, `hr_missing_60`, `hr_age_min`. |
| Data-quality layer | **Adopted** | Duplicate timestamps removed and counted, gap count, longest gap, sensor agreement, phase clarity per patient. |
| Exclude the ~2,000 microbiome features | Already had | Never used (45 people cannot support them). |
| (Second audit) tuning, ensemble weights and alert thresholds were chosen on the same patients CV reported on | **Fixed** | A locked, stratified test set (9 of 45 patients) is excluded from every development step; `python -m glucotwin.test` scores it once. CV numbers are now "development" numbers. |
| (Second audit) personalization calibration rows near the cutoff had targets inside the evaluation days | **Fixed** | Purge gap equal to the longest horizon; a model may only be calibrated on patients it never trained on. |
| (Second audit) the dashboard could show personalized forecasts on the calibration days themselves | **Fixed** | Personalization is applied only after each patient's calibration cutoff. |
| (Second audit) tuning used the scored fold for early stopping | **Fixed** | Early stopping uses inner validation patients from the training side. |

## Features and targets

| Proposal | Verdict | What we did |
|---|---|---|
| EWMA, coefficient of variation, 6-hour context, HR slope | **Adopted** | Added to `features.py` (now 93 features). |
| Steps / sleep features | **Corrected** | CGMacros' Fitbit export has heart rate, METs and activity calories only: no steps, no sleep. We use METs and active minutes. |
| Predict the whole 24-step trajectory | **Adopted in the neural model** | P-MRGF outputs all 24 steps; LightGBM keeps 4 horizons and the service now interpolates the expected time of crossing 180 mg/dL. |

## Models

| Proposal | Verdict | What we did |
|---|---|---|
| Baseline ladder M0 persistence, M1 linear, M2 gradient boosting | Already had | Persistence, linear trend, ridge, LightGBM. |
| M3/M5: TCN backbone, FiLM conditioning on the EHR, residual gates for meals, activity, heart rate | **Adopted (compact)** | `nn.py`: dilated causal TCN (1, 2, 4, 8, 16), FiLM from EHR, engineered-feature MLP, zero-initialised gated residual branches (the MealRes-Gate idea), quantile trajectory head and spike-risk head. |
| Multi-task loss: zone-weighted Huber, pinball, velocity, risk | **Adopted** | Exactly this, weights in `configs/config.yaml -> nn`. |
| Zone-weighted training for the tree model | **Adopted as an option** | `model.zone_weights` (off by default; test it as an experiment). |
| M7 ensemble with per-horizon weights learned on validation data | **Adopted, made leakage-safe** | `ensemble.py` picks each fold's weights using only the other folds' unseen patients. If the network does not help, its weight goes to 0. |
| Patient identity embeddings | **Rejected** | They cannot help a new patient and overfit with 45 people. The calibrated twin (residual model on the first days) already gives the cold-start vs calibrated story. |
| M4 Transformer, teacher-student distillation | **Deferred** | Only if P-MRGF beats LightGBM on real data. Our "plan dropout" already gives a simple, deployable version of the future-aware idea. |
| Optuna 50-100 trials for the neural model | **Corrected** | Each neural trial is a full cross-validated training run: 50-100 trials is days of CPU time. Tune LightGBM with Optuna; tune the network by hand on 3 to 4 settings (context 3 h vs 6 h, channels, dropout), or use a free Colab/Kaggle GPU. |

## Evaluation

| Proposal | Verdict | What we did |
|---|---|---|
| Do not chase R2 = 0.99784 | **Adopted** | Results now show R2 for GlucoTwin AND for "no change" persistence, plus R2 of the predicted CHANGE, the honest skill measure. |
| Brier score, calibration, specificity, Pearson r | **Adopted** | Added, with a calibration figure. Accuracy is deliberately not reported. |
| AUROC, AUPRC, recall, false alerts per day, lead time, Clarke grid | Already had | |
| Parkes error grid | Deferred | Parkes has separate type 1 and type 2 grids; Clarke is enough for the jury. |
| Fixed split P001-P032 / P033-P038 / P039-P045 (document 2) | **Adopted, stratified** | A locked test set of 9 patients (3 per group, not consecutive IDs) plus 5-fold patient CV on the other 36 for development. Test results come with bootstrap confidence intervals because 9 patients is few. |
| Regime A: within-patient chronological split | Already had (as personalization) | Population model + calibration on first N days, tested on the patient's later days. Try `calibration_days: 1` and `2` (document 2 suggests 24-48 h). |
| Regime C: external validation on ShanghaiT2DM | Deferred | Good post-Phase-1 work (Asian type 2 cohort, 15-minute CGM). Check its license first. |
| Results by group (healthy, prediabetes, T2D) | **Adopted** | Table in `RESULTS.md`. |

## Product and engineering

| Proposal | Verdict | What we did |
|---|---|---|
| "New sensor reading" twin-update moment | **Adopted** | Dashboard button moves time 5 minutes and shows the previous vs updated forecast. |
| What-if clearly labelled as simulation, not advice | **Adopted** | Warning in the What-if tab. |
| Expected time of threshold crossing | **Adopted** | Shown in the risk panel. |
| Docker | **Adopted** | `Dockerfile` + `docker-compose.yml` (not tested in our build environment; test on a laptop). |
| MLflow | **Replaced** | `artifacts/experiments.csv` logs every development run and `artifacts/test_runs.csv` every test run; simpler for a student team. |
| Synthea for synthetic EHR | **Corrected** | Synthea people cannot be linked to CGMacros participants. Our flagged synthetic fields attached to real participants are the right approach; Synthea is useful only to stress-test the FHIR interface. |
| Different folder layout (`src/data`, `src/models`, `src/twin`) | Not adopted | Same separation exists in flat modules; restructuring costs time and adds no value. |

## Build order with go/no-go gates

1. **Gate 0, data locked (day 2 to 3):** EDA checklist done; leakage tests green on real data;
   Dexcom phase clarity high for most patients (`patient_info.csv`).
2. **Gate 1, strong baseline (day 3 to 4):** LightGBM beats persistence and ridge at every horizon.
   Tune it with Optuna.
3. **Gate 2, neural model (day 5 to 7):** run P-MRGF and the ensemble. Keep the ensemble only if
   it lowers error at two or more horizons or raises spike AUPRC. Otherwise report "the network
   did not beat gradient boosting on 45 patients" as a result; that is honest and juries respect it.
4. **Gate 3, ablation table (day 6 to 8):** `bash scripts/run_ablations.sh` plus the neural
   `nn.branches` ablation: CGM only, + EHR, + meals, + activity, + heart rate, + personalization,
   + ensemble.
5. **After Phase 1:** Transformer branch, distillation, ShanghaiT2DM external validation.
