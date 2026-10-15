# Timeline, effort and compute

## Measured compute times

Measured on a 2-core cloud machine (8 GB RAM) with a synthetic cohort the same size as CGMacros
(45 patients x 10 days, about 128,000 five-minute rows, 93 features). A 4 to 8 core laptop is faster.

| Step | Command | Time |
|---|---|---|
| Download CGMacros (630 MB) | `python scripts/download_data.py` | 5 to 20 min (network) |
| Prepare streams, EHR, split, quality report | `python -m glucotwin.prepare` | under 1 min |
| LightGBM: 5-fold CV (36 patients) + development model | `python -m glucotwin.train` | about 13 min |
| Quick smoke-test training | `python -m glucotwin.train --quick` | 1 to 2 min |
| Hyperparameter tuning | `python -m glucotwin.tune` | capped at 45 min |
| Neural engine P-MRGF: 5 folds + development model | `python -m glucotwin.nn --epochs 8` | about 36 min (45 to 65 s per epoch) |
| Ensemble | `python -m glucotwin.ensemble` | seconds |
| Locked test (LightGBM + neural + ensemble) | `python -m glucotwin.test` | under 1 min |
| Ablations (5 LightGBM runs) | `python scripts/run_ablations.py` | about 1 to 1.5 h (run overnight) |
| Test suite | `python -m pytest -q` | about 2 min |

One LightGBM experiment costs about 15 minutes and one neural experiment 35 to 60. Compute is
not the bottleneck, but run the neural model and the ablations overnight or on a free Colab or
Kaggle machine (`python -m glucotwin.nn --threads 8` uses more CPU threads).

## Team roles (merge roles if you are three)

- **Data and EHR lead**: download, EDA checklist in `docs/DATA.md`, FHIR records, data slides.
- **ML lead**: training runs, tuning, ablations, results tables and figures.
- **App lead**: dashboard and API polish, demo script, screenshots.
- **Clinical and delivery lead**: problem framing, Indian meal values (IFCT 2017), limitations,
  deck, video editing, submission form.

## 15-day plan with go/no-go gates (about 170 to 220 person-hours for 3 to 4 people)

Count backwards from your Phase 1 deadline. The gates come from `docs/CROSSCHECK.md`: do not
start the next stage until the current one passes.

| Day | Goal | Gate / done when |
|---|---|---|
| 1 | Everyone installs; synthetic demo end to end; download real data; read the CGMacros paper | Dashboard opens on every laptop |
| 2 | EDA checklist (`docs/DATA.md`), especially the interpolation section | **Gate 0:** 45 patients, 15/16/14 groups, high `phase_clarity`, leakage tests green |
| 3 | First full LightGBM run on real data (`python -m glucotwin.train`) | **Gate 1:** beats persistence and ridge at every horizon in `artifacts/cv/RESULTS_CV.md` |
| 4 | Optuna tuning, retrain; try `zone_weights` | Results log updated |
| 5 | P-MRGF neural run (`python -m glucotwin.nn`) and `python -m glucotwin.ensemble` | **Gate 2:** keep ensemble only if it helps at 2+ horizons or raises AUPRC |
| 6 | Ablations (`python scripts/run_ablations.py`, neural `nn.branches`) | Ablation table |
| 7 | Personalization days (1 to 4), spike definition, alert threshold; **freeze the model** | Final settings justified with CV numbers only |
| 8 | Verify Indian meal macros against IFCT 2017; case studies for 3 patients | Screenshots |
| 9 | Error analysis on the CV predictions (`artifacts/cv/oof_predictions.parquet`): nights, gaps, unlogged meals, groups | Limitations slide |
| 10 | **Run `python -m glucotwin.test` once**; fill the README results from `artifacts/test/RESULTS_TEST.md`; tests green; repo cleaned | Fresh clone runs end to end |
| 11 to 12 | Presentation and video rehearsal | Deck exported to PDF |
| 13 | Record and edit the 20+ minute video | Unlisted YouTube link |
| 14 | Final check and submission | Form submitted |
| 15 | Buffer | |

After shortlisting: rehearse a 10-minute live demo and a question bank for the Phase 2 jury;
refine the pitch for the 20 November 2026 summit. Transformer branch, distillation and
external validation (ShanghaiT2DM) are post-Phase-1 work.

## Experiment plan for the best model

Change one thing at a time and log every run (date, change, RMSE at 60 and 120 min, spike AUROC,
false alerts per day).

1. **Baseline**: default config. This is the number every other change must beat.
2. **Sensor**: `primary_cgm: dexcom` vs `libre`.
3. **Tuning**: `python -m glucotwin.tune`, then retrain (train.py reads `best_params.yaml`).
4. **Ablations**: `python scripts/run_ablations.py` (CGM only, + EHR, + meals, + activity, + heart
   rate). This shows the jury the value of fusing the two streams.
5. **Carb absorption**: `carb_absorption_tau_min` 30 / 45 / 60.
5b. **Zone weights**: `model.zone_weights: {180: 1.5, 250: 2.0}` for LightGBM.
5c. **Neural context**: `nn.context_min` 180 / 360 / 720; `nn.branches` with one branch removed.
6. **Plan dropout**: 0.3 / 0.5 / 0.7 (trade-off between honest and what-if accuracy).
7. **Spike definition**: `absolute` vs `either`, `spike_rise_mgdl` 40 / 50; threshold `f1` vs
   `recall80`. Choose using false alerts per day, not only AUROC.
8. **Personalization**: `calibration_days` 1 to 4, `ridge_alpha` 3 / 10 / 30.

Honesty rules: every comparison above uses the cross-validation numbers in
`artifacts/experiments.csv`, which come from development patients only. The locked test patients
are scored once, on day 10, by `python -m glucotwin.test`; those are the numbers you report. If you
change anything after seeing them, say so. Never report synthetic-data results.
