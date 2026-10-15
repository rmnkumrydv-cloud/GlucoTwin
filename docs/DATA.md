# Data notes

## What we use

**CGMacros v1.0.0**, PhysioNet, open access, license CC BY-NC-SA 4.0.
Page: https://physionet.org/content/cgmacros/1.0.0/  DOI: 10.13026/3z8q-x658

- 45 participants: 15 healthy (HbA1c < 5.7%), 16 prediabetes (5.7 to 6.4%), 14 type 2 diabetes.
- About 10 days each, free living, in Santa Barbara, USA (recruited 2021 to 2024).
- Two blinded CGMs: Abbott FreeStyle Libre Pro (15-minute sampling, upper arm) and Dexcom G6 Pro
  (5-minute sampling, abdomen). Files are at 1-minute rows.
- Fitbit Sense: heart rate, METs (stored multiplied by 10), activity calories.
- Meals: calories, carbs, protein, fat, fiber, amount consumed, meal type, photo path.
  Breakfasts were protein shakes, lunches came from one fast-casual chain, dinners were free
  choice. Participants were asked to leave at least 3 hours between meals.
- `bio.csv`: age, gender, BMI, weight (lb), height (in), ethnicity, HbA1c, fasting glucose,
  fasting insulin, triglycerides, cholesterol, HDL, non-HDL, LDL, VLDL, three fingerstick
  readings.
- Dates are shifted by 365 to 720 days for privacy, so weekday and season are meaningless.

## The interpolation trap (read before touching the data)

The CGMacros paper states that the 1-minute CGM rows were produced by linear interpolation
between consecutive real readings (15 minutes apart for Libre, 5 minutes for Dexcom). An
interpolated value at 12:02 is 60% the 12:00 reading and 40% the 12:05 reading, so it already
"knows" the future. Training on 1-minute rows, or averaging them into bins, silently leaks the
future and inflates every metric.

What `data.py` does about it:

1. For each 12-hour block, it finds the sampling phase whose samples reconstruct the series
   by linear interpolation (the original readings) and keeps only those.
   `patient_info.csv -> phase_clarity` shows how unambiguous that was (1.0 = perfectly clear).
2. The 5-minute grid is right-labelled: the 12:00 row uses data from 11:55 to 12:00.
3. Gaps are bridged only by carrying the last reading forward for up to 10 minutes; longer
   gaps stay missing. Nothing is interpolated.
4. Meals appear at the first grid time after they were logged.
5. `tests/test_pipeline.py::test_no_leakage_from_raw_minute_data` changes raw data after a
   time T and proves no model input at or before T changes.

If `phase_clarity` is low for many real patients (for example, if the timestamps drift), set
`data.use_true_readings: false` and compare results; report which you used.

## Download

The same data is available from three places (one zip, about 630 to 660 MB):

- PhysioNet: https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip
- PhysioNet's AWS mirror: https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip
- figshare (Scientific Data copy, the zip is nested inside a bundle):
  https://ndownloader.figshare.com/files/56754167, landing page https://doi.org/10.6084/m9.figshare.29733005

```bash
python scripts/download_data.py --from-zip FILE.zip  # after downloading any of them in a browser
python scripts/download_data.py                      # or download in the terminal (--source aws / figshare)
```

Both put the CSV files under `data/raw/cgmacros/` and skip the meal photos. The loader searches
that folder recursively, so any sub-folder layout works. If you unzip by hand instead, unzip
inside `data/raw/cgmacros/`, and if you find another zip inside (figshare, PhysioNet "Download the
ZIP file"), unzip that one too.

The figshare page shows a CC BY 4.0 licence; we follow PhysioNet's stricter CC BY-NC-SA 4.0.

## License rules for the submission

- Never commit raw or processed data (`.gitignore` already excludes it).
- Cite the dataset and paper in the README, deck and video.
- Non-commercial use only. If you share derived data, share it under the same license.

## EDA checklist (do this on day 2, before trusting any metric)

Run `python -m glucotwin.prepare`. It prints a PASS/WARN report and saves it to
`data/processed/DATA_QUALITY.md` (per-patient details are in `patient_info.csv`). Check:

1. **Participant count**: 45 streams and 45 rows in `static_features.csv`.
2. **`bio.csv` id column**: the loader looks for a `subject`/`participant`/`id` column. If none
   exists it assumes rows follow sorted participant numbers and logs a warning. Confirm.
3. **HbA1c units**: the data dictionary says mmol/mol but values of 4.6 to 8.5 are percent.
   The loader converts only values above 20. Confirm the 15/16/14 group split reproduces.
4. **Sensor coverage and quality** per patient (`glucose_coverage`, `gaps_over_30min`,
   `phase_clarity`, `dexcom_vs_libre_mard_pct`); drop patients below about 60% coverage.
   Try `primary_cgm: libre` and `dexcom` in the config and keep the better one.
5. **Meal logging**: about 30 meals per patient expected. Plot a few days and look for large
   glucose rises without a logged meal (unlogged snacks).
6. **Amount Consumed**: check whether macros already reflect the eaten portion. If they are for
   the full plate, set `scale_by_amount_consumed: true`.
7. **Sentinel values**: LDL 800 and VLDL 400 are documented errors; the loader removes them.
8. **Label balance**: `spike_label_rate` in the summary. If it is below about 5%, try
   `spike_definition: either` (default) or lower `spike_rise_mgdl` to 40.

## Synthetic demo data

`scripts/make_synthetic_data.py` writes files in the same format (including the unnamed index
column and duplicate `Time (t)` headers) so the pipeline can be tested without the download.
It lives in `data/raw/synthetic`; never mix it with real data (the loader refuses duplicate
participant ids) and never report results obtained on it.

## Other open datasets worth mentioning (future work)

- AI-READI (741 participants with Dexcom G7 CGM and wearables in its first release) requires a
  controlled-access agreement, so it cannot be used within the challenge timeline.
- Synthea can generate synthetic FHIR patients for testing the EHR interface at scale.
- IFCT 2017 (ICMR-NIN) is the reference for Indian food composition.

## The locked test patients

`prepare` also writes `data/processed/splits.json`: about 20% of patients per metabolic group
(9 of 45 on CGMacros: 3 healthy, 3 prediabetes, 3 type 2) are locked as test patients, and the
other 36 are split into 5 cross-validation folds, stratified the same way. The split depends only
on the patient list, their HbA1c group and the fixed seed in `configs/config.yaml -> split`, so
every team member gets the same one. Do not change the seed after you have looked at results.
