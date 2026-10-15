# Video plan (minimum 20 minutes, upload as unlisted YouTube)

Record the screen with OBS Studio (free) at 1080p. Use a headset microphone. Record in sections
so a mistake costs one section, not the whole video. Show faces briefly at the start and end.

| Time | Section | Speaker | What is on screen |
|---|---|---|---|
| 0:00 to 1:30 | Team introduction, college, challenge, one-line idea | Lead | Title slide, faces |
| 1:30 to 4:00 | Problem: diabetes burden in India, why post-meal spikes matter, what doctors lack today | Clinical lead | Slides 2 and 3 |
| 4:00 to 6:30 | Data: CGMacros, the two streams, license, synthetic EHR fields and why they are not used for learning | Data lead | Slide 4, a raw CSV, one FHIR bundle |
| 6:30 to 9:00 | Architecture walkthrough, left to right | Lead | Slide 5 |
| 9:00 to 12:30 | Code tour: `prepare`, `features.py` (carbs on board, leakage test), `models.py` (horizons, monotone constraints, plan dropout, conformal band), `train.py` (patient-wise CV) | ML lead | Editor + terminal running `pytest -q` |
| 12:30 to 15:30 | Results: RMSE vs baselines, Clarke grid, spike AUROC, warning time, false alerts, personalization gain | ML lead | Slides 10 to 12, `artifacts/test/RESULTS_TEST.md` |
| 15:30 to 20:30 | Live dashboard demo: pick a type 2 patient, replay a lunch, show the alert and drivers, reveal what actually happened; What-if: white rice vs foxtail millet thali, add a 15-minute walk; health record tab; clinic overview; model evidence | App lead | Streamlit |
| 20:30 to 22:00 | API demo in `/docs` (forecast and what-if requests) | App lead | Browser |
| 22:00 to 24:00 | Limitations, path to clinical use (validation study, CDSCO, DPDP, ABDM), next steps, thanks | Lead | Slide 14 |

Tips

- Rehearse the demo with a fixed patient, day and time so it never surprises you.
- Say out loud that every number comes from patients the model never saw.
- Keep "synthetic" and "real" clearly separated whenever you mention data.
- End with the repository link on screen.
