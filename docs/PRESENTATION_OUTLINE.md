# Presentation outline (export as docs/presentation.pdf)

Aim for 14 slides. One idea per slide, a real chart or screenshot on most of them, and numbers
taken only from `artifacts/test/RESULTS_TEST.md` produced on the real CGMacros data.

1. **Title.** GlucoTwin: a digital twin that warns of post-meal glucose spikes 2 hours ahead.
   Team name, college, members, challenge name.
2. **The problem in India.** 101 million people with diabetes and 136 million with prediabetes
   (ICMR-INDIAB 2023). Carbohydrate-heavy meals, and the same meal affects people differently.
   Doctors see HbA1c every 3 months, not what happens after lunch.
3. **Our idea.** A virtual copy of one patient's glucose metabolism that predicts the next 2 hours
   and lets the doctor test advice before giving it. One sentence, one picture of the forecast cone.
4. **Data.** CGMacros: 45 people, 3 groups, 10 days, two CGMs, Fitbit, logged meals. Static vs
   dynamic stream table. Why open data only; license; synthetic EHR fields clearly flagged.
5. **Architecture.** `docs/architecture.png`. Walk left to right in 40 seconds.
6. **EHR as FHIR.** Screenshot of one bundle: LOINC, SNOMED and ICD-10 codes. Link to ABDM.
7. **Fusion and features.** Carbs-on-board curve from the gamma absorption model, activity and
   heart rate, EHR features. Leakage test explained in one line.
8. **Twin engine.** LightGBM per horizon, uncertainty band with conformal calibration, spike and
   hypo classifiers, monotone constraints, plan dropout. Keep it visual.
9. **How we validated.** The patient-flow diagram from the README: 9 locked test patients, 36
   development patients in 5-fold patient CV, purge gap for personalization. Baselines; metrics
   chosen and why (Clarke grid is the clinical standard for glucose accuracy). One line on the
   CGMacros interpolation trap and how the leakage tests prove it is handled.
10. **Results: forecasting.** RMSE by horizon vs baselines (`figures/rmse_by_horizon.png`) and
    the 60-minute Clarke grid. State zone A+B percentage.
11. **Results: early warning.** Spike AUROC, share of spikes caught in advance, median warning time,
    false alerts per day. Example forecast figure.
12. **Personalization and explanations.** Personalization gain chart; plain-language drivers.
13. **Live demo screenshots.** Forecast tab, what-if (rice vs millet thali plus a walk), clinic
    overview. (The live demo itself happens in Phase 2.)
14. **Limitations and road to real use.** Small US cohort, self-reported meals, approximate
    Indian meal macros, need for Indian validation study, CDSCO SaMD route, DPDP Act consent,
    ABDM integration. Next steps and thank you.

Backup slides: ablation table (without EHR, without activity, without meals), tuning details,
per-group results (healthy vs prediabetes vs T2D), failure cases.
