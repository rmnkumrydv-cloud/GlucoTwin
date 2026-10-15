"""End-to-end tests:  pytest -q

The session fixture in conftest.py runs the WHOLE pipeline on a tiny synthetic cohort
(prepare -> train -> neural model -> ensemble -> test), then these tests check the outputs,
the what-if simulator, the API and the dashboard. About 2 to 4 minutes on a laptop.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from glucotwin.ehr import build_ehr_record, ehr_to_features
from glucotwin.evaluate import clarke_zone


def test_clarke_zones():
    assert clarke_zone(100, 110) == "A"
    assert clarke_zone(60, 65) == "A"
    assert clarke_zone(250, 60) == "E"
    assert clarke_zone(50, 200) == "E"
    assert clarke_zone(100, 150) == "B"


def test_ehr_round_trip_and_synthetic_flag():
    row = pd.Series({"pid": 7, "age": 50, "sex": "female", "hba1c": 7.1, "fasting_glucose": 140,
                     "fasting_insulin": 12, "bmi": 31, "weight_lb": 180, "height_in": 64,
                     "triglycerides": 150, "hdl": 45})
    rec = build_ehr_record(row, np.random.default_rng(0), add_synthetic=True)
    f = ehr_to_features(rec, use_synthetic=False)
    assert f["hba1c"] == 7.1 and f["dx_t2d"] == 1.0 and f["age"] == 50
    assert "med_metformin" not in f and "tcf7l2_risk_alleles" not in f
    assert ehr_to_features(rec, use_synthetic=True).get("med_metformin") == 1.0
    assert abs(f["homa_ir"] - 140 * 12 / 405) < 1e-6


def test_prepare_writes_split_and_quality_report(pipeline):
    proc = pipeline["cfg"].path("processed_dir")
    assert pipeline["summary"]["patients"] == 10
    assert (proc / "DATA_QUALITY.md").exists()
    s = json.loads((proc / "splits.json").read_text())
    assert s["fingerprint"] == pipeline["summary"]["fingerprint"]


def test_model_beats_persistence_in_cv_and_on_test(pipeline):
    for name in ("cv", "test"):
        m = pipeline[name]
        for h, r in m["regression"].items():
            base = m["baselines"][h]["Persistence (no change)"]["rmse"]
            assert r["rmse"] < base, f"{name}: model should beat persistence at {h} min"
    art = pipeline["cfg"].path("artifacts_dir")
    assert (art / "cv" / "RESULTS_CV.md").exists() and (art / "test" / "RESULTS_TEST.md").exists()
    assert (art / "test_runs.csv").exists()


def test_neural_model_and_ensemble_outputs(pipeline):
    if not pipeline["nn_ran"]:
        pytest.skip("PyTorch not installed (optional neural engine)")
    art = pipeline["cfg"].path("artifacts_dir")
    weights = json.loads((art / "ensemble_weights.json").read_text())
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    assert "ensemble" in pipeline["test"]
    preds = pd.read_parquet(art / "test" / "test_predictions.parquet")
    assert {"pred_nn_60", "pred_ens_60"} <= set(preds.columns)


def test_test_refuses_models_trained_on_other_data(pipeline):
    from glucotwin import test
    cfg = pipeline["cfg"]
    info_p = cfg.path("artifacts_dir") / "run_info.json"
    info = json.loads(info_p.read_text())
    info_p.write_text(json.dumps({**info, "fingerprint": "different"}))
    try:
        try:
            test.run(cfg)
            raise AssertionError("test.run should refuse mismatched models")
        except RuntimeError:
            pass
    finally:
        info_p.write_text(json.dumps(info))


def test_whatif_is_physiologically_monotone(pipeline):
    from glucotwin.service import TwinService
    svc = TwinService(pipeline["cfg"])
    pid = int(svc.dataset["pid"].iloc[0])
    at = svc.patient_meals(pid)["timestamp"].iloc[3]
    base = {"protein": 15, "fat": 10, "fiber": 4}
    res = svc.whatif(pid, at, [
        {"label": "small", "meal": {**base, "carbs": 30, "meal_kcal": 300}},
        {"label": "large", "meal": {**base, "carbs": 120, "meal_kcal": 660}},
        {"label": "large+walk", "meal": {**base, "carbs": 120, "meal_kcal": 660}, "walk_minutes": 30},
    ])
    small, large, walk = (np.array(r["forecast"]) for r in res)
    assert np.all(large >= small - 1e-6)
    assert np.all(walk <= large + 1e-6)
    risk = [r["p_spike"] for r in res]
    if not np.isnan(risk).any():
        assert risk[1] >= risk[0] - 1e-9 and risk[2] <= risk[1] + 1e-9


def test_api_and_dashboard(pipeline, monkeypatch):
    cfg = pipeline["cfg"]
    import glucotwin.service as service
    service.get_service.cache_clear()
    monkeypatch.setattr(service, "load_config", lambda *a, **k: cfg)
    from fastapi.testclient import TestClient

    from app.api import app
    c = TestClient(app)
    pts = c.get("/patients").json()
    pid = pts[0]["pid"]
    assert c.get(f"/patients/{pid}/ehr").status_code == 200
    fc = c.get(f"/patients/{pid}/forecast").json()
    assert len(fc["forecast"]) == 1 + len(cfg["task"]["horizons_min"])
    r = c.post(f"/patients/{pid}/whatif", json={"at": fc["at"], "scenarios": [
        {"label": "idli", "indian_meal": {"Idli": 2, "Sambar": 1}}, {"label": "walk", "walk_minutes": 20}]})
    assert r.status_code == 200 and len(r.json()) == 2

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception


def test_alert_threshold_modes():
    from glucotwin.models import _choose_threshold
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(float)
    p = np.clip(0.3 * y + rng.normal(0.3, 0.15, 5000), 0, 1)
    thr = _choose_threshold(y, p, "spec90")
    assert abs(((p < thr) & (y == 0)).sum() / (y == 0).sum() - 0.90) < 0.01      # 90% specificity
    assert _choose_threshold(y, p, "spec95") > thr > _choose_threshold(y, p, "spec85")
    be = _choose_threshold(y, p, "breakeven")
    assert abs((p >= be).mean() - y.mean()) < 0.01                                 # alerts as often as events
    assert 0 <= _choose_threshold(y, p, "f1") <= 1


def test_personal_gain_calibration(pipeline):
    """The 'gain' personalization runs on an unseen patient and stays within physiological bounds."""
    from glucotwin.features import hide_plan
    from glucotwin.models import GlucoTwinModel, _PersonalGain
    from glucotwin.splits import calibration_split, purge_minutes
    cfg = pipeline["cfg"]
    art = cfg.path("artifacts_dir")
    splits = json.loads((cfg.path("processed_dir") / "splits.json").read_text())
    model = GlucoTwinModel.load(art / "models" / "fold_0.joblib")
    model.cfg["personalization"]["method"] = "gain"
    pid = splits["folds"][0][0]
    ds = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    d = ds[ds["pid"] == pid]
    calib, later, cutoff = calibration_split(d, 3, purge_minutes(cfg))
    cal = model.calibrate_patient(pid, calib, cutoff=cutoff)
    for c in cal.values():
        assert isinstance(c, _PersonalGain) and -0.5 <= c.c <= 1.0
    out = model.predict(hide_plan(later[model.features]), pid=pid, personalized=True)
    assert all(np.isfinite(v["delta"]).all() for v in out.values())
