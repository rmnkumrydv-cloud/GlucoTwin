"""Data-leakage tests. Each test is one way a glucose model can look better than it really is.

Run with:  pytest -q tests/test_leakage.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic, toy_stream

from glucotwin.config import load_config
from glucotwin.features import NON_FEATURE_PREFIXES, feature_columns, hide_plan, patient_features
from glucotwin.models import GlucoTwinModel
from glucotwin.splits import calibration_split, make_splits, purge_minutes


# ---------------------------------------------------------------- 1. time: no future in the inputs
def test_features_never_use_the_future():
    """Changing anything after time t must not change any (non-plan) feature at or before t."""
    cfg = load_config()
    s = toy_stream()
    t = 300
    f1 = patient_features(s, cfg, with_targets=False)
    s2 = s.copy()
    s2.loc[t + 1:, "glucose"] += 80
    s2.loc[t + 1:, ["hr", "mets"]] *= 2
    s2.loc[t + 5, "carbs"] = 150
    f2 = patient_features(s2, cfg, with_targets=False)
    cols = [c for c in feature_columns(f1) if not c.startswith("plan_")]
    pd.testing.assert_frame_equal(f1.loc[:t, cols], f2.loc[:t, cols])


def test_raw_minute_data_after_t_cannot_change_the_stream_before_t(tmp_path):
    """CGMacros interpolated CGM to 1 minute; our 5-minute grid must still be strictly causal."""
    make_synthetic(tmp_path, patients=1, days=2, seed=5)
    from glucotwin.data import build_stream, load_participant
    cfg = load_config()
    raw = load_participant(next(tmp_path.rglob("CGMacros-*.csv")))
    s1, _, _ = build_stream(raw, cfg, 1)
    T = s1["timestamp"].iloc[300]
    raw2 = raw.copy()
    fut = raw2["timestamp"] > T
    raw2.loc[fut, ["dexcom", "libre"]] += 60
    raw2.loc[fut, ["hr", "mets"]] += 25
    raw2.loc[fut & (raw2.index % 97 == 0), "carbs"] = 80
    s2, _, _ = build_stream(raw2, cfg, 1)
    cols = ["glucose", "glucose_age_min", "hr", "mets", "act_kcal", "carbs"]
    pd.testing.assert_frame_equal(s1.loc[s1["timestamp"] <= T, cols].reset_index(drop=True),
                                  s2.loc[s2["timestamp"] <= T, cols].reset_index(drop=True))


def test_original_cgm_readings_are_recovered_from_interpolated_series():
    from glucotwin.data import true_reading_mask
    rng = np.random.default_rng(1)
    truth = 120 + np.cumsum(rng.normal(0, 3, 200))
    t_read = np.arange(3, 3 + 5 * 200, 5)[:200]             # real readings at phase 3, every 5 min
    series = np.interp(np.arange(0, t_read[-1] + 1), t_read, truth)
    series[:3] = np.nan
    mask, _ = true_reading_mask(series, 5)
    assert set(np.flatnonzero(mask) % 5) == {3}


def test_targets_and_identifiers_are_never_features():
    cfg = load_config()
    df = patient_features(toy_stream(), cfg, with_targets=True)
    feats = feature_columns(df)
    assert not [f for f in feats if f.startswith(NON_FEATURE_PREFIXES)]
    assert not {"pid", "timestamp", "day", "minute_of_day"} & set(feats)


def test_honest_mode_hides_every_future_plan_input():
    cfg = load_config()
    df = patient_features(toy_stream(), cfg, with_targets=False)
    X = hide_plan(df[feature_columns(df)])
    assert X["plan_carbs"].isna().all() and X["plan_mets_mean"].isna().all() and (X["plan_known"] == 0).all()


# ---------------------------------------------------------------- 2. people: no patient on both sides
def _groups(n_per_group=(15, 16, 14)) -> pd.Series:
    names = ["healthy", "prediabetes", "t2d"]
    pids, grp = [], []
    for name, n in zip(names, n_per_group):
        for _ in range(n):
            pids.append(len(pids) + 1)
            grp.append(name)
    return pd.Series(grp, index=pids)


def test_split_is_patient_level_stratified_and_deterministic():
    g = _groups()
    a, b = make_splits(g, 0.2, 5, seed=2026), make_splits(g, 0.2, 5, seed=2026)
    assert a["test"] == b["test"] and a["folds"] == b["folds"]          # same split for every team member
    assert not set(a["test"]) & set(a["dev"])
    assert sorted(p for f in a["folds"] for p in f) == a["dev"]           # folds partition the dev patients
    assert len(set(a["test"]) | set(a["dev"])) == 45
    test_groups = g.loc[a["test"]].value_counts().to_dict()
    assert test_groups == {"healthy": 3, "prediabetes": 3, "t2d": 3}     # stratified by group


def test_personalization_calibration_is_purged():
    """No calibration target (up to the longest horizon ahead) may fall in the evaluation days."""
    cfg = load_config()
    d = patient_features(toy_stream(n_days=5), cfg, with_targets=True)
    purge = purge_minutes(cfg)
    calib, later, cutoff = calibration_split(d, 3, purge)
    last_target_time = calib["timestamp"].max() + pd.Timedelta(minutes=purge)
    assert last_target_time <= cutoff <= later["timestamp"].min()
    assert purge >= max(cfg["task"]["horizons_min"])


def test_a_model_cannot_be_calibrated_on_a_patient_it_was_trained_on():
    m = GlucoTwinModel(load_config(), ["glucose"], train_pids=[7])
    with pytest.raises(ValueError):
        m.calibrate_patient(7, pd.DataFrame())


# ---------------------------------------------------------------- 3. the whole pipeline
def test_no_model_ever_trained_on_a_locked_test_patient(pipeline):
    cfg = pipeline["cfg"]
    art = cfg.path("artifacts_dir")
    splits = json.loads((cfg.path("processed_dir") / "splits.json").read_text())
    test = set(splits["test"])
    assert test, "the pipeline must lock at least one test patient"
    dev_model = GlucoTwinModel.load(art / "models" / "model_dev.joblib")
    assert not set(dev_model.train_pids) & test
    for k, held in enumerate(splits["folds"]):
        fold = GlucoTwinModel.load(art / "models" / f"fold_{k}.joblib")
        assert not set(fold.train_pids) & test
        assert not set(fold.train_pids) & set(held)
    oof = pd.read_parquet(art / "cv" / "oof_predictions.parquet")
    assert not set(oof["pid"]) & test, "cross-validation must only score development patients"
    if pipeline["nn_ran"]:
        import torch
        ck = torch.load(art / "models" / "nn_dev.pt", weights_only=False)
        assert not set(ck["train_pids"]) & test


def test_dashboard_serves_every_patient_with_a_model_that_never_saw_them(pipeline):
    from glucotwin.service import TwinService
    svc = TwinService(pipeline["cfg"])
    for pid in svc.dataset["pid"].unique():
        assert int(pid) not in set(svc.model_for(int(pid)).train_pids), f"patient {pid} served by a model that saw them"


def test_personalization_is_never_applied_to_the_calibration_days(pipeline):
    from glucotwin.service import TwinService
    svc = TwinService(pipeline["cfg"])
    pid = int(json.loads((pipeline["cfg"].path("processed_dir") / "splits.json").read_text())["test"][0])
    model = svc.model_for(pid)
    assert pid in model.cal_cutoff
    cutoff = model.cal_cutoff[pid]
    early = svc.forecast(pid, cutoff - pd.Timedelta(hours=6), personalized=True)
    assert early["personalized"] is False
