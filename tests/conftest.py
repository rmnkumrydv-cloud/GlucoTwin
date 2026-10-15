"""Shared test setup: a tiny synthetic cohort run through the WHOLE pipeline once per test session
(prepare -> train -> neural model -> ensemble -> test), in a temporary folder."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from glucotwin.config import load_config  # noqa: E402


def make_synthetic(out: Path, patients: int, days: int, seed: int) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(out),
                    "--patients", str(patients), "--days", str(days), "--seed", str(seed)],
                   check=True, capture_output=True)


def toy_stream(n_days: int = 2, seed: int = 0) -> pd.DataFrame:
    """A regular 5-minute stream for one patient (already in processed format)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-03-01", periods=n_days * 288, freq="5min")
    s = pd.DataFrame({"pid": 1, "timestamp": ts, "glucose": 100 + rng.normal(0, 5, len(ts)).cumsum() * 0.1,
                      "glucose_age_min": 0.0, "hr": 70 + rng.normal(0, 3, len(ts)),
                      "mets": 1.2 + rng.random(len(ts)), "act_kcal": 1.3, "carbs": 0.0, "protein": 0.0,
                      "fat": 0.0, "fiber": 0.0, "meal_kcal": 0.0})
    for i in (100, 160, 250, 390, 450):
        s.loc[i, ["carbs", "protein", "fat", "fiber", "meal_kcal"]] = [60, 20, 10, 5, 410]
    return s


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("glucotwin")
    raw = tmp / "raw"
    make_synthetic(raw, patients=10, days=5, seed=3)
    cfg = load_config(overrides={"paths": {"raw_dir": str(raw), "processed_dir": str(tmp / "proc"),
                                           "ehr_dir": str(tmp / "ehr"), "artifacts_dir": str(tmp / "art")}})
    from glucotwin import prepare, test, train
    summary = prepare.run(cfg)
    cv_metrics = train.run(cfg, quick=True)
    nn_ran = False
    try:
        import torch  # noqa: F401
        from glucotwin import ensemble, nn
        nn.run(cfg, quick=True)
        ensemble.run(cfg)
        nn_ran = True
    except ImportError:
        pass
    test_metrics = test.run(cfg)
    return {"cfg": cfg, "summary": summary, "cv": cv_metrics, "test": test_metrics, "nn_ran": nn_ran}
