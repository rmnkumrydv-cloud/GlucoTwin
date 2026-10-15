"""Patient-level data splits. This module is the main defence against data leakage.

Rules enforced here and checked by tests/test_leakage.py:

1. **Locked test patients.** About 20% of patients (stratified by metabolic group) are set aside
   once, deterministically, when the data is prepared. Tuning, cross-validation, the neural model,
   ensemble weights and alert thresholds never see them. Only ``python -m glucotwin.test`` reads
   them, and it should be run once, at the end.
2. **Development patients** are split into cross-validation folds, again by patient and stratified
   by group, so a person's data is never on both sides of a split.
3. **Purged calibration.** When a twin is personalized on a patient's first days, calibration rows
   whose prediction target lies inside the evaluation period are dropped (a purge gap as long as the
   longest horizon), so no future glucose value is ever used to calibrate.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_ORDER = ["healthy", "prediabetes", "t2d", "unknown"]


def patient_groups(static: pd.DataFrame) -> pd.Series:
    """pid -> metabolic group, from the EHR features (HbA1c-derived diagnosis)."""
    t2d = static.get("ehr_dx_t2d", pd.Series(0, index=static.index)).fillna(0) == 1
    pre = static.get("ehr_dx_prediabetes", pd.Series(0, index=static.index)).fillna(0) == 1
    has_a1c = static.get("ehr_hba1c", pd.Series(np.nan, index=static.index)).notna()
    grp = np.select([t2d, pre, has_a1c], ["t2d", "prediabetes", "healthy"], "unknown")
    return pd.Series(grp, index=static.index.astype(int), name="group")


def _stratified_folds(pids: list[int], groups: pd.Series, n_folds: int, rng: np.random.Generator) -> list[list[int]]:
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    start = 0
    for g in GROUP_ORDER:
        members = [p for p in pids if groups.get(p, "unknown") == g]
        for i, p in enumerate(rng.permutation(members)):
            folds[(start + i) % n_folds].append(int(p))
        start += len(members)  # continue round-robin so fold sizes stay balanced
    return [sorted(f) for f in folds]


def make_splits(groups: pd.Series, test_fraction: float, n_folds: int, seed: int) -> dict:
    """Deterministic, stratified patient split: locked test set + CV folds over the rest."""
    rng = np.random.default_rng(seed)
    test: list[int] = []
    for g in GROUP_ORDER:
        members = sorted(int(p) for p in groups[groups == g].index)
        if not members:
            continue
        n_test = int(round(len(members) * test_fraction))
        if len(members) >= 3:
            n_test = max(n_test, 1)
        test += [int(p) for p in rng.permutation(members)[:n_test]]
    if not test and len(groups) >= 4:  # tiny cohorts (tests, demos): still lock at least one patient
        biggest = groups.value_counts().index[0]
        test = [int(rng.permutation(sorted(int(p) for p in groups[groups == biggest].index))[0])]
    test = sorted(test)
    dev = sorted(int(p) for p in groups.index if int(p) not in set(test))
    n_folds = max(2, min(n_folds, len(dev)))
    folds = _stratified_folds(dev, groups, n_folds, rng)
    return {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "test_fraction": test_fraction,
        "test": test, "dev": dev, "folds": folds,
        "groups": {str(int(p)): g for p, g in groups.items()},
    }


def splits_path(cfg) -> Path:
    return cfg.path("processed_dir") / "splits.json"


def write_splits(cfg, groups: pd.Series) -> dict:
    s = cfg["split"]
    splits = make_splits(groups, float(s["test_fraction"]), int(s["n_folds"]), int(s["seed"]))
    splits_path(cfg).write_text(json.dumps(splits, indent=2))
    return splits


def load_splits(cfg) -> dict:
    p = splits_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python -m glucotwin.prepare` first.")
    s = json.loads(p.read_text())
    overlap = set(s["test"]) & set(s["dev"])
    if overlap:  # pragma: no cover - would only happen if the file was edited by hand
        raise ValueError(f"Patients {sorted(overlap)} are in both the test and development sets.")
    return s


def dev_only(ds: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Development rows only, with a hard check that no test patient slipped through."""
    out = ds[ds["pid"].isin(splits["dev"])]
    assert not set(out["pid"].unique()) & set(splits["test"]), "test patient in development data"
    return out


def purge_minutes(cfg) -> int:
    """Longest look-ahead of any target: no calibration row may 'see' past the cutoff."""
    t = cfg["task"]
    return int(max(max(t["horizons_min"]), t["spike_window_min"], t["hypo_window_min"]))


def calibration_split(d: pd.DataFrame, cal_days: float, purge_min: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split one patient's rows into (calibration, evaluation, cutoff) without overlap.

    Calibration rows end ``purge_min`` minutes before the cutoff, so every calibration target
    (glucose up to ``purge_min`` ahead) is observed before the evaluation period begins.
    """
    d = d.sort_values("timestamp")
    cutoff = d["timestamp"].min() + pd.Timedelta(days=cal_days)
    calib = d[d["timestamp"] + pd.Timedelta(minutes=purge_min) <= cutoff]
    later = d[d["timestamp"] >= cutoff]
    return calib, later, cutoff
