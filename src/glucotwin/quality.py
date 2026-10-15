"""Data-quality gate, printed at the end of ``python -m glucotwin.prepare`` and saved to
data/processed/DATA_QUALITY.md. Read it before trusting any metric (it is "Gate 0" in the README).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_REAL = {"patients": 45, "groups": {"healthy": 15, "prediabetes": 16, "t2d": 14}}


def quality_report(info: pd.DataFrame, static: pd.DataFrame, groups: pd.Series, summary: dict,
                   splits: dict, synthetic: bool) -> tuple[list[tuple[str, str]], str]:
    checks: list[tuple[str, str]] = []

    def add(ok: bool, msg: str, warn_only: bool = True) -> None:
        checks.append(("PASS" if ok else ("WARN" if warn_only else "FAIL"), msg))

    n = int(info["pid"].nunique())
    counts = groups.value_counts().to_dict()
    gtxt = ", ".join(f"{g} {counts.get(g, 0)}" for g in ("healthy", "prediabetes", "t2d", "unknown") if counts.get(g))
    if synthetic:
        checks.append(("INFO", f"SYNTHETIC data: {n} patients ({gtxt}). Results on it are not scientific results."))
    else:
        add(n == EXPECTED_REAL["patients"], f"{n} participants found (CGMacros has 45)")
        exp = EXPECTED_REAL["groups"]
        add(all(counts.get(g, 0) == v for g, v in exp.items()),
            f"groups from HbA1c: {gtxt} (paper: healthy 15, prediabetes 16, t2d 14). A mismatch usually "
            "means a wrong bio.csv subject-id mapping or HbA1c units")
    add(counts.get("unknown", 0) == 0, f"{counts.get('unknown', 0)} patients without HbA1c (group unknown)")

    low_cov = info.loc[info["glucose_coverage"] < 0.6, "pid"].tolist()
    add(not low_cov, f"CGM coverage below 60%: {low_cov or 'none'}")
    if "phase_clarity" in info:
        unclear = info.loc[info["phase_clarity"].fillna(0) < 0.8, "pid"].tolist()
        add(not unclear, "original CGM readings recovered cleanly (phase clarity >= 0.8) "
            f"except: {unclear or 'none'}. If many fail, see docs/DATA.md")
    if "dexcom_vs_libre_mard_pct" in info:
        bad = info.loc[info["dexcom_vs_libre_mard_pct"] > 25, "pid"].tolist()
        add(not bad, f"Dexcom and Libre disagree by > 25% MARD: {bad or 'none'}")
    few_meals = info.loc[info["n_meals"] < 15, "pid"].tolist()
    add(not few_meals, f"fewer than 15 logged meals: {few_meals or 'none'}")
    dups = int(info.get("duplicate_timestamps", pd.Series([0])).sum())
    checks.append(("INFO", f"{dups} duplicate timestamps removed"))
    rate = summary.get("spike_label_rate", 0)
    add(0.05 <= rate <= 0.5, f"spike label rate {100 * rate:.1f}% (healthy range 5-50%)")
    add(bool(splits["test"]) and not set(splits["test"]) & set(splits["dev"]),
        f"locked test patients: {splits['test']} ({len(splits['test'])}); development patients: "
        f"{len(splits['dev'])} in {len(splits['folds'])} folds", warn_only=False)

    lines = ["# Data quality report", "", "| Status | Check |", "|---|---|"]
    lines += [f"| {lvl} | {msg} |" for lvl, msg in checks]
    lines += ["", "## Per-patient details", "", "```", info.to_string(index=False), "```"]
    return checks, "\n".join(lines) + "\n"


def is_synthetic(raw_dir: Path) -> bool:
    return any(Path(raw_dir).rglob("SYNTHETIC_DATA_README.txt"))
