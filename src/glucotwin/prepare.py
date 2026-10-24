"""Step 2 of the pipeline:  python -m glucotwin.prepare [--raw data/raw/synthetic]

raw CGMacros  ->  dynamic stream (causal 5-min grid)  +  EHR records (FHIR-style JSON)
              ->  fused, model-ready dataset (data/processed/dataset.parquet)
              ->  locked, stratified patient split (data/processed/splits.json)
              ->  data-quality report (data/processed/DATA_QUALITY.md)
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import hashlib

from .config import load_config
from .data import load_cgmacros
from .ehr import build_all_ehr, static_feature_table
from .features import build_dataset, feature_columns
from .quality import is_synthetic, quality_report
from .splits import patient_groups, write_splits

log = logging.getLogger("glucotwin.prepare")


def run(cfg) -> dict:
    t0 = time.time()
    proc = cfg.path("processed_dir")
    proc.mkdir(parents=True, exist_ok=True)

    loaded = load_cgmacros(cfg)
    streams, meals, bio, info = loaded["streams"], loaded["meals"], loaded["bio"], loaded["info"]
    streams.to_parquet(proc / "wearable_stream.parquet", index=False)
    meals.to_parquet(proc / "meals.parquet", index=False)
    info.to_csv(proc / "patient_info.csv", index=False)

    for old in cfg.path("ehr_dir").glob("patient_*.json"):  # no stale records from another dataset
        old.unlink()
    records = build_all_ehr(bio, cfg, cfg.path("ehr_dir"))
    static = static_feature_table(records, cfg["ehr"]["use_synthetic_fields_as_features"])
    static.to_csv(proc / "static_features.csv")

    ds = build_dataset(streams, static, cfg)
    ds.to_parquet(proc / "dataset.parquet", index=False)

    summary = {
        "patients": int(ds["pid"].nunique()),
        "rows": int(len(ds)),
        "features": len(feature_columns(ds)),
        "spike_label_rate": round(float(ds["label_spike"].mean()), 4),
        "spike_positive_rows": int(ds["label_spike"].sum()),
        "hypo_positive_rows": int(ds["label_hypo"].sum()),
        "meals": int(len(meals)),
        "seconds": round(time.time() - t0, 1),
    }
    # Locked patient split: deterministic (fixed seed), so every team member gets the same one.
    groups = patient_groups(static.loc[sorted(ds["pid"].unique())])
    splits = write_splits(cfg, groups)
    # The fingerprint changes whenever the data, the split or any setting that shapes the dataset
    # (labels, features, preprocessing) changes, so models trained on an older dataset are refused.
    fp = hashlib.sha256(json.dumps([sorted(map(int, ds["pid"].unique())), len(ds), splits["test"],
                                    splits["folds"], cfg["data"], cfg["task"], cfg["features"]],
                                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    splits["fingerprint"] = fp
    synthetic = is_synthetic(cfg.path("raw_dir"))
    splits["synthetic"] = synthetic          # read by the dashboard, which has no raw folder online
    (proc / "splits.json").write_text(json.dumps(splits, indent=2))
    summary["fingerprint"] = fp
    summary["test_patients"] = splits["test"]
    (proc / "prepare_summary.json").write_text(json.dumps(summary, indent=2))

    synthetic = is_synthetic(cfg.path("raw_dir"))
    checks, md = quality_report(info, static, groups, summary, splits, synthetic)
    (proc / "DATA_QUALITY.md").write_text(md, encoding="utf-8")
    print("\nDATA QUALITY (also saved to data/processed/DATA_QUALITY.md)")
    for lvl, msg in checks:
        print(f"  [{lvl}] {msg}")
    stale = cfg.path("artifacts_dir") / "run_info.json"
    if stale.exists() and json.loads(stale.read_text()).get("fingerprint") != fp:
        print("\nNOTE: artifacts/ holds models trained on different data. Run training again.")
    log.info("prepared dataset: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="override paths.raw_dir, e.g. data/raw/synthetic")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, {"paths": {"raw_dir": args.raw}} if args.raw else None)
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
