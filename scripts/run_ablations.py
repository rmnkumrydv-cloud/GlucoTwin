"""Ablation study: what does each data stream add?  (development patients only)

    python scripts/run_ablations.py            # 5 runs, about 1 to 1.5 hours on a laptop
    python scripts/run_ablations.py --quick    # fast check that everything runs

Each run retrains with some feature groups removed and is scored by the same leave-patients-out
cross-validation as the main model. The locked test patients are never used. The table is written
to artifacts/ablations/ABLATIONS.md, ready for the "why fuse two streams" slide.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from glucotwin import train
from glucotwin.config import load_config

MEALS = ["carbs_", "protein_", "fat_", "fiber_", "kcal_", "cob_", "carb_rate", "time_since_meal",
         "last_meal", "fat_on_board"]
ACTIVITY = ["mets_", "active_", "act_kcal"]
HEART = ["hr_"]
EHR = ["ehr_"]

RUNS = {
    "CGM only": EHR + MEALS + ACTIVITY + HEART,
    "CGM + EHR": MEALS + ACTIVITY + HEART,
    "CGM + EHR + meals": ACTIVITY + HEART,
    "CGM + EHR + meals + activity": HEART,
    "all streams except EHR": EHR,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = load_config(args.config)
    out_root = base.path("artifacts_dir") / "ablations"
    results = {}
    for name, exclude in RUNS.items():
        slug = name.lower().replace(" + ", "_").replace(" ", "_")
        cfg = load_config(args.config, {"features": {"exclude_prefixes": exclude},
                                        "paths": {"artifacts_dir": str(out_root / slug)}})
        logging.info("=== ablation: %s", name)
        results[name] = train.run(cfg, quick=args.quick, tag=f"ablation: {name}")
    full = base.path("artifacts_dir") / "cv" / "metrics_cv.json"
    if full.exists():
        results["all streams (main model)"] = json.loads(full.read_text())
    hs = base["task"]["horizons_min"]
    lines = ["# Ablation: what each data stream adds (cross-validation, development patients)", "",
             "| Inputs | " + " | ".join(f"RMSE {h} min" for h in hs) + " | Spike AUROC |",
             "|---|" + "---|" * len(hs) + "---|"]
    for name, m in results.items():
        reg = {int(k): v for k, v in m["regression"].items()}
        au = m.get("classification", {}).get("spike", {}).get("auroc")
        lines.append(f"| {name} | " + " | ".join(f"{reg[h]['rmse']:.1f}" for h in hs)
                     + f" | {f'{au:.3f}' if au is not None else 'n/a'} |")
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "ABLATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
