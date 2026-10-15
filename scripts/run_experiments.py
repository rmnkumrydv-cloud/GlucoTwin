"""Compare candidate settings in one go (development patients only; the test set is never used).

    python scripts/run_experiments.py            # about 4 x the time of one training run
    python scripts/run_experiments.py --quick    # fast check that everything runs

Each experiment trains with leave-patients-out cross-validation on the 36 development patients and
writes to artifacts/experiments/<name>/. Your main artifacts/ folder (used by the dashboard and by
`python -m glucotwin.test`) is not touched. The comparison table is printed and saved to
artifacts/experiments/EXPERIMENTS.md. Choose the winner, put its settings in configs/config.yaml,
and run `python -m glucotwin.train` once more.

Edit EXPERIMENTS below to try other settings (same keys as configs/config.yaml).
"""
from __future__ import annotations

import argparse
import json
import logging
import re

from glucotwin import prepare, train
from glucotwin.config import load_config

EXPERIMENTS = {
    "A current (spec90 alerts, residual personalization)": {},
    "B personal sensitivity (gain)": {"personalization": {"method": "gain"}},
    "C bias-only personalization": {"personalization": {"method": "bias"}},
    "D spike = glucose reaches 180 only": {"task": {"spike_definition": "absolute"}},
}
DATASET_KEYS = ("data", "task", "features", "ehr")   # changing these needs its own prepared dataset


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def deep_merge(a: dict, b: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    for k, v in b.items():
        out[k] = deep_merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    base = load_config(args.config)
    root = base.path("artifacts_dir") / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    main_splits = json.loads((base.path("processed_dir") / "splits.json").read_text())
    main_test, n_dev = main_splits["test"], len(main_splits["dev"])

    rows = []
    for name, over in EXPERIMENTS.items():
        s = slug(name)
        paths = {"artifacts_dir": str(root / s)}
        if any(k in over for k in DATASET_KEYS):          # labels/features change: prepare a copy
            paths["processed_dir"] = str(base.path("processed_dir") / "experiments" / s)
            paths["ehr_dir"] = str(base.path("processed_dir") / "experiments" / s / "ehr")
        cfg = load_config(args.config, deep_merge(over, {"paths": paths}))
        print(f"\n=== {name}", flush=True)
        if "processed_dir" in paths:
            print("  preparing a dataset with these settings ...", flush=True)
            prepare.run(cfg)
            test = json.loads((cfg.path("processed_dir") / "splits.json").read_text())["test"]
            assert test == main_test, "experiment dataset has different test patients"
        m = train.run(cfg, quick=args.quick, tag=f"experiment: {name}")
        reg, pers = m["regression"], m.get("personalization", {})
        c = m.get("classification", {}).get("spike", {})
        e = m.get("event_level_spike", {})
        rows.append({
            "Experiment": name,
            "RMSE 30": reg[30]["rmse"], "RMSE 60": reg[60]["rmse"], "RMSE 120": reg[120]["rmse"],
            "Personal. gain 60": pers.get(60, {}).get("population_rmse", 0) - pers.get(60, {}).get("personalized_rmse", 0),
            "Personal. gain 120": pers.get(120, {}).get("population_rmse", 0) - pers.get(120, {}).get("personalized_rmse", 0),
            "Spike AUROC": c.get("auroc", float("nan")), "Precision": c.get("precision", float("nan")),
            "Recall": c.get("recall", float("nan")),
            "Spikes caught %": 100 * e.get("event_recall", float("nan")),
            "False alerts/day": e.get("false_alerts_per_day", float("nan")),
            "Alert on %": e.get("alert_on_time_pct", float("nan")),
        })

    cols = list(rows[0])
    fmt = {k: (".1f" if k.startswith(("RMSE", "Spikes", "Alert")) else ".2f") for k in cols[1:]}
    fmt.update({"Spike AUROC": ".3f"})
    lines = [f"# Experiments (cross-validation, {n_dev} development patients)", "",
             "RMSE in mg/dL (lower is better). Personalization gain: how much the personalized twin lowers RMSE "
             "(positive = it helps). Spike AUROC/precision/recall depend on the spike definition, so compare D with "
             "the others using the event columns: share of crossings above 180 mg/dL warned in advance, false "
             "alerts per patient-day and the share of time an alert is showing.", "",
             "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([r["Experiment"]] + [format(r[k], fmt[k]) for k in cols[1:]]) + " |")
    text = "\n".join(lines) + "\n"
    (root / "EXPERIMENTS.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Saved {root / 'EXPERIMENTS.md'}")


if __name__ == "__main__":
    main()
