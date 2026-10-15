"""Final step:  python -m glucotwin.test

Scores the LOCKED TEST PATIENTS, once, with the models chosen on development patients:

* LightGBM development model (models/model_dev.joblib) and the three baselines,
* the twin personalized on each test patient's own first days (purged, scored on later days),
* if present, the neural development model (models/nn_dev.pt) and the ensemble with the weights
  fixed on development patients (artifacts/ensemble_weights.json).

Nothing here is fitted on test outcomes except the per-patient twin calibration, which uses only
that patient's earlier days, exactly as it would in a clinic. Alert thresholds, ensemble weights and
hyperparameters all come from development patients.

Run it when every decision is final. Each run is logged in artifacts/test_runs.csv; if you change
the model after looking at these numbers, the test set is no longer an unbiased estimate (say so).
Outputs: artifacts/test/RESULTS_TEST.md, metrics_test.json, test_predictions.parquet, figures/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging

import pandas as pd

from . import evaluate as ev
from .config import load_config
from .models import GlucoTwinModel, RidgeBaseline
from .splits import dev_only, load_splits
from .train import keep_columns, personalize_frame, predict_frame, summarize, write_results_md

log = logging.getLogger("glucotwin.test")


def run(cfg) -> dict:
    art = cfg.path("artifacts_dir")
    splits = load_splits(cfg)
    info_p = art / "run_info.json"
    if not info_p.exists():
        raise FileNotFoundError("No trained model found. Run `python -m glucotwin.train` first.")
    info = json.loads(info_p.read_text())
    if info.get("fingerprint") != splits.get("fingerprint"):
        raise RuntimeError("The models in artifacts/ were trained on a different dataset or split. "
                           "Run `python -m glucotwin.train` again before testing.")
    horizons = cfg["task"]["horizons_min"]
    ds = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    te = ds[ds["pid"].isin(splits["test"])]
    dev = dev_only(ds, splits)
    model = GlucoTwinModel.load(art / "models" / "model_dev.joblib")
    leaked = set(model.train_pids) & set(splits["test"])
    if leaked:
        raise RuntimeError(f"model_dev was trained on test patients {sorted(leaked)}")
    feats = model.features
    ridge = RidgeBaseline(feats).fit(dev, horizons)          # baseline fitted on development patients only

    frame = pd.concat([keep_columns(te, cfg, splits["groups"]), predict_frame(model, ridge, te, feats, horizons),
                       personalize_frame(model, te, feats, cfg)], axis=1)
    out_dir = art / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize(frame, cfg, out_dir / "figures", "locked test patients")
    ci = ev.bootstrap_ci(frame, horizons)
    metrics["bootstrap_ci_95"] = ci
    model.save(art / "models" / "model_dev.joblib")         # now holds each test patient's calibration

    extra_md = ""
    try:
        from . import nn as nn_mod
        nn_pred = nn_mod.predict_test(cfg)
    except ImportError:
        nn_pred = None
        log.info("PyTorch not installed: skipping the neural model")
    if nn_pred is not None:
        from .ensemble import blend, comparison_table, markdown
        frame = frame.reset_index(drop=True).merge(nn_pred, on=["pid", "timestamp"], how="left")
        w_path = art / "ensemble_weights.json"
        if w_path.exists():
            weights = json.loads(w_path.read_text())
            frame = frame.join(blend(frame, weights, [h for h in horizons if f"pred_nn_{h}" in frame]))
            table, spike = comparison_table(frame, [h for h in horizons if f"pred_nn_{h}" in frame])
            metrics["ensemble"] = {"table": table.to_dict(orient="records"), "spike_auroc": spike,
                                   "weights": weights}
            extra_md = "\n" + markdown(table, spike, "Neural model and ensemble on the locked test patients",
                                       "Ensemble weights were fixed on development patients before this test.")
    frame.to_parquet(out_dir / "test_predictions.parquet", index=False)
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_results_md(
        metrics, cfg, out_dir / "RESULTS_TEST.md",
        "GlucoTwin results on the locked test patients",
        f"{len(splits['test'])} patients ({', '.join(f'{p:03d}' for p in splits['test'])}) were set aside "
        "before any training or tuning, stratified by metabolic group. Every model, threshold and weight "
        "below was chosen on the development patients only. These are the numbers to report. With so few "
        "test patients the intervals are wide; quote them.", ci=ci)
    if extra_md:
        with open(out_dir / "RESULTS_TEST.md", "a", encoding="utf-8") as fh:
            fh.write(extra_md)

    log_p = art / "test_runs.csv"
    row = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "fingerprint": splits.get("fingerprint"),
           **{f"test_rmse_{h}": round(metrics["regression"][h]["rmse"], 2) for h in horizons},
           "test_spike_auroc": round(metrics.get("classification", {}).get("spike", {}).get("auroc", float("nan")), 4)}
    previous = pd.read_csv(log_p) if log_p.exists() else pd.DataFrame()
    pd.concat([previous, pd.DataFrame([row])]).to_csv(log_p, index=False)
    n_runs = int((pd.concat([previous, pd.DataFrame([row])])["fingerprint"] == row["fingerprint"]).sum())
    if n_runs > 1:
        log.warning("The test set has now been scored %d times on this split. If you changed the model "
                    "because of earlier test results, report that the test estimate is optimistic.", n_runs)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run(load_config(args.config))
    ci = m.get("bootstrap_ci_95", {}).get("rmse", {})
    print(json.dumps({"test_rmse": {h: round(r["rmse"], 1) for h, r in m["regression"].items()},
                      "test_rmse_95ci": {h: [round(a, 1), round(b, 1)] for h, (a, b) in ci.items()},
                      "test_spike_auroc": m.get("classification", {}).get("spike", {}).get("auroc")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
