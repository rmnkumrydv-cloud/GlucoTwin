"""Per-horizon ensemble of LightGBM and P-MRGF:  python -m glucotwin.ensemble

Needs artifacts/cv/oof_predictions.parquet (train.py) and artifacts/cv/oof_nn.parquet (nn.py).
Both come from the same development-patient folds.

* Development estimate: for each fold k, the blend weight for each horizon is chosen on the
  out-of-fold predictions of the OTHER folds only, then applied to fold k. No patient ever
  influences the weight used to score them.
* Weights for the final test: chosen once on all development out-of-fold predictions and saved
  to artifacts/ensemble_weights.json. ``python -m glucotwin.test`` applies them, unchanged, to the
  locked test patients.

If the network does not help, its weight is 0 and the ensemble equals LightGBM.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import load_config

log = logging.getLogger("glucotwin.ensemble")
GRID = np.round(np.linspace(0, 1, 21), 2)


def _best_weight(y, a, b, prob: bool = False) -> float:
    """Weight on b (the network) minimising RMSE (or the Brier score for probabilities)."""
    ok = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    errs = [np.mean((w * b[ok] + (1 - w) * a[ok] - y[ok]) ** 2) for w in GRID]
    return float(GRID[int(np.argmin(errs))])


def blend(frame: pd.DataFrame, weights: dict, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for h in horizons:
        w = weights[str(h)]
        out[f"pred_ens_{h}"] = w * frame[f"pred_nn_{h}"] + (1 - w) * frame[f"pred_{h}"]
    w = weights["spike"]
    out["p_spike_ens"] = w * frame["p_spike_nn"] + (1 - w) * frame["p_spike"]
    return out


def comparison_table(m: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for h in horizons:
        y = m["glucose"] + m[f"target_delta_{h}"]
        for name, col in (("LightGBM", f"pred_{h}"), ("P-MRGF (neural)", f"pred_nn_{h}"),
                          ("Ensemble", f"pred_ens_{h}")):
            r = ev.regression_metrics(y, m[col], m["glucose"])
            rows.append({"horizon": h, "model": name, "rmse": r["rmse"], "mae": r["mae"],
                         "r2_delta": r.get("r2_delta"), "clarke_AB_pct": r["clarke_AB_pct"]})
    spike = {name: ev.classification_metrics(m["label_spike"], m[col], 0.5).get("auroc")
             for name, col in (("LightGBM", "p_spike"), ("P-MRGF (neural)", "p_spike_nn"),
                               ("Ensemble", "p_spike_ens"))}
    return pd.DataFrame(rows), spike


def markdown(table: pd.DataFrame, spike: dict, title: str, note: str) -> str:
    piv = table.pivot(index="model", columns="horizon", values="rmse")
    header = "| Model | " + " | ".join(f"{h} min" for h in piv.columns) + " |"
    body = [f"| {name} | " + " | ".join(f"{v:.1f}" for v in r.values) + " |" for name, r in piv.iterrows()]
    return "\n".join([f"# {title}", "", note, "", "RMSE in mg/dL (lower is better):", "", header,
                      "|---|" + "---|" * len(piv.columns), *body, "",
                      "Spike AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in spike.items() if v is not None)]) + "\n"


def run(cfg) -> dict:
    art = cfg.path("artifacts_dir")
    gbm = pd.read_parquet(art / "cv" / "oof_predictions.parquet")
    nn = pd.read_parquet(art / "cv" / "oof_nn.parquet").drop(columns=["fold"], errors="ignore")
    m = gbm.merge(nn, on=["pid", "timestamp"], how="inner")
    horizons = [h for h in cfg["task"]["horizons_min"] if f"pred_nn_{h}" in m]

    # 1) honest development estimate: weights for fold k come from the other folds only
    parts = []
    for k in sorted(m["fold"].unique()):
        other, this = m[m["fold"] != k], m[m["fold"] == k]
        w = {str(h): _best_weight((other["glucose"] + other[f"target_delta_{h}"]).to_numpy(),
                                  other[f"pred_{h}"].to_numpy(), other[f"pred_nn_{h}"].to_numpy())
             for h in horizons}
        w["spike"] = _best_weight(other["label_spike"].to_numpy(float), other["p_spike"].to_numpy(),
                                  other["p_spike_nn"].to_numpy(), prob=True)
        parts.append(blend(this, w, horizons))
    m = m.join(pd.concat(parts))
    table, spike = comparison_table(m, horizons)

    # 2) weights for the locked test: all development out-of-fold predictions
    final = {str(h): _best_weight((m["glucose"] + m[f"target_delta_{h}"]).to_numpy(),
                                  m[f"pred_{h}"].to_numpy(), m[f"pred_nn_{h}"].to_numpy()) for h in horizons}
    final["spike"] = _best_weight(m["label_spike"].to_numpy(float), m["p_spike"].to_numpy(),
                                  m["p_spike_nn"].to_numpy(), prob=True)
    (art / "ensemble_weights.json").write_text(json.dumps(final, indent=2))
    (art / "cv" / "RESULTS_ENSEMBLE_CV.md").write_text(markdown(
        table, spike, "Ensemble comparison (cross-validation, development patients)",
        "Same unseen development patients for all three models. Neural-model weight used for the locked "
        "test, per horizon: " + ", ".join(f"{k}: {v:.2f}" for k, v in final.items()) + "."), encoding="utf-8")
    m.to_parquet(art / "cv" / "oof_ensemble.parquet", index=False)
    log.info("\n%s", table.pivot(index="model", columns="horizon", values="rmse").round(1))
    return {"table": table.to_dict(orient="records"), "spike_auroc": spike, "weights": final}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
