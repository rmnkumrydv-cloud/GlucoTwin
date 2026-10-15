"""Optional step:  python -m glucotwin.tune

Searches LightGBM hyperparameters with Optuna, scoring each trial by RMSE on held-out
DEVELOPMENT patients (grouped, stratified CV, honest mode). The locked test patients are never
loaded, so tuning cannot leak into the final test. Writes artifacts/best_params.yaml, which
train.py picks up automatically. Budget is set in configs/config.yaml -> tuning.
"""
from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml

from .config import load_config
from .features import apply_plan_dropout, feature_columns, hide_plan
from .models import _fit
from .splits import _stratified_folds, dev_only, load_splits

log = logging.getLogger("glucotwin.tune")


def run(cfg) -> dict:
    tcfg = cfg["tuning"]
    splits = load_splits(cfg)
    ds = dev_only(pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet"), splits)
    feats = feature_columns(ds, cfg["features"].get("exclude_prefixes"))
    h = tcfg["tune_horizon"]
    ds = ds[ds[f"target_delta_{h}"].notna()]
    rng = np.random.default_rng(cfg["model"]["seed"])
    groups = pd.Series({int(p): splits["groups"][str(p)] for p in splits["dev"]})
    folds = _stratified_folds(splits["dev"], groups, tcfg["n_folds"], np.random.default_rng(cfg["split"]["seed"]))
    mono_cfg = cfg["model"].get("monotone", {}) or {}
    mono = [int(mono_cfg.get(f, 0)) for f in feats]
    drop = cfg["features"]["plan_dropout"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        scores, iters = [], []
        for i, test in enumerate(folds):
            tr_all, te = ds[~ds["pid"].isin(test)], ds[ds["pid"].isin(test)]
            # Early stopping uses inner validation patients taken from the training side, so the
            # held-out fold that produces the score never influences the model.
            tr_p = np.array(sorted(tr_all["pid"].unique()))
            inner = set(np.random.default_rng(i).choice(tr_p, size=max(1, len(tr_p) // 7), replace=False))
            tr, va = tr_all[~tr_all["pid"].isin(inner)], tr_all[tr_all["pid"].isin(inner)]
            m = lgb.LGBMRegressor(objective="regression", n_estimators=3000, subsample_freq=1, verbose=-1,
                                  monotone_constraints=mono, monotone_constraints_method="intermediate",
                                  random_state=cfg["model"]["seed"], **params)
            _fit(m, apply_plan_dropout(tr[feats], drop, rng), tr[f"target_delta_{h}"],
                 hide_plan(va[feats]), va[f"target_delta_{h}"], [lgb.early_stopping(100, verbose=False)])
            err = m.predict(hide_plan(te[feats])) - te[f"target_delta_{h}"].values
            scores.append(float(np.sqrt(np.mean(err**2))))
            iters.append(int(m.best_iteration_ or 3000))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_estimators", int(max(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["model"]["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=tcfg["n_trials"], timeout=tcfg["timeout_minutes"] * 60)
    best = dict(study.best_params)
    # Upper bound on trees; training still early-stops on its own validation patients.
    best["n_estimators"] = int(study.best_trial.user_attrs.get("n_estimators", 1000) * 1.2)
    out = cfg.path("artifacts_dir") / "best_params.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(best))
    log.info("best RMSE %.2f with %s", study.best_value, best)
    return {"best_rmse": study.best_value, "params": best, "trials": len(study.trials)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.trials:
        cfg["tuning"]["n_trials"] = args.trials
    print(run(cfg))


if __name__ == "__main__":
    main()
