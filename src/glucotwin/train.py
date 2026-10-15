"""Step 3 of the pipeline:  python -m glucotwin.train [--quick]

Uses DEVELOPMENT patients only. The locked test patients are never loaded here.

1. Leave-patients-out cross-validation over the development patients (stratified folds from
   data/processed/splits.json): every development patient is scored by a model that never saw
   them. Use these numbers to compare ideas and choose settings.
2. Baselines (persistence, linear trend, ridge) are scored on exactly the same rows.
3. Personalization: each held-out patient's twin is calibrated on their first N days (purged so
   no calibration target reaches into the evaluation days) and scored on their later days.
4. A development model is trained on all development patients; ``python -m glucotwin.test``
   scores it once on the locked test patients.

Outputs: artifacts/models/, artifacts/cv/ (RESULTS_CV.md, metrics, figures, out-of-fold predictions).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import evaluate as ev
from .config import load_config
from .features import feature_columns, hide_plan
from .models import GlucoTwinModel, PersistenceBaseline, RidgeBaseline, TrendBaseline
from .splits import calibration_split, dev_only, load_splits, purge_minutes

log = logging.getLogger("glucotwin.train")
BASELINE_NAMES = [PersistenceBaseline.name, TrendBaseline.name, RidgeBaseline.name]


def load_best_params(cfg) -> dict:
    """Tuned parameters from `python -m glucotwin.tune`. Experiment and ablation runs (which write to
    sub-folders of artifacts/) use the main artifacts/best_params.yaml, so they compare like with like."""
    art, main = cfg.path("artifacts_dir").resolve(), (cfg["_root"] / "artifacts").resolve()
    candidates = [art / "best_params.yaml"] + ([main / "best_params.yaml"] if main in art.parents else [])
    for p in candidates:
        if p.exists():
            params = yaml.safe_load(p.read_text()) or {}
            log.info("using tuned parameters from %s", p)
            return params
    return {}


def keep_columns(ds: pd.DataFrame, cfg, groups: dict) -> pd.DataFrame:
    """Identity, truth and context columns carried into every prediction frame."""
    horizons = cfg["task"]["horizons_min"]
    keep = ["pid", "timestamp", "day", "glucose", "carbs_0_30", "time_since_meal", "label_spike",
            "label_hypo", "fut_max_120"] + [f"target_delta_{h}" for h in horizons]
    out = ds[keep].copy()
    out["group"] = out["pid"].astype(str).map(groups).fillna("unknown")
    return out


def predict_frame(model: GlucoTwinModel, ridge: RidgeBaseline, te: pd.DataFrame, feats: list[str],
                  horizons: list[int]) -> pd.DataFrame:
    """All predictions for rows of patients the model never saw."""
    out = pd.DataFrame(index=te.index)
    Xh = hide_plan(te[feats])          # honest mode: no knowledge of future meals or activity
    ph = model.predict(Xh)
    pp = model.predict(te[feats])      # oracle mode: the real future meals/activity are given
    g = te["glucose"].to_numpy()
    for h in horizons:
        out[f"pred_{h}"] = g + ph[h]["delta"].to_numpy()
        out[f"lo_{h}"] = g + ph[h]["lo"].to_numpy()
        out[f"hi_{h}"] = g + ph[h]["hi"].to_numpy()
        out[f"pred_plan_{h}"] = g + pp[h]["delta"].to_numpy()
        for b in (PersistenceBaseline(), TrendBaseline(), ridge):
            out[f"base_{b.name}_{h}"] = g + b.predict_delta(te, h)
    for event in ("spike", "hypo"):
        out[f"p_{event}"] = model.predict_event(Xh, event)
        out[f"thr_{event}"] = model.thresholds.get(event, np.nan)
    return out


def personalize_frame(model: GlucoTwinModel, te: pd.DataFrame, feats: list[str], cfg) -> pd.DataFrame:
    """Calibrate each unseen patient on their first days (purged) and predict their later days."""
    horizons = cfg["task"]["horizons_min"]
    cal_days = cfg["personalization"]["calibration_days"]
    purge = purge_minutes(cfg)
    out = pd.DataFrame(index=te.index, columns=[f"pred_pers_{h}" for h in horizons], dtype=float)
    for pid, d in te.groupby("pid"):
        calib, later, cutoff = calibration_split(d, cal_days, purge)
        if len(calib) < 100 or later.empty:
            continue
        model.calibrate_patient(int(pid), calib, cutoff=cutoff, purge_min=purge)
        pers = model.predict(hide_plan(later[feats]), pid=int(pid), personalized=True)
        for h in horizons:
            out.loc[later.index, f"pred_pers_{h}"] = later["glucose"].to_numpy() + pers[h]["delta"].to_numpy()
    return out


def cross_validate(dev: pd.DataFrame, splits: dict, cfg, feats: list[str], params: dict, art: Path,
                   groups: dict) -> pd.DataFrame:
    horizons = cfg["task"]["horizons_min"]
    rng = np.random.default_rng(cfg["model"]["seed"])
    folds = splits["folds"]
    parts = []
    (art / "models").mkdir(parents=True, exist_ok=True)
    for k, test_pids in enumerate(folds):
        t0 = time.time()
        tr, te = dev[~dev["pid"].isin(test_pids)], dev[dev["pid"].isin(test_pids)]
        assert not set(tr["pid"]) & set(te["pid"]), "patient on both sides of a CV split"
        model = GlucoTwinModel(cfg, feats, params).fit(tr, rng)
        ridge = RidgeBaseline(feats).fit(tr, horizons)
        frame = pd.concat([keep_columns(te, cfg, groups), predict_frame(model, ridge, te, feats, horizons),
                           personalize_frame(model, te, feats, cfg)], axis=1)
        frame["fold"] = k
        for h in horizons:
            frame[f"band_adj_{h}"] = model.band_adjust.get(h, 0.0)
        parts.append(frame)
        model.save(art / "models" / f"fold_{k}.joblib")
        log.info("fold %d/%d done (%d held-out patients) in %.0fs", k + 1, len(folds), len(test_pids),
                 time.time() - t0)
    return pd.concat(parts).sort_index()


# ============================================================================ metrics
def summarize(frame: pd.DataFrame, cfg, figs: Path, who: str = "unseen patients") -> dict:
    """Metrics + figures from a prediction frame (CV out-of-fold or locked test)."""
    horizons = cfg["task"]["horizons_min"]
    figs.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"regression": {}, "regression_post_meal": {}, "plan_known": {}, "personalization": {},
                     "baselines": {}, "patients": int(frame["pid"].nunique())}
    post_meal = frame["time_since_meal"] <= 30
    rmse_table = {}
    g0 = frame["glucose"]
    for h in horizons:
        y = frame["glucose"] + frame[f"target_delta_{h}"]
        metrics["regression"][h] = ev.regression_metrics(y, frame[f"pred_{h}"], g0)
        metrics["regression_post_meal"][h] = ev.regression_metrics(
            y[post_meal], frame.loc[post_meal, f"pred_{h}"], g0[post_meal])
        metrics["plan_known"][h] = ev.regression_metrics(y, frame[f"pred_plan_{h}"], g0)
        inside = (y >= frame[f"lo_{h}"]) & (y <= frame[f"hi_{h}"])
        metrics["regression"][h]["band_coverage_pct"] = float(inside[y.notna()].mean() * 100)
        metrics["baselines"][h] = {b: ev.regression_metrics(y, frame[f"base_{b}_{h}"], g0) for b in BASELINE_NAMES}
        rmse_table[h] = {**{b: metrics["baselines"][h][b]["rmse"] for b in BASELINE_NAMES},
                         "GlucoTwin (LightGBM)": metrics["regression"][h]["rmse"]}
        col = f"pred_pers_{h}"
        if col in frame and frame[col].notna().any():
            m = frame[col].notna() & y.notna()
            metrics["personalization"][h] = {
                "population_rmse": ev.regression_metrics(y[m], frame.loc[m, f"pred_{h}"])["rmse"],
                "personalized_rmse": ev.regression_metrics(y[m], frame.loc[m, col])["rmse"],
            }
    ev.plot_rmse_by_horizon(pd.DataFrame(rmse_table), figs / "rmse_by_horizon.png", who)

    h_main = 60 if 60 in horizons else horizons[0]
    for h in sorted({h_main, max(horizons)}):
        y = frame["glucose"] + frame[f"target_delta_{h}"]
        m = y.notna() & frame[f"pred_{h}"].notna()
        ev.plot_clarke(y[m].to_numpy(), frame.loc[m, f"pred_{h}"].to_numpy(),
                       f"Clarke error grid, {h}-min forecast ({who})", figs / f"clarke_{h}min.png")
    if metrics["personalization"]:
        ev.plot_personalization({h: v["population_rmse"] for h, v in metrics["personalization"].items()},
                                {h: v["personalized_rmse"] for h, v in metrics["personalization"].items()},
                                figs / "personalization.png")

    metrics["classification"] = {}
    for event in ("spike", "hypo"):
        if frame[f"p_{event}"].notna().any():
            metrics["classification"][event] = ev.classification_metrics(
                frame[f"label_{event}"], frame[f"p_{event}"], frame[f"thr_{event}"].fillna(0.5))
    metrics["by_group"] = {}
    for grp, d in frame.groupby("group"):
        row = {"patients": int(d["pid"].nunique())}
        for h in horizons:
            row[f"rmse_{h}"] = ev.regression_metrics(d["glucose"] + d[f"target_delta_{h}"], d[f"pred_{h}"])["rmse"]
        if d["label_spike"].nunique() > 1 and d["p_spike"].notna().any():
            row["spike_auroc"] = ev.classification_metrics(d["label_spike"], d["p_spike"],
                                                           d["thr_spike"].fillna(0.5)).get("auroc")
        metrics["by_group"][grp] = row
    if "spike" in metrics["classification"]:
        ev.plot_calibration(frame["label_spike"], frame["p_spike"], figs / "spike_calibration.png")
        alerts = frame.assign(alert_spike=frame["p_spike"] >= frame["thr_spike"])
        metrics["event_level_spike"] = ev.event_level_metrics(
            alerts, "alert_spike", cfg["task"]["spike_abs_mgdl"], cfg["task"]["spike_window_min"],
            cfg["data"]["resample_minutes"])

    daily = frame.groupby(["pid", "day"])["glucose"].apply(lambda s: (s > 180).sum())
    if len(daily):
        pid, day = daily.idxmax()
        d = frame[(frame["pid"] == pid) & (frame["day"] == day)]
        ev.plot_example(d, f"pred_{h_main}", f"lo_{h_main}", f"hi_{h_main}", h_main,
                        f"Patient {pid:03d}, day {day}: {h_main}-min forecasts vs measured glucose",
                        figs / "example_forecast.png")
    return metrics


def cross_conformal_band(oof: pd.DataFrame, cfg) -> dict:
    """Band widening for the development model, calibrated on ALL development patients' out-of-fold
    predictions (each scored by a model that never saw them). Far more patients than the few
    early-stopping patients inside one model, so the 80% band holds better on new patients."""
    level = max(cfg["model"]["quantiles"]) - min(cfg["model"]["quantiles"])
    out = {}
    for h in cfg["task"]["horizons_min"]:
        y = oof["glucose"] + oof[f"target_delta_{h}"]
        lo_raw = oof[f"lo_{h}"] + oof[f"band_adj_{h}"]       # undo each fold model's own widening
        hi_raw = oof[f"hi_{h}"] - oof[f"band_adj_{h}"]
        score = np.maximum(lo_raw - y, y - hi_raw).dropna().to_numpy()
        if len(score):
            out[h] = float(max(np.quantile(score, min(level * (1 + 1 / len(score)), 1.0)), 0.0))
    return out


def write_results_md(metrics: dict, cfg, path: Path, title: str, intro: str, timings: dict | None = None,
                     ci: dict | None = None) -> None:
    hs = cfg["task"]["horizons_min"]
    lines = [f"# {title}", "", intro, "",
             "## Glucose forecast error (RMSE, mg/dL; lower is better)", "",
             "| Model | " + " | ".join(f"{h} min" for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for b in BASELINE_NAMES:
        lines.append(f"| {b} | " + " | ".join(f"{metrics['baselines'][h][b]['rmse']:.1f}" for h in hs) + " |")

    def cell(h):
        v = metrics["regression"][h]["rmse"]
        if ci and h in ci.get("rmse", {}):
            lo, hi = ci["rmse"][h]
            return f"**{v:.1f}** ({lo:.1f} to {hi:.1f})"
        return f"**{v:.1f}**"

    lines.append("| **GlucoTwin (honest: future meals and activity unknown)** | " + " | ".join(cell(h) for h in hs) + " |")
    if metrics["personalization"]:
        lines.append("| GlucoTwin personalized (scored after the calibration days) | " + " | ".join(
            f"{metrics['personalization'][h]['personalized_rmse']:.1f}" if h in metrics["personalization"] else "-"
            for h in hs) + " |")
        lines.append("| same rows, population model (for comparison) | " + " | ".join(
            f"{metrics['personalization'][h]['population_rmse']:.1f}" if h in metrics["personalization"] else "-"
            for h in hs) + " |")
    lines.append("| Oracle: real future meals and activity given (what-if accuracy, not a forecast) | "
                 + " | ".join(f"{metrics['plan_known'][h]['rmse']:.1f}" for h in hs) + " |")
    if ci:
        lines += ["", "Brackets: 95% confidence interval from resampling patients (bootstrap)."]
    lines += ["", "## Clinical accuracy (Clarke error grid, honest mode)", "",
              "| Horizon | MAE | MARD % | Zone A % | Zone A+B % | Zone D+E % | 80% band coverage % |",
              "|---|---|---|---|---|---|---|"]
    for h in hs:
        r = metrics["regression"][h]
        lines.append(f"| {h} min | {r['mae']:.1f} | {r['mard_pct']:.1f} | {r['clarke_A_pct']:.1f} | "
                     f"{r['clarke_AB_pct']:.1f} | {r['clarke_DE_pct']:.2f} | {r['band_coverage_pct']:.1f} |")
    lines += ["", "## R2: read this before quoting it", "",
              "R2 on absolute glucose is inflated by autocorrelation: doing nothing (persistence) already "
              "scores high. R2 of the predicted *change* is the honest skill measure.", "",
              "| Horizon | R2 absolute (GlucoTwin) | R2 absolute (persistence) | R2 of change (GlucoTwin) | Pearson r |",
              "|---|---|---|---|---|"]
    for h in hs:
        r, p0 = metrics["regression"][h], metrics["baselines"][h][BASELINE_NAMES[0]]
        lines.append(f"| {h} min | {r['r2']:.3f} | {p0['r2']:.3f} | {r.get('r2_delta', float('nan')):.3f} | "
                     f"{r['pearson_r']:.3f} |")
    if metrics.get("by_group"):
        lines += ["", "## By metabolic group", "",
                  "| Group | Patients | " + " | ".join(f"RMSE {h} min" for h in hs) + " | Spike AUROC |",
                  "|---|---|" + "---|" * len(hs) + "---|"]
        for grp, r in metrics["by_group"].items():
            au = r.get("spike_auroc")
            rm = " | ".join(f"{r[f'rmse_{h}']:.1f}" for h in hs)
            lines.append(f"| {grp} | {r['patients']} | {rm} | {f'{au:.3f}' if au is not None else 'n/a'} |")
    if "spike" in metrics.get("classification", {}) and "auroc" in metrics["classification"]["spike"]:
        c = metrics["classification"]["spike"]
        e = metrics.get("event_level_spike", {})
        lead = e.get("median_lead_time_min")
        au_ci = ""
        if ci and "spike_auroc" in ci:
            au_ci = f" (95% CI {ci['spike_auroc'][0]:.3f} to {ci['spike_auroc'][1]:.3f})"
        lines += ["", "## Adverse-event prediction: glucose excursion within 2 hours", "",
                  f"- Rows evaluated: {c['n']:,} (positive rate {100 * c.get('prevalence', 0):.1f}%)",
                  f"- AUROC {c['auroc']:.3f}{au_ci}, AUPRC {c['auprc']:.3f}, Brier score {c['brier']:.3f} "
                  f"(accuracy is deliberately not reported: with {100 * c.get('prevalence', 0):.0f}% positives "
                  "it rewards a model that never alerts)",
                  f"- At the threshold chosen on development patients: precision {c['precision']:.2f}, recall "
                  f"(sensitivity) {c['recall']:.2f}, specificity {c['specificity']:.2f}, F1 {c['f1']:.2f}",
                  f"- Event level: {e.get('events', 0)} crossings above {cfg['task']['spike_abs_mgdl']} mg/dL; "
                  f"{100 * e.get('event_recall', 0):.0f}% caught in advance, median warning "
                  f"{lead if lead is not None else 'n/a'} min (longest measurable: "
                  f"{cfg['task']['spike_window_min']} min), "
                  f"{e.get('false_alerts_per_day', 0):.2f} false alerts per patient-day, alert active "
                  f"{e.get('alert_on_time_pct', 0):.0f}% of the time"]
    if "hypo" in metrics.get("classification", {}) and "auroc" in metrics["classification"]["hypo"]:
        c = metrics["classification"]["hypo"]
        lines += ["", "## Hypoglycaemia (< 70 mg/dL within 60 min)", "",
                  f"- AUROC {c['auroc']:.3f}, AUPRC {c['auprc']:.3f}, precision {c['precision']:.2f}, "
                  f"recall {c['recall']:.2f}"]
    lines += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in
                                       sorted((path.parent / "figures").glob("*.png"))]
    if timings:
        lines += ["", "## Run time", ""] + [f"- {k}: {v}" for k, v in timings.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_experiment(cfg, metrics: dict, tag: str, feats: list[str]) -> None:
    """Append one line per training run to artifacts/experiments.csv (the team's results log)."""
    import datetime as _dt
    row = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "tag": tag,
           "artifacts": str(cfg.path("artifacts_dir")), "n_features": len(feats),
           "excluded": "|".join(cfg["features"].get("exclude_prefixes") or [])}
    row["threshold_mode"] = cfg["model"]["classifier_threshold"]
    row["spike_definition"] = cfg["task"]["spike_definition"]
    row["personalization"] = cfg["personalization"].get("method", "residual")
    for h, r in metrics["regression"].items():
        row[f"cv_rmse_{h}"] = round(r["rmse"], 2)
        row[f"cv_mae_{h}"] = round(r["mae"], 2)
    for h, r in metrics.get("personalization", {}).items():
        row[f"cv_pers_gain_{h}"] = round(r["population_rmse"] - r["personalized_rmse"], 2)
    c = metrics.get("classification", {}).get("spike", {})
    row.update({"cv_spike_auroc": round(c.get("auroc", float("nan")), 4),
                "cv_spike_auprc": round(c.get("auprc", float("nan")), 4),
                "cv_false_alerts_per_day": round(metrics.get("event_level_spike", {}).get(
                    "false_alerts_per_day", float("nan")), 2),
                "cv_alert_on_time_pct": round(metrics.get("event_level_spike", {}).get(
                    "alert_on_time_pct", float("nan")), 1),
                "cv_spike_events_caught_pct": round(100 * metrics.get("event_level_spike", {}).get(
                    "event_recall", float("nan")), 1)})
    art, main = cfg.path("artifacts_dir").resolve(), (cfg["_root"] / "artifacts").resolve()
    # ablations/experiments (artifacts/<sub-folder>) log to the main file; runs elsewhere (tests) log locally
    log_path = (main if (art == main or main in art.parents) else art) / "experiments.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:   # re-write the whole file so rows stay aligned when new columns appear
        previous = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    except Exception:  # pragma: no cover - a hand-edited or damaged log
        previous = pd.DataFrame()
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)


def run(cfg, quick: bool = False, tag: str = "default") -> dict:
    timings = {}
    t0 = time.time()
    art = cfg.path("artifacts_dir")
    cv_dir = art / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(cfg)
    ds = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    dev = dev_only(ds, splits)                     # the locked test patients are dropped here
    del ds
    feats = feature_columns(dev, cfg["features"].get("exclude_prefixes"))
    params = load_best_params(cfg)
    if quick:
        params = {**params, "n_estimators": 150, "learning_rate": 0.1}
    (art / "feature_list.json").write_text(json.dumps(feats, indent=2))
    pid_fold = {str(p): k for k, f in enumerate(splits["folds"]) for p in f}
    pid_fold.update({str(p): "test" for p in splits["test"]})
    (art / "pid_fold.json").write_text(json.dumps(pid_fold, indent=2))

    oof = cross_validate(dev, splits, cfg, feats, params, art, splits["groups"])
    timings["cross-validation"] = f"{time.time() - t0:.0f} s"
    metrics = summarize(oof, cfg, cv_dir / "figures", "unseen development patients")
    oof.to_parquet(cv_dir / "oof_predictions.parquet", index=False)

    t1 = time.time()
    model_dev = GlucoTwinModel(cfg, feats, params).fit(dev, np.random.default_rng(cfg["model"]["seed"]))
    model_dev.band_adjust.update(cross_conformal_band(oof, cfg))
    model_dev.save(art / "models" / "model_dev.joblib")
    timings["development model"] = f"{time.time() - t1:.0f} s"

    sample = dev.sample(min(5000, len(dev)), random_state=0)
    h_imp = 60 if 60 in model_dev.horizons else model_dev.horizons[0]
    imp = model_dev.grouped_contributions(hide_plan(sample[feats]), h_imp)
    ev.plot_importance(imp.abs().mean(), cv_dir / "figures" / "feature_groups.png")
    metrics["feature_group_importance"] = imp.abs().mean().sort_values(ascending=False).round(3).to_dict()
    timings["total"] = f"{time.time() - t0:.0f} s"
    metrics["timings"] = timings
    metrics["n_dev_patients"], metrics["n_test_patients_locked"] = len(splits["dev"]), len(splits["test"])
    (cv_dir / "metrics_cv.json").write_text(json.dumps(metrics, indent=2, default=float))
    (art / "run_info.json").write_text(json.dumps({
        "fingerprint": splits.get("fingerprint"), "dev_patients": splits["dev"], "test_patients": splits["test"],
        "quick": quick, "tag": tag, "n_features": len(feats)}, indent=2))
    write_results_md(
        metrics, cfg, cv_dir / "RESULTS_CV.md",
        "GlucoTwin development results (cross-validation)",
        f"Leave-patients-out cross-validation over {len(splits['dev'])} development patients: each one is "
        "scored by a model that never saw them. Use these numbers to choose settings. The "
        f"{len(splits['test'])} locked test patients were not used; their results come from "
        "`python -m glucotwin.test` (artifacts/test/RESULTS_TEST.md).", timings)
    _log_experiment(cfg, metrics, tag, feats)
    log.info("done: %s", timings)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer trees for a fast smoke test")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="ablation: drop features starting with these prefixes, e.g. --exclude ehr_")
    ap.add_argument("--artifacts", default=None, help="write to another folder, e.g. artifacts/ablation_no_ehr")
    ap.add_argument("--tag", default="default", help="name of this run in artifacts/experiments.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    over: dict = {}
    if args.exclude is not None:
        over["features"] = {"exclude_prefixes": args.exclude}
    if args.artifacts:
        over["paths"] = {"artifacts_dir": args.artifacts}
    m = run(load_config(args.config, over or None), quick=args.quick, tag=args.tag)
    print(json.dumps({"cv_rmse": {h: round(v["rmse"], 1) for h, v in m["regression"].items()},
                      "cv_spike_auroc": m.get("classification", {}).get("spike", {}).get("auroc")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
