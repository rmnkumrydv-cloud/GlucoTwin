"""Twin engine.

* One LightGBM regressor per horizon predicts the glucose CHANGE (delta) from now, plus
  two quantile models (10th / 90th percentile) that form the uncertainty band.
* A LightGBM classifier predicts the adverse event (post-meal spike in the next 2 h);
  a second one predicts hypoglycaemia if the data contains enough events.
* Monotonic constraints keep the what-if simulator physiologically sensible.
* Personalization: a small per-patient ridge model learns the residual error of the
  population model from the patient's first days ("calibrating the twin").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PLAN_COLS, apply_plan_dropout, feature_group, hide_plan

log = logging.getLogger(__name__)

PERSONAL_COLS = ["glucose", "glucose_delta_30", "cob_g", "carb_rate", "mets_mean_30", "hr_rel_rest",
                 "tod_sin", "tod_cos"]


# ============================================================================ baselines
class PersistenceBaseline:
    name = "Persistence (no change)"

    def predict_delta(self, X: pd.DataFrame, h: int) -> np.ndarray:
        return np.zeros(len(X))


class TrendBaseline:
    name = "Linear trend (last 30 min)"

    def predict_delta(self, X: pd.DataFrame, h: int) -> np.ndarray:
        slope = X["glucose_delta_30"].fillna(0.0).values / 30.0
        return np.clip(slope * h, -120, 120)


class RidgeBaseline:
    name = "Ridge regression"

    def __init__(self, features: list[str]):
        self.features = [f for f in features if f not in PLAN_COLS]
        self.models: dict[int, object] = {}
        self.medians: pd.Series | None = None

    def fit(self, df: pd.DataFrame, horizons: list[int]) -> "RidgeBaseline":
        X = df[self.features]
        self.medians = X.median(numeric_only=True).fillna(0.0)
        Xf = X.fillna(self.medians)
        for h in horizons:
            y = df[f"target_delta_{h}"]
            m = y.notna().values
            self.models[h] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xf[m], y[m])
        return self

    def predict_delta(self, X: pd.DataFrame, h: int) -> np.ndarray:
        return self.models[h].predict(X[self.features].fillna(self.medians))


# ============================================================================ main model
@dataclass
class GlucoTwinModel:
    cfg: dict
    features: list[str]
    params: dict = field(default_factory=dict)
    regs: dict = field(default_factory=dict)          # h -> {"point": m, "q10": m, "q90": m}
    clfs: dict = field(default_factory=dict)          # "spike"/"hypo" -> classifier
    thresholds: dict = field(default_factory=dict)
    calibrations: dict = field(default_factory=dict)  # pid -> {h: fitted pipeline}
    band_adjust: dict = field(default_factory=dict)   # h -> conformal widening (mg/dL)
    cal_cutoff: dict = field(default_factory=dict)    # pid -> first time personalization may be used
    train_pids: list = field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    @property
    def horizons(self) -> list[int]:
        return list(self.cfg["task"]["horizons_min"])

    def _lgb_params(self) -> dict:
        p = dict(self.cfg["model"]["lgbm"])
        p.update(self.params or {})
        p["random_state"] = self.cfg["model"]["seed"]
        p["n_jobs"] = p.get("n_jobs", -1)
        return p

    def _monotone(self) -> list[int]:
        mono = self.cfg["model"].get("monotone", {}) or {}
        return [int(mono.get(f, 0)) for f in self.features]

    # ------------------------------------------------------------------ training
    def fit(self, df: pd.DataFrame, rng: np.random.Generator | None = None) -> "GlucoTwinModel":
        rng = rng or np.random.default_rng(self.cfg["model"]["seed"])
        pids = np.array(sorted(df["pid"].unique()))
        self.train_pids = pids.tolist()
        n_val = max(1, int(round(len(pids) * self.cfg["model"]["early_stopping_val_frac"])))
        val_pids = set(rng.choice(pids, size=min(n_val, len(pids) - 1), replace=False)) if len(pids) > 1 else set()
        tr = df[~df["pid"].isin(val_pids)]
        va = df[df["pid"].isin(val_pids)] if val_pids else df.sample(frac=0.1, random_state=0)

        drop = float(self.cfg["features"]["plan_dropout"])
        Xtr = apply_plan_dropout(tr[self.features], drop, rng)
        Xva = apply_plan_dropout(va[self.features], drop, rng)
        params = self._lgb_params()
        mono = self._monotone()
        stop = [lgb.early_stopping(100, verbose=False)]

        for h in self.horizons:
            ytr, yva = tr[f"target_delta_{h}"], va[f"target_delta_{h}"]
            mt, mv = ytr.notna().values, yva.notna().values
            models = {}
            point = lgb.LGBMRegressor(objective="regression", monotone_constraints=mono,
                                      monotone_constraints_method="intermediate", **params)
            sw = zone_weights((tr["glucose"] + ytr).values[mt], self.cfg)
            _fit(point, Xtr[mt], ytr[mt], Xva[mv], yva[mv], stop, sample_weight=sw)
            models["point"] = point
            for q in self.cfg["model"]["quantiles"]:
                # LightGBM does not allow monotone constraints with the quantile objective; the
                # band is clipped to always contain the (monotone) point forecast in predict().
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
                _fit(m, Xtr[mt], ytr[mt], Xva[mv], yva[mv], stop)
                models[f"q{int(round(q * 100))}"] = m
            self.regs[h] = models
            # Conformalized quantile regression (Romano et al., 2019): widen the band on
            # validation patients so its real coverage matches the nominal level.
            lo_q, hi_q = f"q{int(round(min(self.cfg['model']['quantiles']) * 100))}", \
                f"q{int(round(max(self.cfg['model']['quantiles']) * 100))}"
            Xv_h = hide_plan(va[self.features])[mv]
            pv = models["point"].predict(Xv_h)
            lo_v = np.minimum(models[lo_q].predict(Xv_h), pv)
            hi_v = np.maximum(models[hi_q].predict(Xv_h), pv)
            score = np.maximum(lo_v - yva[mv].values, yva[mv].values - hi_v)
            level = max(self.cfg["model"]["quantiles"]) - min(self.cfg["model"]["quantiles"])
            self.band_adjust[h] = float(max(np.quantile(score, min(level * (1 + 1 / len(score)), 1.0)), 0.0))
            log.info("horizon %d min: %d trees", h, point.best_iteration_ or params["n_estimators"])

        for event in ("spike", "hypo"):
            col = f"label_{event}"
            ytr, yva = tr[col], va[col]
            mt, mv = ytr.notna().values, yva.notna().values
            n_pos = int(ytr[mt].sum())
            if event == "hypo" and not self.cfg["model"].get("hypo_alerts", True):
                continue
            if event == "hypo" and n_pos < self.cfg["model"]["min_hypo_events"]:
                log.info("hypo model skipped: only %d positive rows", n_pos)
                continue
            if n_pos == 0 or yva[mv].sum() == 0:
                log.warning("%s model skipped: no positive examples", event)
                continue
            spw = float((len(ytr[mt]) - n_pos) / max(n_pos, 1))
            # Same physiological constraints as the forecaster, so a planned walk or a smaller
            # meal can never increase the predicted risk.
            clf = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, monotone_constraints=mono,
                                     monotone_constraints_method="intermediate", **params)
            _fit(clf, Xtr[mt], ytr[mt], Xva[mv], yva[mv], stop, eval_metric="average_precision")
            self.clfs[event] = clf
            # Alerts in deployment are made without a plan, so pick the threshold that way.
            p_val = clf.predict_proba(hide_plan(va[self.features])[mv])[:, 1]
            self.thresholds[event] = _choose_threshold(yva[mv].values, p_val,
                                                       self.cfg["model"]["classifier_threshold"])
        return self

    # ------------------------------------------------------------------ prediction
    def predict(self, X: pd.DataFrame, pid: int | None = None, personalized: bool = False) -> dict:
        """Returns {h: DataFrame(delta, lo, hi)} of predicted glucose change."""
        out = {}
        cal = self.calibrations.get(pid) if (personalized and pid is not None) else None
        qs = [f"q{int(round(q * 100))}" for q in self.cfg["model"]["quantiles"]]
        for h, models in self.regs.items():
            Xf = X[self.features]
            point = models["point"].predict(Xf)
            lo = models[qs[0]].predict(Xf)
            hi = models[qs[-1]].predict(Xf)
            if cal is not None and h in cal:
                if isinstance(cal[h], _PersonalGain):
                    corr = cal[h].predict(point)
                else:
                    corr = cal[h].predict(X[PERSONAL_COLS].fillna(0.0).values)
                point, lo, hi = point + corr, lo + corr, hi + corr
            lo, hi = np.minimum(lo, point), np.maximum(hi, point)
            adj = self.band_adjust.get(h, 0.0)
            lo, hi = lo - adj, hi + adj
            out[h] = pd.DataFrame({"delta": point, "lo": lo, "hi": hi}, index=X.index)
        return out

    def predict_event(self, X: pd.DataFrame, event: str = "spike") -> np.ndarray:
        if event not in self.clfs:
            return np.full(len(X), np.nan)
        return self.clfs[event].predict_proba(X[self.features])[:, 1]

    def contributions(self, X: pd.DataFrame, h: int = 60) -> pd.DataFrame:
        """Per-row SHAP-style contributions (LightGBM TreeSHAP) for the point forecast."""
        contrib = self.regs[h]["point"].predict(X[self.features], pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.features, index=X.index)

    def grouped_contributions(self, X: pd.DataFrame, h: int = 60) -> pd.DataFrame:
        c = self.contributions(X, h)
        groups = {f: feature_group(f) for f in self.features}
        return c.T.groupby(groups).sum().T

    # ------------------------------------------------------------------ personalization
    def calibrate_patient(self, pid: int, df_patient: pd.DataFrame, cutoff=None, purge_min: int = 120) -> dict:
        """Personalize the twin on this patient's calibration rows (honest mode).

        ``df_patient`` must contain only rows whose targets end before ``cutoff`` (use
        ``splits.calibration_split``). Method (config ``personalization.method``):

        residual  small ridge model of the population model's error from 8 context features
        gain      the patient's own sensitivity: corrected change = a + (1 + c) x predicted change,
                  i.e. "this person rises more (or less) than the average patient"; c is clipped so
                  the gain stays between 0.5x and 2x, and both terms are shrunk towards the
                  population model when there are few days of data
        bias      a constant offset only

        Safety check: a horizon's correction is kept only if it lowers the error on the last third
        of the calibration rows when fitted on the first two thirds, with a purge gap between them
        so the check itself cannot peek ahead.
        """
        if pid in self.train_pids:
            raise ValueError(f"patient {pid} was used to train this model; calibrate only unseen patients")
        pcfg = self.cfg["personalization"]
        method = pcfg.get("method", "residual")
        alpha = float(pcfg["ridge_alpha"])
        self.calibrations.pop(pid, None)
        d = df_patient.sort_values("timestamp")
        X = hide_plan(d[self.features])
        base = self.predict(X)
        Z_all = d[PERSONAL_COLS].fillna(0.0).values
        t = d["timestamp"]
        split = t.min() + (t.max() - t.min()) * 2 / 3
        first = (t + pd.Timedelta(minutes=purge_min) <= split).values
        second = (t >= split).values
        cal = {}
        for h in self.horizons:
            y = d[f"target_delta_{h}"].values
            ok = ~np.isnan(y)
            if ok.sum() < 100:
                continue
            pred = base[h]["delta"].values
            resid = y - pred
            fit_m, val_m = ok & first, ok & second
            if fit_m.sum() < 50 or val_m.sum() < 30:
                continue
            if method == "residual":
                trial = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Z_all[fit_m], resid[fit_m])
                corr_val = trial.predict(Z_all[val_m])
            else:
                trial = _PersonalGain(method).fit(pred[fit_m], resid[fit_m], t[fit_m])
                corr_val = trial.predict(pred[val_m])
            before = np.sqrt(np.mean(resid[val_m] ** 2))
            after = np.sqrt(np.mean((resid[val_m] - corr_val) ** 2))
            if after < before:
                if method == "residual":
                    cal[h] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Z_all[ok], resid[ok])
                else:
                    cal[h] = _PersonalGain(method).fit(pred[ok], resid[ok], t[ok])
        self.calibrations[pid] = cal
        self.cal_cutoff[pid] = pd.Timestamp(cutoff) if cutoff is not None else t.max()
        return cal

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "GlucoTwinModel":
        return joblib.load(path)


class _PersonalGain:
    """Per-patient correction  a + c * (predicted change), shrunk towards zero with few days."""

    def __init__(self, method: str = "gain"):
        self.method, self.a, self.c = method, 0.0, 0.0

    def fit(self, pred: np.ndarray, resid: np.ndarray, times: pd.Series) -> "_PersonalGain":
        days = max((times.max() - times.min()) / pd.Timedelta(days=1), 0.0)
        shrink = days / (days + 1.0)                      # 3 days of data -> 75% of the estimate
        if self.method == "gain" and np.var(pred) > 1e-6:
            c = np.cov(pred, resid, bias=True)[0, 1] / np.var(pred)
            self.c = float(np.clip(c * shrink, -0.5, 1.0))  # personal gain between 0.5x and 2x
        self.a = float(np.mean(resid - self.c * pred) * shrink)
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        return self.a + self.c * np.asarray(pred, float)


def _fit(model, X, y, Xv, yv, callbacks, sample_weight=None, **kw):
    """Fit with early stopping on validation patients, across LightGBM versions."""
    try:  # LightGBM >= 4.7 prefers eval_X / eval_y
        return model.fit(X, y, sample_weight=sample_weight, eval_X=(Xv,), eval_y=(yv,),
                         callbacks=callbacks, **kw)
    except TypeError:
        return model.fit(X, y, sample_weight=sample_weight, eval_set=[(Xv, yv)], callbacks=callbacks, **kw)


def zone_weights(future_glucose: np.ndarray, cfg: dict) -> np.ndarray | None:
    """Heavier weight on rows whose outcome is high glucose (where errors matter most)."""
    zw = cfg["model"].get("zone_weights") or {}
    if not zw:
        return None
    w = np.ones(len(future_glucose))
    for thr, weight in sorted(((float(k), float(v)) for k, v in zw.items())):
        w[np.nan_to_num(future_glucose, nan=0.0) >= thr] = weight
    return w


def _choose_threshold(y: np.ndarray, p: np.ndarray, mode: str = "f1") -> float:
    """Alert threshold chosen on validation PATIENTS (never on the patients being scored).

    f1        maximise F1 (sensitive, but on CGMacros it alerts about half the time)
    recall80  catch at least 80% of event rows
    spec90    at most 10% of non-event rows alert (fewer false alarms); also spec85, spec95, ...
    breakeven alert as often as events actually happen (precision ~ recall)
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if mode.startswith("spec"):
        level = float(mode[4:]) / 100 if len(mode) > 4 else 0.9
        neg = p[y == 0]
        return float(np.quantile(neg, level)) if len(neg) else 0.5
    if mode == "breakeven":
        return float(np.quantile(p, 1 - y.mean())) if 0 < y.mean() < 1 else 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if len(thr) == 0:
        return 0.5
    if mode == "recall80":
        ok = np.where(rec >= 0.8)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.nanargmax(f1))])
