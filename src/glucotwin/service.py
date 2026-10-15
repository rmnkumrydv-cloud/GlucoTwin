"""TwinService: the virtual patient, used by both the FastAPI backend and the dashboard.

Every forecast shown in the demo is an honest unseen-patient prediction:
* development patients are served by the cross-validation fold model that never saw them;
* locked test patients are served by the development model, which never saw them either.
Personalization is only applied after a patient's calibration cutoff, never to the days it was
calibrated on.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import load_config
from .ehr import diagnosis_from_hba1c, ehr_summary, load_ehr_records, static_feature_table
from .features import feature_columns, hide_plan, patient_features
from .models import GlucoTwinModel


def _read_json(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class TwinService:
    def __init__(self, cfg=None, use_oof_models: bool = True):
        self.cfg = cfg or load_config()
        proc, art = self.cfg.path("processed_dir"), self.cfg.path("artifacts_dir")
        self.streams = pd.read_parquet(proc / "wearable_stream.parquet")
        self.dataset = pd.read_parquet(proc / "dataset.parquet")
        self.meals = pd.read_parquet(proc / "meals.parquet")
        self.features = feature_columns(self.dataset)
        self.records = load_ehr_records(self.cfg.path("ehr_dir"))
        self.static = static_feature_table(self.records, self.cfg["ehr"]["use_synthetic_fields_as_features"])
        run_info, splits = _read_json(art / "run_info.json"), _read_json(proc / "splits.json")
        if not run_info:
            raise FileNotFoundError("no trained model in artifacts/ (run `python -m glucotwin.train`)")
        if run_info.get("fingerprint") != splits.get("fingerprint"):
            raise FileNotFoundError("the models in artifacts/ were trained on different data "
                                    "(run `python -m glucotwin.train` again)")
        self.dev_model = GlucoTwinModel.load(art / "models" / "model_dev.joblib")
        pf = art / "pid_fold.json"
        self.pid_fold = {int(k): v for k, v in json.loads(pf.read_text()).items()} if pf.exists() else {}
        self.use_oof = use_oof_models
        self._fold_models: dict[int, GlucoTwinModel] = {}
        self.metrics_cv = _read_json(art / "cv" / "metrics_cv.json")
        self.metrics_test = _read_json(art / "test" / "metrics_test.json")
        self.metrics = self.metrics_test or self.metrics_cv
        self.figure_dirs = {"test": art / "test" / "figures", "cv": art / "cv" / "figures"}
        frames = [pd.read_parquet(p) for p in (art / "cv" / "oof_predictions.parquet",
                                                art / "test" / "test_predictions.parquet") if p.exists()]
        self.oof = pd.concat(frames, ignore_index=True) if frames else None
        self.horizons = list(self.cfg["task"]["horizons_min"])
        self.step = int(self.cfg["data"]["resample_minutes"])

    # ------------------------------------------------------------------ models
    def model_for(self, pid: int) -> GlucoTwinModel:
        k = self.pid_fold.get(pid)
        if self.use_oof and isinstance(k, int):
            if k not in self._fold_models:
                self._fold_models[k] = GlucoTwinModel.load(
                    self.cfg.path("artifacts_dir") / "models" / f"fold_{k}.joblib")
            return self._fold_models[k]
        return self.dev_model  # locked test patients (and any new patient)

    def split_of(self, pid: int) -> str:
        k = self.pid_fold.get(pid)
        return "test" if k == "test" else ("development" if k is not None else "new")

    # ------------------------------------------------------------------ patients
    def patients(self) -> pd.DataFrame:
        rows = []
        for pid in sorted(self.dataset["pid"].unique()):
            s = ehr_summary(self.records[pid]) if pid in self.records else {}
            st = self.static.loc[pid] if pid in self.static.index else pd.Series(dtype=float)
            a1c = st.get("ehr_hba1c", np.nan)
            d = self.dataset[self.dataset["pid"] == pid]
            rows.append({
                "pid": int(pid), "age": s.get("age"), "gender": s.get("gender"),
                "hba1c": a1c, "bmi": st.get("ehr_bmi", np.nan), "group": diagnosis_from_hba1c(a1c),
                "days": int(d["day"].max()) + 1,
                "time_above_180_pct": float((d["glucose"] > 180).mean() * 100),
                "mean_glucose": float(d["glucose"].mean()),
            })
        return pd.DataFrame(rows)

    def ehr(self, pid: int) -> dict:
        return {"summary": ehr_summary(self.records[pid]), "fhir_bundle": self.records[pid]}

    def timeline(self, pid: int, start=None, end=None) -> pd.DataFrame:
        d = self.dataset[self.dataset["pid"] == pid]
        if start is not None:
            d = d[d["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            d = d[d["timestamp"] <= pd.Timestamp(end)]
        cols = ["timestamp", "day", "glucose", "carbs_0_30", "mets_mean_15", "hr_mean_15"]
        return d[cols].reset_index(drop=True)

    def patient_meals(self, pid: int) -> pd.DataFrame:
        return self.meals[self.meals["pid"] == pid].reset_index(drop=True)

    def _row_at(self, pid: int, at) -> pd.DataFrame:
        d = self.dataset[self.dataset["pid"] == pid]
        d = d[d["timestamp"] <= pd.Timestamp(at)] if at is not None else d
        if d.empty:
            d = self.dataset[self.dataset["pid"] == pid].iloc[[0]]
        return d.iloc[[-1]]

    # ------------------------------------------------------------------ forecasting
    def forecast(self, pid: int, at=None, personalized: bool = False) -> dict:
        row = self._row_at(pid, at)
        model = self.model_for(pid)
        return self._forecast_from_row(model, row, pid, personalized, plan_known=False)

    def _forecast_from_row(self, model: GlucoTwinModel, row: pd.DataFrame, pid: int, personalized: bool,
                           plan_known: bool) -> dict:
        X = row[self.features] if plan_known else hide_plan(row[self.features])
        t_row = pd.Timestamp(row["timestamp"].iloc[0])
        cutoff = getattr(model, "cal_cutoff", {}).get(pid)
        # only personalize AFTER the calibration period, never on the days used to calibrate
        pers = personalized and pid in model.calibrations and cutoff is not None and t_row >= cutoff
        preds = model.predict(X, pid=pid, personalized=pers)
        t0 = pd.Timestamp(row["timestamp"].iloc[0])
        g0 = float(row["glucose"].iloc[0])
        times = [t0] + [t0 + pd.Timedelta(minutes=h) for h in self.horizons]
        mid = [g0] + [g0 + float(preds[h]["delta"].iloc[0]) for h in self.horizons]
        lo = [g0] + [g0 + float(preds[h]["lo"].iloc[0]) for h in self.horizons]
        hi = [g0] + [g0 + float(preds[h]["hi"].iloc[0]) for h in self.horizons]
        p_spike = float(model.predict_event(X, "spike")[0])
        thr = float(model.thresholds.get("spike", 0.5))
        p_hypo = float(model.predict_event(X, "hypo")[0]) if "hypo" in model.clfs else None
        h_exp = 60 if 60 in self.horizons else self.horizons[0]
        contrib = model.grouped_contributions(X, h_exp).iloc[0].sort_values(key=np.abs, ascending=False)
        drivers = [{"factor": k, "effect_mgdl": round(float(v), 1)} for k, v in contrib.items() if abs(v) >= 0.5][:5]
        thr_mgdl = float(self.cfg["task"]["spike_abs_mgdl"])
        crossing = None
        if g0 < thr_mgdl:
            mins = [0] + self.horizons
            for i in range(1, len(mid)):
                if mid[i] >= thr_mgdl:  # linear interpolation between forecast points
                    frac = (thr_mgdl - mid[i - 1]) / max(mid[i] - mid[i - 1], 1e-9)
                    crossing = round(mins[i - 1] + frac * (mins[i] - mins[i - 1]))
                    break
        return {
            "pid": pid, "at": t0.isoformat(), "glucose_now": g0, "expected_crossing_min": crossing,
            "times": [t.isoformat() for t in times], "forecast": mid, "lo": lo, "hi": hi,
            "peak_forecast": max(mid), "p_spike": p_spike, "spike_threshold": thr,
            "alert": bool(p_spike >= thr), "p_hypo": p_hypo,
            "hypo_alert": bool(p_hypo is not None and p_hypo >= model.thresholds.get("hypo", 1.0)),
            "drivers_60min": drivers, "personalized": bool(pers),
        }

    # ------------------------------------------------------------------ what-if
    def whatif(self, pid: int, at, scenarios: list[dict], personalized: bool = False) -> list[dict]:
        """Each scenario: {"label", "meal": dict|None (None = keep the logged meal),
        "walk_minutes": int, "walk_mets": float, "walk_start_min": int}.

        The meal is placed at time ``at``; no other food is eaten in the next 2 h; activity in
        the next hour is the patient's usual resting level except for the planned walk.
        """
        at = pd.Timestamp(self._row_at(pid, at)["timestamp"].iloc[0])
        s_full = self.streams[self.streams["pid"] == pid].sort_values("timestamp")
        win = s_full[(s_full["timestamp"] >= at - pd.Timedelta(hours=26)) &
                     (s_full["timestamp"] <= at + pd.Timedelta(hours=3))].reset_index(drop=True)
        rest_mets = float(np.nanpercentile(s_full["mets"], 25)) if s_full["mets"].notna().any() else 1.0
        static = self.static.loc[pid].to_dict() if pid in self.static.index else {}
        model = self.model_for(pid)
        macro_cols = ["carbs", "protein", "fat", "fiber", "meal_kcal"]
        results = []
        for sc in scenarios:
            s = win.copy()
            fut = (s["timestamp"] > at) & (s["timestamp"] <= at + pd.Timedelta(hours=2))
            s.loc[fut, macro_cols] = 0.0
            if sc.get("meal") is not None:
                near = (s["timestamp"] - at).abs() <= pd.Timedelta(minutes=15)
                s.loc[near, macro_cols] = 0.0
                i = s.index[s["timestamp"] == at][0]
                for c in macro_cols:
                    s.loc[i, c] = float(sc["meal"].get(c, 0.0))
            nxt = (s["timestamp"] > at) & (s["timestamp"] <= at + pd.Timedelta(hours=2))
            s.loc[nxt, "mets"] = rest_mets
            wm = int(sc.get("walk_minutes", 0) or 0)
            if wm > 0:
                w0 = at + pd.Timedelta(minutes=int(sc.get("walk_start_min", 0)))
                walk = (s["timestamp"] > w0) & (s["timestamp"] <= w0 + pd.Timedelta(minutes=wm))
                s.loc[walk, "mets"] = float(sc.get("walk_mets", 3.5))
            feats = patient_features(s, self.cfg, static, with_targets=False)
            row = feats[feats["timestamp"] == at]
            res = self._forecast_from_row(model, row, pid, personalized, plan_known=True)
            res["label"] = sc.get("label", "Scenario")
            res["meal_used"] = {c: float(row[c].iloc[0]) for c in ("carbs_0_30",)}
            results.append(res)
        return results


@lru_cache(maxsize=1)
def get_service() -> TwinService:
    return TwinService()
