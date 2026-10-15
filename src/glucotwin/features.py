"""Fusion layer: turn a patient's 5-minute stream + EHR features into model rows.

Every feature at time t uses data at or before t only (verified by tests/test_leakage).
The only forward-looking inputs are the explicit *plan* columns (planned carbs and planned
activity), which the doctor or patient types into the what-if simulator. During training
those plan columns are randomly hidden (``plan_dropout``) so the same model can forecast
honestly when no plan is known, and respond to a plan when one is given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAN_COLS = ["plan_carbs", "plan_mets_mean", "plan_known"]
ID_COLS = ["pid", "timestamp", "day", "minute_of_day"]
NON_FEATURE_PREFIXES = ("target_", "label_", "fut_")


def _k(minutes: float, step: int) -> int:
    return max(int(round(minutes / step)), 1)


def _window_sum(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(k, min_periods=1).sum()


def _future_window(x: pd.Series, k: int, how: str, min_periods: int = 1) -> pd.Series:
    """Aggregate over (t, t + k*step] using only future values."""
    rev = x.iloc[::-1].rolling(k, min_periods=min_periods)
    agg = getattr(rev, how)().iloc[::-1]
    return agg.shift(-1)


def _causal_convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    return np.convolve(x, kernel)[: len(x)]


def carb_kernels(tau: float, step: int, horizon_min: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Gamma(2) absorption model: fraction of carbs remaining, and absorption rate per minute."""
    d = np.arange(0, horizon_min + step, step, dtype=float)
    remaining = np.exp(-d / tau) * (1 + d / tau)
    rate = (d / tau**2) * np.exp(-d / tau)
    return remaining, rate


def patient_features(s: pd.DataFrame, cfg: dict, static: dict | None = None,
                     with_targets: bool = True) -> pd.DataFrame:
    """Features (and optionally targets) for one patient's regular-grid stream."""
    step = int(cfg["data"]["resample_minutes"])
    fcfg, tcfg = cfg["features"], cfg["task"]
    s = s.sort_values("timestamp").reset_index(drop=True)
    g = s["glucose"].astype(float)
    out: dict[str, pd.Series] = {
        "pid": s["pid"],
        "timestamp": s["timestamp"],
    }
    t0 = s["timestamp"].iloc[0].normalize()
    out["day"] = ((s["timestamp"] - t0) / pd.Timedelta(days=1)).astype(int)
    mod = s["timestamp"].dt.hour * 60 + s["timestamp"].dt.minute
    out["minute_of_day"] = mod

    # --- glucose history ---------------------------------------------------------------
    out["glucose"] = g
    for lag in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240):
        out[f"glucose_lag_{lag}"] = g.shift(_k(lag, step))
    for w in (15, 30, 60):
        out[f"glucose_delta_{w}"] = g - g.shift(_k(w, step))
    k15 = _k(15, step)
    out["glucose_accel"] = (g - g.shift(k15)) - (g.shift(k15) - g.shift(2 * k15))
    for w in (60, 120, 240, 360):
        r = g.rolling(_k(w, step), min_periods=max(_k(w, step) // 2, 1))
        out[f"glucose_mean_{w}"] = r.mean()
        out[f"glucose_std_{w}"] = r.std()
        out[f"glucose_min_{w}"] = r.min()
        out[f"glucose_max_{w}"] = r.max()
    out["glucose_cv_120"] = out["glucose_std_120"] / out["glucose_mean_120"]
    for w in (30, 60, 180, 360):  # exponentially weighted memory of the glucose trajectory
        out[f"glucose_ewm_{w}"] = g.ewm(halflife=w / step, min_periods=1, ignore_na=True).mean()
    # missingness is information: how old is the reading, how much of the last hour is missing
    if "glucose_age_min" in s:
        out["glucose_age_min"] = s["glucose_age_min"].astype(float)
    out["glucose_missing_60"] = g.isna().astype(float).rolling(_k(60, step), min_periods=1).mean()
    k24 = _k(1440, step)
    mean24 = g.rolling(k24, min_periods=k24 // 4).mean()
    out["glucose_mean_24h"] = mean24
    out["glucose_rel_24h"] = g - mean24
    above = (g > tcfg["spike_abs_mgdl"]).astype(float).where(g.notna())
    out["time_above_180_24h"] = above.rolling(k24, min_periods=k24 // 4).mean()

    # --- meals ---------------------------------------------------------------------------
    carbs, fiber = s["carbs"].fillna(0.0), s["fiber"].fillna(0.0)
    protein, fat, kcal = s["protein"].fillna(0.0), s["fat"].fillna(0.0), s["meal_kcal"].fillna(0.0)
    c30, c60, c120, c240 = (_window_sum(carbs, _k(m, step)) for m in (30, 60, 120, 240))
    out["carbs_0_30"] = c30
    out["carbs_30_60"] = c60 - c30
    out["carbs_60_120"] = c120 - c60
    out["carbs_120_240"] = c240 - c120
    for name, x in (("protein", protein), ("fat", fat), ("fiber", fiber)):
        a60, a180 = _window_sum(x, _k(60, step)), _window_sum(x, _k(180, step))
        out[f"{name}_0_60"] = a60
        out[f"{name}_60_180"] = a180 - a60
    out["kcal_0_60"] = _window_sum(kcal, _k(60, step))
    remaining, rate = carb_kernels(float(fcfg["carb_absorption_tau_min"]), step)
    out["cob_g"] = pd.Series(_causal_convolve(carbs.values, remaining), index=s.index)
    out["carb_rate"] = pd.Series(_causal_convolve(carbs.values, rate), index=s.index)
    fat_kernel = np.exp(-np.arange(0, 481, step) / 180.0)
    out["fat_on_board"] = pd.Series(_causal_convolve(fat.values, fat_kernel), index=s.index)
    is_meal = (carbs + protein + fat + kcal) > 0
    last_meal_time = s["timestamp"].where(is_meal).ffill()
    tsm = (s["timestamp"] - last_meal_time) / pd.Timedelta(minutes=1)
    out["time_since_meal"] = tsm.fillna(720).clip(upper=720)
    out["last_meal_carbs"] = carbs.where(is_meal).ffill().fillna(0.0)
    out["last_meal_fiber_ratio"] = (fiber / carbs.replace(0, np.nan)).where(is_meal).ffill().fillna(0.0)

    # --- activity & heart rate (wearable) --------------------------------------------
    mets, hr = s["mets"].astype(float), s["hr"].astype(float)
    for w in (15, 30, 60, 120):
        k = _k(w, step)
        out[f"mets_mean_{w}"] = mets.rolling(k, min_periods=max(k // 2, 1)).mean()
    out["active_minutes_60"] = (mets >= 3).astype(float).where(mets.notna()).rolling(
        _k(60, step), min_periods=1).sum() * step
    out["act_kcal_60"] = s["act_kcal"].rolling(_k(60, step), min_periods=1).sum()
    hr15 = hr.rolling(_k(15, step), min_periods=1).mean()
    out["hr_mean_15"] = hr15
    out["hr_mean_60"] = hr.rolling(_k(60, step), min_periods=3).mean()
    out["hr_std_60"] = hr.rolling(_k(60, step), min_periods=3).std()
    hr_base = hr.rolling(k24, min_periods=k24 // 4).quantile(0.1)
    out["hr_rel_rest"] = hr15 - hr_base
    out["hr_slope_30"] = hr15 - hr15.shift(_k(30, step))
    out["hr_missing_60"] = hr.isna().astype(float).rolling(_k(60, step), min_periods=1).mean()
    last_hr = s["timestamp"].where(hr.notna()).ffill()
    out["hr_age_min"] = ((s["timestamp"] - last_hr) / pd.Timedelta(minutes=1)).fillna(720).clip(upper=720)

    # --- circadian ---------------------------------------------------------------------
    ang = 2 * np.pi * mod / 1440.0
    out["tod_sin"], out["tod_cos"] = np.sin(ang), np.cos(ang)

    # --- plan (known-future inputs; hidden at random during training) -----------------
    out["plan_carbs"] = _future_window(carbs, _k(fcfg["plan_carbs_window_min"], step), "sum")
    out["plan_mets_mean"] = _future_window(mets, _k(fcfg["plan_mets_window_min"], step), "mean",
                                           min_periods=max(_k(fcfg["plan_mets_window_min"], step) // 2, 1))
    out["plan_known"] = pd.Series(1.0, index=s.index)

    df = pd.DataFrame(out)
    if static:
        for k_, v in static.items():
            df[k_] = v

    if with_targets:
        df = pd.concat([df, _targets(g, cfg)], axis=1)
    return df


def _targets(g: pd.Series, cfg: dict) -> pd.DataFrame:
    step = int(cfg["data"]["resample_minutes"])
    t = cfg["task"]
    out = {}
    for h in t["horizons_min"]:
        out[f"target_delta_{h}"] = g.shift(-_k(h, step)) - g
    kw = _k(t["spike_window_min"], step)
    fut_max = _future_window(g, kw, "max")
    cov = _future_window(g.notna().astype(float), kw, "sum") / kw
    ok = g.notna() & (cov >= t["min_future_coverage"]) & (g < t["spike_abs_mgdl"])
    abs_ev = fut_max >= t["spike_abs_mgdl"]
    rise_ev = (fut_max - g) >= t["spike_rise_mgdl"]
    ev = {"absolute": abs_ev, "rise": rise_ev, "either": abs_ev | rise_ev}[t["spike_definition"]]
    out["label_spike"] = ev.astype(float).where(ok)
    out["fut_max_120"] = fut_max

    kh = _k(t["hypo_window_min"], step)
    fut_min = _future_window(g, kh, "min")
    cov_h = _future_window(g.notna().astype(float), kh, "sum") / kh
    ok_h = g.notna() & (cov_h >= t["min_future_coverage"]) & (g >= t["hypo_mgdl"])
    out["label_hypo"] = (fut_min < t["hypo_mgdl"]).astype(float).where(ok_h)
    return pd.DataFrame(out)


def build_dataset(streams: pd.DataFrame, static: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    parts = []
    for pid, s in streams.groupby("pid", sort=True):
        st = static.loc[pid].to_dict() if pid in static.index else {}
        parts.append(patient_features(s, cfg, st, with_targets=True))
    df = pd.concat(parts, ignore_index=True)
    return df[df["glucose"].notna()].reset_index(drop=True)


def feature_columns(df: pd.DataFrame, exclude_prefixes: list[str] | None = None) -> list[str]:
    cols = [c for c in df.columns if c not in ID_COLS and not c.startswith(NON_FEATURE_PREFIXES)]
    if exclude_prefixes:
        keep_always = {"glucose", "plan_known", "plan_carbs", "plan_mets_mean"}
        cols = [c for c in cols if c in keep_always or not c.startswith(tuple(exclude_prefixes))]
    return cols


def hide_plan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["plan_carbs"] = np.nan
    X["plan_mets_mean"] = np.nan
    X["plan_known"] = 0.0
    return X


def apply_plan_dropout(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    X = X.copy()
    mask = rng.random(len(X)) < rate
    X.loc[mask, "plan_carbs"] = np.nan
    X.loc[mask, "plan_mets_mean"] = np.nan
    X.loc[mask, "plan_known"] = 0.0
    return X


# Plain-language groups used for explanations in the dashboard.
FEATURE_GROUPS = [
    ("Recent glucose level and trend", ("glucose", "time_above")),
    ("Carbs still being absorbed", ("cob_g", "carb_rate")),
    ("Recent meal composition", ("carbs_", "protein_", "fat_", "fiber_", "kcal_", "time_since_meal",
                                 "last_meal", "fat_on_board")),
    ("Physical activity", ("mets_", "active_minutes", "act_kcal")),
    ("Heart rate", ("hr_",)),
    ("Time of day", ("tod_",)),
    ("Planned meal / activity", ("plan_",)),
    ("Patient profile (EHR)", ("ehr_",)),
]


def feature_group(name: str) -> str:
    for label, prefixes in FEATURE_GROUPS:
        if name.startswith(prefixes) or name in prefixes:
            return label
    return "Other"
