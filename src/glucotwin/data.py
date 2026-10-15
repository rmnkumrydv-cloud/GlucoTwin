"""Load CGMacros and turn it into two separate streams.

* Dynamic / wearable stream: CGM glucose, Fitbit heart rate and METs, logged meals,
  resampled onto a regular 5-minute grid per patient.
* Static / EHR inputs: bio.csv (demographics + fasting labs). These are handed to
  ``ehr.py`` which wraps them into a FHIR-style record; the model only ever reads
  static features back out of that record.

The loader is deliberately tolerant about column names (case, spacing, "METs" vs
"Mets", a leading unnamed index column) because small formatting differences are
the most common reason pipelines break on a real dataset.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- columns
_MAIN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime"],
    "libre": ["libre gl", "libre", "libre_gl"],
    "dexcom": ["dexcom gl", "dexcom", "dexcom_gl"],
    "hr": ["hr", "heart rate", "heartrate"],
    "act_kcal": ["calories (activity)", "activity calories", "calories_activity"],
    "mets": ["mets", "met", "mets x10"],
    "meal_type": ["meal type", "meal_type"],
    "meal_kcal": ["calories", "meal calories"],
    "carbs": ["carbs", "carbohydrates", "carb"],
    "protein": ["protein"],
    "fat": ["fat"],
    "fiber": ["fiber", "fibre"],
    "amount_consumed": ["amount consumed", "amount_consumed"],
    "image_path": ["image path", "image_path"],
}

_BIO_ALIASES = {
    "pid": ["subject", "subject id", "participant", "participant id", "id", "pid"],
    "age": ["age"],
    "sex": ["gender", "sex"],
    "bmi": ["bmi"],
    "weight_lb": ["body weight", "weight"],
    "height_in": ["height"],
    "ethnicity": ["self-identify", "ethnicity", "race"],
    "hba1c": ["a1c pdl (lab)", "a1c", "hba1c"],
    "fasting_glucose": ["fasting glu - pdl (lab)", "fasting glu", "fasting glucose"],
    "fasting_insulin": ["insulin"],
    "triglycerides": ["triglycerides"],
    "cholesterol": ["cholesterol"],
    "hdl": ["hdl"],
    "non_hdl": ["non hdl"],
    "ldl": ["ldl (cal)", "ldl"],
    "vldl": ["vldl (cal)", "vldl"],
    "chol_hdl_ratio": ["cho/hdl ratio"],
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _rename(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    lookup = {_norm(c): c for c in df.columns}
    mapping = {}
    for std, options in aliases.items():
        for opt in options:
            if opt in lookup and lookup[opt] not in mapping:
                mapping[lookup[opt]] = std
                break
    return df.rename(columns=mapping)


# ----------------------------------------------------------------------------- discovery
def find_dataset_files(raw_dir: Path) -> tuple[Path | None, list[Path]]:
    """Find bio.csv and the per-participant CSVs anywhere below ``raw_dir``."""
    raw_dir = Path(raw_dir)
    bio = sorted(p for p in raw_dir.rglob("*.csv") if p.name.lower() == "bio.csv")
    parts = sorted(
        p for p in raw_dir.rglob("CGMacros-*.csv")
        if "datadictionary" not in p.name.lower() and "__macosx" not in str(p).lower()
    )
    return (bio[0] if bio else None), parts


def pid_from_path(path: Path) -> int:
    m = re.search(r"CGMacros-0*(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse participant id from {path}")
    return int(m.group(1))


# ----------------------------------------------------------------------------- participant files
def load_participant(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]
    df = _rename(df, _MAIN_ALIASES)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: no timestamp column found; columns = {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    n_dup = int(df["timestamp"].duplicated().sum())
    for col in ["libre", "dexcom", "hr", "act_kcal", "mets", "meal_kcal", "carbs", "protein",
                "fat", "fiber", "amount_consumed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    if "meal_type" not in df.columns:
        df["meal_type"] = np.nan
    df = df.reset_index(drop=True)
    df.attrs["n_duplicates"] = n_dup
    return df


def extract_meals(raw: pd.DataFrame, scale_by_amount: bool = False) -> pd.DataFrame:
    """One row per logged meal: timestamp + macronutrients."""
    macro_cols = ["meal_kcal", "carbs", "protein", "fat", "fiber"]
    has_macro = raw[macro_cols].notna().any(axis=1)
    has_type = raw["meal_type"].notna() & (raw["meal_type"].astype(str).str.strip() != "")
    meals = raw.loc[has_macro | has_type, ["timestamp", "meal_type", *macro_cols, "amount_consumed"]].copy()
    if meals.empty:
        return pd.DataFrame(columns=["timestamp", "meal_type", *macro_cols])
    meals[macro_cols] = meals[macro_cols].fillna(0.0)
    if scale_by_amount:
        frac = (meals["amount_consumed"].fillna(100.0) / 100.0).clip(0, 1)
        meals[macro_cols] = meals[macro_cols].mul(frac, axis=0)
    # Photos are logged before *and* after a meal: merge rows closer than 10 minutes.
    meals = meals.sort_values("timestamp")
    new_group = meals["timestamp"].diff().gt(pd.Timedelta(minutes=10)) | meals["timestamp"].diff().isna()
    meals["grp"] = new_group.cumsum()
    agg = meals.groupby("grp").agg(
        timestamp=("timestamp", "first"),
        meal_type=("meal_type", "first"),
        meal_kcal=("meal_kcal", "max"),
        carbs=("carbs", "max"),
        protein=("protein", "max"),
        fat=("fat", "max"),
        fiber=("fiber", "max"),
    )
    return agg.reset_index(drop=True)


def choose_cgm(raw: pd.DataFrame, preference: str = "auto") -> str:
    cov = {s: raw[s].notna().mean() for s in ("dexcom", "libre")}
    if preference in cov and cov[preference] > 0.2:
        return preference
    # Prefer Dexcom (5-min native sampling) unless its coverage is clearly worse.
    return "dexcom" if cov["dexcom"] >= 0.7 * cov["libre"] and cov["dexcom"] > 0 else "libre"


SENSOR_PERIOD_MIN = {"dexcom": 5, "libre": 15}


def true_reading_mask(x: np.ndarray, period: int, block: int = 720) -> tuple[np.ndarray, float]:
    """Find the ORIGINAL sensor readings inside a CGM series that was linearly interpolated to 1 min.

    CGMacros interpolated Dexcom (5 min) and Libre (15 min) readings to 1-minute rows. An
    interpolated value between two readings is partly built from the NEXT reading, i.e. from the
    future. We therefore keep only the original readings. For each block (12 h by default) of each
    gap-free segment we choose the sampling phase whose samples best reconstruct the block by
    linear interpolation; for truly interpolated data that error is ~0 at the right phase.
    Returns (mask, fraction of blocks where the best phase was clearly identified).
    """
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return mask, 0.0
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    clear, total = 0, 0
    for a, b in zip(edges[::2], edges[1::2]):
        seg = x[a:b]
        for s0 in range(0, len(seg), block):
            blk = seg[s0:s0 + block + period]
            if len(blk) < 3 * period:
                mask[a + s0:a + s0 + len(blk):period] = True
                continue
            errs = []
            for phi in range(period):
                idx = np.arange(phi, len(blk), period)
                rec = np.interp(np.arange(len(blk)), idx, blk[idx])
                inner = slice(idx[0], idx[-1] + 1)
                errs.append(float(np.mean(np.abs(blk[inner] - rec[inner]))))
            best = int(np.argmin(errs))
            total += 1
            # Neighbouring phases reconstruct a smooth curve almost as well, so compare the best
            # phase with the typical (median) phase rather than the runner-up.
            clear += int(np.median(errs) > 2.0 * errs[best] + 0.05)
            upto = min(block, len(seg) - s0)
            mask[a + s0 + best:a + s0 + upto:period] = True
    return mask, (clear / total if total else 0.0)


def build_stream(raw: pd.DataFrame, cfg: dict, pid: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Resample one participant onto a CAUSAL regular grid. Returns (stream, meals, info).

    The row labelled t only contains information available at time t:
    * glucose = the latest ORIGINAL sensor reading in (t - step, t], held forward for at most
      ``max_hold_minutes`` (a hold is causal; interpolation is not);
    * heart rate / METs = mean over (t - step, t];
    * a meal logged at 12:03 appears at the 12:05 row.
    """
    dcfg = cfg["data"]
    step = int(dcfg["resample_minutes"])
    sensor = choose_cgm(raw, dcfg["primary_cgm"])
    n_dup = int(raw.attrs.get("n_duplicates", 0))
    raw = raw.drop_duplicates("timestamp", keep="first")
    x = raw.set_index("timestamp").sort_index()
    rule = f"{step}min"

    # --- original CGM readings only
    g1 = x[sensor].astype(float)
    period = SENSOR_PERIOD_MIN[sensor]
    if dcfg.get("use_true_readings", True):
        # put the series on an exact 1-minute index so the phase search works on gaps too
        full = g1.resample("1min").mean()
        m, clarity = true_reading_mask(full.to_numpy(), period)
        readings = full[m].dropna()
    else:
        readings, clarity = g1.dropna(), float("nan")
    readings = readings[readings.between(39, 401)]
    rd = pd.DataFrame({"g": readings, "t_read": readings.index}, index=readings.index)
    binned = rd.resample(rule, label="right", closed="right").last()

    start = x.index.min().ceil(rule)
    end = x.index.max().floor(rule)
    grid = pd.date_range(start, end, freq=rule)
    binned = binned.reindex(grid)
    hold = max(int(dcfg.get("max_hold_minutes", 10) // step), 0)
    if sensor == "libre":
        hold = max(hold, int(15 // step))
    binned = binned.ffill(limit=hold) if hold else binned
    age = (binned.index.to_series() - binned["t_read"]) / pd.Timedelta(minutes=1)

    def causal_mean(col: str, lo=None, hi=None, scale: float = 1.0) -> pd.Series:
        v = x[col].astype(float) / scale
        if lo is not None:
            v = v.where(v.between(lo, hi))
        return v.resample(rule, label="right", closed="right").mean().reindex(grid)

    hr = causal_mean("hr", 30, 220)
    mets = causal_mean("mets", scale=float(dcfg["mets_scale"]))
    act = x["act_kcal"].resample(rule, label="right", closed="right").sum(min_count=1).reindex(grid)

    meals = extract_meals(raw, dcfg["scale_by_amount_consumed"])
    stream = pd.DataFrame({"timestamp": grid, "glucose": binned["g"].to_numpy(),
                           "glucose_age_min": age.to_numpy(), "hr": hr.to_numpy(),
                           "mets": mets.to_numpy(), "act_kcal": act.to_numpy()})
    for col in ["carbs", "protein", "fat", "fiber", "meal_kcal"]:
        stream[col] = 0.0
    if not meals.empty:
        bins = meals["timestamp"].dt.ceil(rule)  # usable from the first grid time AFTER the log
        m = meals.assign(bin=bins).groupby("bin")[["carbs", "protein", "fat", "fiber", "meal_kcal"]].sum()
        m = m.reindex(pd.DatetimeIndex(grid)).fillna(0.0)
        for col in m.columns:
            stream[col] = m[col].to_numpy()
    stream.insert(0, "pid", pid)

    # --- data-quality report
    gaps = readings.index.to_series().diff() / pd.Timedelta(minutes=1)
    both = x[["dexcom", "libre"]].dropna()
    agree = float((np.abs(both["libre"] - both["dexcom"]) / both["dexcom"].clip(lower=1)).mean() * 100) \
        if len(both) > 100 else float("nan")
    info = {
        "pid": pid, "sensor": sensor, "days": round((end - start) / pd.Timedelta(days=1), 2),
        "true_readings": int(len(readings)),
        "phase_clarity": round(clarity, 3) if clarity == clarity else None,
        "glucose_coverage": round(float(stream["glucose"].notna().mean()), 3),
        "gaps_over_30min": int((gaps > 30).sum()),
        "longest_gap_min": float(gaps.max()) if len(gaps) > 1 else None,
        "dexcom_vs_libre_mard_pct": round(agree, 1) if agree == agree else None,
        "hr_coverage": round(float(stream["hr"].notna().mean()), 3),
        "duplicate_timestamps": n_dup,
        "n_meals": int(len(meals)),
    }
    meals = meals.assign(pid=pid)
    return stream, meals, info


# ----------------------------------------------------------------------------- bio / static
def load_bio(path: Path, participant_ids: list[int] | None = None) -> pd.DataFrame:
    bio = pd.read_csv(path)
    bio = bio.loc[:, [c for c in bio.columns if not str(c).lower().startswith("unnamed")]]
    bio = _rename(bio, _BIO_ALIASES)
    if "pid" not in bio.columns:
        if participant_ids is not None and len(participant_ids) == len(bio):
            log.warning("bio.csv has no subject column; assuming rows follow sorted participant ids.")
            bio.insert(0, "pid", sorted(participant_ids))
        else:
            raise ValueError("bio.csv has no subject/participant id column. Add one mapping rows to "
                             "participant numbers (see docs/DATA.md).")
    bio["pid"] = bio["pid"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    num_cols = [c for c in _BIO_ALIASES if c not in ("pid", "sex", "ethnicity") and c in bio.columns]
    for c in num_cols:
        bio[c] = pd.to_numeric(bio[c], errors="coerce")
    # Known data-entry sentinels documented in the CGMacros data dictionary.
    if "ldl" in bio:
        bio.loc[bio["ldl"] >= 800, "ldl"] = np.nan
    if "vldl" in bio:
        bio.loc[bio["vldl"] >= 400, "vldl"] = np.nan
    if "chol_hdl_ratio" in bio:
        bio.loc[bio["chol_hdl_ratio"] > 30, "chol_hdl_ratio"] = np.nan
    if "hba1c" in bio:  # dictionary says mmol/mol but values look like %; convert if needed
        mmol = bio["hba1c"] > 20
        bio.loc[mmol, "hba1c"] = bio.loc[mmol, "hba1c"] / 10.929 + 2.15
    if "sex" in bio:
        bio["sex"] = bio["sex"].astype(str).str.upper().str[0].map({"M": "male", "F": "female"})
    keep = ["pid"] + [c for c in _BIO_ALIASES if c != "pid" and c in bio.columns]
    return bio[keep].drop_duplicates("pid").sort_values("pid").reset_index(drop=True)


# ----------------------------------------------------------------------------- orchestration
def load_cgmacros(cfg: dict) -> dict:
    raw_dir = cfg.path("raw_dir") if hasattr(cfg, "path") else Path(cfg["paths"]["raw_dir"])
    bio_path, part_paths = find_dataset_files(raw_dir)
    if not part_paths:
        raise FileNotFoundError(
            f"No CGMacros-XXX.csv files under {raw_dir}. Run `python scripts/download_data.py` "
            "(real data) or `python scripts/make_synthetic_data.py --out data/raw/synthetic` (demo data).")
    if bio_path is None:
        raise FileNotFoundError(f"bio.csv not found under {raw_dir}.")
    pids = [pid_from_path(p) for p in part_paths]
    dup = sorted({x for x in pids if pids.count(x) > 1})
    if dup:
        raise ValueError(f"Participant ids {dup} appear more than once under {raw_dir}. Did you mix real "
                         "and synthetic data? Point paths.raw_dir (or --raw) at one dataset only.")
    streams, meals, infos = [], [], []
    for p in part_paths:
        pid = pid_from_path(p)
        raw = load_participant(p)
        s, m, info = build_stream(raw, cfg, pid)
        streams.append(s)
        meals.append(m)
        infos.append(info)
        log.info("patient %03d: %s days, glucose coverage %.0f%%, %d meals", pid, info["days"],
                 100 * info["glucose_coverage"], info["n_meals"])
    ids = [i["pid"] for i in infos]
    bio = load_bio(bio_path, ids)
    bio = bio[bio["pid"].isin(ids)].reset_index(drop=True)
    return {
        "streams": pd.concat(streams, ignore_index=True),
        "meals": pd.concat(meals, ignore_index=True),
        "bio": bio,
        "info": pd.DataFrame(infos),
    }
