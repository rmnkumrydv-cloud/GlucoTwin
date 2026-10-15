"""Check that logged meal times line up with the glucose rises they cause.

    python scripts/check_meal_alignment.py          # after python -m glucotwin.prepare

Why: if meal timestamps are shifted (logged late, or a clock offset between the phone photos and
the CGM), meal features carry little information and the what-if simulator responds weakly. A
normal mixed meal starts raising glucose about 5 to 30 minutes after the first bite and peaks
about 30 to 90 minutes later.

For every meal with at least 20 g of carbohydrate and no other meal in the 2 hours before or
3 hours after it, the glucose curve from 45 min before to 3 h after is taken relative to the
level 30 to 45 min before the meal. Per patient, the median curve gives the rise onset and peak.

Outputs: artifacts/meal_alignment.png (median response by metabolic group: a good slide) and
artifacts/meal_alignment.csv (one row per patient). Uses development and test patients alike,
but only to check timestamps: nothing here is used for training.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from glucotwin.config import load_config  # noqa: E402

OFFSETS = np.arange(-45, 181, 5)
COLORS = {"healthy": "#0F766E", "prediabetes": "#B45309", "t2d": "#B42318"}
NAMES = {"healthy": "No diabetes", "prediabetes": "Prediabetes", "t2d": "Type 2 diabetes"}


def meal_curves(streams: pd.DataFrame, meals: pd.DataFrame, min_carbs: float) -> pd.DataFrame:
    rows = []
    for pid, m in meals.groupby("pid"):
        s = streams[streams["pid"] == pid].set_index("timestamp")["glucose"]
        m = m.sort_values("timestamp").reset_index(drop=True)
        gap_before = m["timestamp"].diff()            # time since the previous logged meal
        gap_after = -m["timestamp"].diff(-1)          # time until the next logged meal
        for i, meal in m.iterrows():
            if meal["carbs"] < min_carbs:
                continue
            if (pd.notna(gap_before[i]) and gap_before[i] < pd.Timedelta(hours=2)) or \
                    (pd.notna(gap_after[i]) and gap_after[i] < pd.Timedelta(hours=3)):
                continue
            t0 = meal["timestamp"].ceil("5min")
            g = s.reindex(t0 + pd.to_timedelta(OFFSETS, unit="min")).to_numpy(float)
            if np.isnan(g).mean() > 0.2:
                continue
            pre = g[(OFFSETS >= -45) & (OFFSETS <= -30)]
            if np.isnan(pre).all():
                continue
            base = float(np.nanmean(pre))
            rows.append({"pid": pid, "meal_type": str(meal.get("meal_type", "")), "carbs": meal["carbs"],
                         **{int(o): v - base for o, v in zip(OFFSETS, g)}})
    return pd.DataFrame(rows)


def summarize_patient(curve: np.ndarray) -> dict:
    after = OFFSETS >= 0
    peak = float(np.nanmax(curve[after]))
    t_peak = int(OFFSETS[after][int(np.nanargmax(curve[after]))])
    onset = None
    if peak >= 15:
        idx = np.where((OFFSETS >= -30) & (curve >= 0.2 * peak))[0]
        onset = int(OFFSETS[idx[0]]) if len(idx) else None
    if peak < 15:
        verdict = "no clear rise after meals (logging or timing problem, or very flat responder)"
    elif onset is not None and onset < -5:
        verdict = "rise starts BEFORE the logged time (meals logged late?)"
    elif onset is not None and onset > 45:
        verdict = "rise starts late (meals logged early, or slow responder)"
    else:
        verdict = "aligned"
    return {"peak_rise_mgdl": round(peak, 1), "time_to_peak_min": t_peak, "onset_min": onset, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-carbs", type=float, default=20.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc, art = cfg.path("processed_dir"), cfg.path("artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    streams = pd.read_parquet(proc / "wearable_stream.parquet")
    meals = pd.read_parquet(proc / "meals.parquet")
    groups = json.loads((proc / "splits.json").read_text())["groups"]

    curves = meal_curves(streams, meals, args.min_carbs)
    if curves.empty:
        raise SystemExit("No isolated meals with enough CGM data were found.")
    cols = [int(o) for o in OFFSETS]
    curves["group"] = curves["pid"].astype(str).map(groups)

    per_patient = []
    for pid, c in curves.groupby("pid"):
        med = c[cols].median().to_numpy(float)
        per_patient.append({"pid": int(pid), "group": groups.get(str(pid), "unknown"), "meals_used": len(c),
                            **summarize_patient(med)})
    table = pd.DataFrame(per_patient)
    table.to_csv(art / "meal_alignment.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g, c in curves.groupby("group"):
        med = c[cols].median()
        q1, q3 = c[cols].quantile(0.25), c[cols].quantile(0.75)
        ax.plot(OFFSETS, med, color=COLORS.get(g, "#64748B"), lw=2,
                label=f"{NAMES.get(g, g)} ({c['pid'].nunique()} people, {len(c)} meals)")
        ax.fill_between(OFFSETS, q1, q3, color=COLORS.get(g, "#64748B"), alpha=0.12, lw=0)
    ax.axvline(0, color="#1B2A41", lw=1, ls=":")
    ax.axhline(0, color="#94A3B8", lw=0.8)
    ax.set_xlabel("Minutes after the logged meal time")
    ax.set_ylabel("Glucose change (mg/dL)")
    ax.set_title("Median glucose response to a meal (shaded: middle 50% of meals)", loc="left", color="#1B2A41")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(art / "meal_alignment.png", dpi=160)

    overall = curves[cols].median().to_numpy(float)
    summ = summarize_patient(overall)
    counts = table["verdict"].value_counts()
    print(f"Meals used: {len(curves)} from {curves['pid'].nunique()} patients (>= {args.min_carbs:.0f} g carbs, "
          "no other meal 2 h before / 3 h after)")
    print(f"All patients together: rise starts at {summ['onset_min']} min, peaks at {summ['time_to_peak_min']} min, "
          f"median peak rise {summ['peak_rise_mgdl']} mg/dL")
    print("\nPer group (median of meals):")
    for g, c in curves.groupby("group"):
        sg = summarize_patient(c[cols].median().to_numpy(float))
        print(f"  {NAMES.get(g, g):16s} onset {sg['onset_min']} min, peak {sg['time_to_peak_min']} min, "
              f"rise {sg['peak_rise_mgdl']} mg/dL")
    print("\nPatients by verdict:")
    for v, n in counts.items():
        ids = table.loc[table["verdict"] == v, "pid"].tolist()
        print(f"  {n:2d}  {v}: {ids if v != 'aligned' else ''}")
    aligned = counts.get("aligned", 0) / len(table)
    onset_ok = summ["onset_min"] is not None and -5 <= summ["onset_min"] <= 30
    if aligned >= 0.75 and onset_ok:
        print("\nVERDICT: meal times look aligned with glucose rises.")
    else:
        msg = "\nVERDICT: meal timing looks off for many patients; send this output for review."
        if summ["onset_min"] is not None:
            lag = 10 - summ["onset_min"]   # a typical rise starts ~10 min after the first bite
            msg += f" Rough estimate: meals are logged about {abs(lag)} min {'late' if lag > 0 else 'early'}."
        print(msg)
    print(f"Saved {art / 'meal_alignment.png'} and {art / 'meal_alignment.csv'}")


if __name__ == "__main__":
    main()
