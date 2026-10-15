"""Generate SYNTHETIC data in exactly the CGMacros file format.

Use it to test the whole pipeline (prepare -> train -> dashboard) before or without the real
download. Results on synthetic data are NOT scientific results; never report them.

    python scripts/make_synthetic_data.py --out data/raw/synthetic --patients 45 --days 10

Physiology (deliberately simple): glucose = personal baseline + circadian drift
+ meal responses (carb absorption convolved with a personal response kernel, slowed by fat and
fiber) - activity effect + correlated noise. Heart rate follows activity, sleep and meals.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

GROUPS = ["healthy"] * 15 + ["prediabetes"] * 16 + ["t2d"] * 14


def meal_response(n: int, t: int, carbs: float, fiber: float, fat: float, gain: float, tau_a: float,
                  tau_c: float) -> np.ndarray:
    d = np.arange(0, min(360, n - t), dtype=float)
    ta = tau_a * (1 + 0.012 * fat + 0.02 * fiber)
    k = np.exp(-d / tau_c) - np.exp(-d / ta)
    k = k / k.max() if k.max() > 0 else k
    eff = max(carbs - 0.5 * fiber, 0) * (1 - 0.003 * fat)
    out = np.zeros(n)
    out[t:t + len(d)] = gain * eff * k
    return out


def simulate(pid: int, group: str, days: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    n = days * 1440
    start = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 600)))
    ts = pd.date_range(start, periods=n, freq="1min")
    minute = np.arange(n) % 1440

    prof = {
        "healthy": dict(base=(90, 5), gain=(0.55, 0.12), a1c=(4.8, 5.6), tau_c=(55, 10)),
        "prediabetes": dict(base=(101, 6), gain=(0.85, 0.15), a1c=(5.7, 6.4), tau_c=(75, 12)),
        "t2d": dict(base=(135, 18), gain=(1.35, 0.3), a1c=(6.5, 8.6), tau_c=(100, 18)),
    }[group]
    base = rng.normal(*prof["base"])
    gain = max(rng.normal(*prof["gain"]), 0.25)
    tau_c = max(rng.normal(*prof["tau_c"]), 35)
    tau_a = rng.uniform(22, 35)
    act_sens = rng.uniform(4, 9) * (1.3 if group == "healthy" else 1.0)
    a1c = rng.uniform(*prof["a1c"])

    # activity (METs), sleep 23:00-06:30
    asleep = (minute >= 23 * 60) | (minute < 6 * 60 + 30)
    mets = np.where(asleep, 0.95, 1.25) + rng.normal(0, 0.08, n)
    walk_after = np.zeros(n, bool)
    meals = []
    for d in range(days):
        off = d * 1440
        plan = [("Breakfast", rng.normal(8 * 60, 35), (20, 70), (15, 40), (4, 20), (0, 8)),
                ("Lunch", rng.normal(12.9 * 60, 40), (45, 120), (15, 45), (10, 35), (3, 14)),
                ("Dinner", rng.normal(19.5 * 60, 45), (35, 115), (15, 45), (10, 40), (2, 12))]
        if rng.random() < 0.3:
            plan.append(("Snacks", rng.normal(16.3 * 60, 30), (15, 40), (2, 8), (3, 12), (0, 4)))
        for mtype, tmin, cr, pr, fr, fb in plan:
            t = int(off + np.clip(tmin, 0, 1439))
            c, p, f, fi = (rng.uniform(*cr), rng.uniform(*pr), rng.uniform(*fr), rng.uniform(*fb))
            meals.append((t, mtype, c, p, f, fi, rng.random() > 0.05))  # 5% of meals go unlogged
            if mtype != "Snacks" and rng.random() < 0.35:  # post-meal walk
                w0, wl = t + int(rng.integers(5, 25)), int(rng.integers(10, 35))
                mets[w0:w0 + wl] = rng.uniform(3.0, 4.5)
                walk_after[w0:w0 + wl] = True
        for _ in range(rng.integers(0, 3)):  # other activity
            w0 = off + int(rng.integers(7 * 60, 21 * 60))
            wl = int(rng.integers(10, 50))
            mets[w0:w0 + wl] = rng.uniform(2.5, 6.0)

    # glucose
    g = np.full(n, base, dtype=float)
    g += 6 * np.sin(2 * np.pi * (minute - 5 * 60) / 1440)
    if group == "t2d":
        g += 12 * np.exp(-((minute - 7 * 60) / 70.0) ** 2)  # dawn phenomenon
    for t, _, c, p, f, fi, _ in meals:
        g += meal_response(n, t, c, fi, f, gain, tau_a, tau_c)
    # Muscle glucose uptake during activity scales with the glucose level, so a walk during
    # a post-meal rise lowers glucose more than the same walk while fasting.
    excess = np.clip(mets - 1.3, 0, None)
    kern = np.exp(-np.arange(0, 180) / 50.0)
    g -= act_sens * np.convolve(excess, kern)[:n] / 15.0 * (g / 110.0)
    ar = lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.6, n))  # slowly drifting AR(1) noise
    g += ar * (1.4 if group == "t2d" else 1.0)
    g = np.where(g < 75, 75 - 0.25 * (75 - g), g)  # counter-regulation keeps lows rare
    g = np.clip(g, 45, 380)

    def sensor(period: int, bias: float, sd: float, drop_hours: int) -> np.ndarray:
        idx = np.arange(0, n, period)
        vals = g[idx] * bias + rng.normal(0, sd, len(idx))
        s = pd.Series(np.nan, index=np.arange(n))
        s.iloc[idx] = vals
        s = s.interpolate(limit_area="inside")
        for _ in range(drop_hours):
            a = int(rng.integers(0, n - 240))
            s.iloc[a:a + int(rng.integers(60, 240))] = np.nan
        return np.clip(s.round(0).values, 40, 400)

    dexcom = sensor(5, 1.0, 4.0, int(rng.integers(0, 4)))
    libre = sensor(15, rng.uniform(0.9, 1.0), 6.0, int(rng.integers(0, 3)))
    if rng.random() < 0.1:
        dexcom[:] = np.nan  # some participants have only one working sensor

    rest_hr = rng.uniform(58, 78) + (5 if group == "t2d" else 0)
    hr = rest_hr + 11 * (mets - 1.1) - np.where(asleep, 6, 0) + rng.normal(0, 2.5, n)
    for t, *_ in meals:
        hr[t:t + 90] += 4
    hr = np.round(np.clip(hr, 40, 175), 0)
    watch_off = rng.random(n // 60) < 0.04
    hr[np.repeat(watch_off, 60)[:n]] = np.nan
    act_kcal = np.round(np.clip(mets, 0.9, None) * rng.uniform(1.0, 1.4) * (1.1 if group == "t2d" else 1.0), 3)

    df = pd.DataFrame({"Timestamp": ts, "Libre GL": libre, "Dexcom GL": dexcom, "HR": hr,
                       "Calories (Activity)": act_kcal, "METs": np.round(np.clip(mets, 1, None) * 10, 0),
                       "Meal Type": "", "Calories": np.nan, "Carbs": np.nan, "Protein": np.nan, "Fat": np.nan,
                       "Fiber": np.nan, "Amount Consumed": np.nan, "Image path": ""})
    for t, mtype, c, p, f, fi, logged in meals:
        if not logged:
            continue
        df.loc[t, ["Meal Type", "Image path"]] = [mtype, f"CGMacros-{pid:03d}/photos/{t}.jpg"]
        df.loc[t, ["Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed"]] = [
            round(4 * c + 4 * p + 9 * f), round(c), round(p), round(f), round(fi), 100]

    age = int(np.clip(rng.normal({"healthy": 35, "prediabetes": 46, "t2d": 54}[group], 10), 18, 69))
    bmi = float(np.clip(rng.normal({"healthy": 25, "prediabetes": 29, "t2d": 32}[group], 4), 20.7, 46))
    height = float(rng.uniform(60, 72))
    weight = bmi * (height * 0.0254) ** 2 / 0.453592
    fasting = float(np.clip(base - 5 + rng.normal(0, 5), 79, 218))
    insulin = float(np.clip(rng.lognormal(np.log({"healthy": 6, "prediabetes": 10, "t2d": 14}[group]), 0.4), 2.5, 46))
    tg = float(np.clip(rng.lognormal(np.log(110 + 30 * (group == "t2d")), 0.4), 40, 600))
    hdl = float(np.clip(rng.normal(52 - 5 * (group == "t2d"), 12), 24, 106))
    chol = float(np.clip(rng.normal(185, 35), 91, 345))
    bio = {"subject": pid, "Age": age, "Gender": rng.choice(["F", "M"]), "BMI": round(bmi, 2),
           "Body weight": round(weight, 1), "Height": round(height, 1),
           "Self-identify": rng.choice(["White", "Hispanic/Latino", "African American"]),
           "A1c PDL (Lab)": round(a1c, 1), "Fasting GLU - PDL (Lab)": round(fasting),
           "Insulin": round(insulin, 1), "Triglycerides": round(tg), "Cholesterol": round(chol),
           "HDL": round(hdl), "Non HDL": round(chol - hdl), "LDL (Cal)": round(max(chol - hdl - tg / 5, 21)),
           "VLDL (Cal)": round(tg / 5), "Cho/HDL Ratio": round(chol / hdl, 1),
           "Collection time PDL (Lab)": "08:10", "#1 Contour Fingerstick GLU": round(fasting + rng.normal(0, 6)),
           "Time (t)": "08:15", "#2 Contour Fingerstick GLU": round(fasting + rng.normal(0, 6)),
           "Time (t) ": "08:20", "#3 Contour Fingerstick GLU": round(fasting + rng.normal(0, 6)),
           "Time (t)  ": "08:25"}
    return df, bio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--patients", type=int, default=45)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = [i for i in range(1, 50) if i not in (24, 25, 37, 40)][: args.patients]
    groups = list(GROUPS)
    rng.shuffle(groups)
    groups = (groups * 4)[: len(ids)]
    bios = []
    for pid, grp in zip(ids, groups):
        df, bio = simulate(pid, grp, args.days, rng)
        folder = out / f"CGMacros-{pid:03d}"
        folder.mkdir(exist_ok=True)
        df.insert(0, "", range(len(df)))  # real files may carry an unnamed index column
        df.to_csv(folder / f"CGMacros-{pid:03d}.csv", index=False)
        bios.append(bio)
        print(f"patient {pid:03d} ({grp}) written")
    with open(out / "bio.csv", "w", newline="") as fh:
        header = [k.rstrip() for k in bios[0].keys()]  # duplicate "Time (t)" headers, like the real file
        w = csv.writer(fh)
        w.writerow(header)
        for b in bios:
            w.writerow(list(b.values()))
    (out / "SYNTHETIC_DATA_README.txt").write_text(
        "These files are SYNTHETIC, generated by scripts/make_synthetic_data.py for pipeline testing.\n"
        "Do not report any result obtained on them.\n")
    print(f"done: {len(ids)} synthetic patients in {out}")


if __name__ == "__main__":
    main()
