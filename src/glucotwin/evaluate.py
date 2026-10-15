"""Evaluation metrics and figures (all computed on patients the model never saw)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

INK = "#1B2A41"
TEAL = "#0F766E"
AMBER = "#B45309"
RED = "#B42318"
GREY = "#94A3B8"


# ============================================================================ regression
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, g_now: np.ndarray | None = None) -> dict:
    """Error metrics on absolute glucose.

    R2 on absolute glucose looks excellent even for a useless model (persistence scores ~0.9 at
    30 min because glucose changes slowly), so when ``g_now`` is given we also report R2 of the
    predicted CHANGE (``r2_delta``), which is the honest measure of forecasting skill.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if g_now is not None:
        g_now = np.asarray(g_now, float)
        m &= ~np.isnan(g_now)
    yt, yp = y_true[m], y_pred[m]
    err = yp - yt
    zones = clarke_zones(yt, yp)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    out = {
        "n": int(m.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mard_pct": float(np.mean(np.abs(err) / np.clip(yt, 1, None)) * 100),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else float("nan"),
        "clarke_A_pct": zones["A"],
        "clarke_AB_pct": zones["A"] + zones["B"],
        "clarke_DE_pct": zones["D"] + zones["E"],
    }
    if g_now is not None:
        dt, dp = yt - g_now[m], yp - g_now[m]
        ss_d = np.sum((dt - dt.mean()) ** 2)
        out["r2_delta"] = float(1 - np.sum((dp - dt) ** 2) / ss_d) if ss_d > 0 else float("nan")
    return out


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke Error Grid zone (Clarke et al., Diabetes Care 1987)."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or \
            (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> dict:
    z = pd.Series([clarke_zone(r, p) for r, p in zip(ref, pred)])
    pct = (z.value_counts(normalize=True) * 100).to_dict()
    return {k: float(pct.get(k, 0.0)) for k in "ABCDE"}


# ============================================================================ classification
def classification_metrics(y: np.ndarray, p: np.ndarray, thr) -> dict:
    """thr may be a scalar or a per-row array (each CV fold picks its own threshold)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    thr = np.broadcast_to(np.asarray(thr, float), y.shape)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p, thr = y[m], p[m], thr[m]
    if len(np.unique(y)) < 2:
        return {"n": int(m.sum()), "positives": int(y.sum())}
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": int(m.sum()), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_threshold": float(np.mean(thr)), "precision": prec, "recall": rec,
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
    }


def event_level_metrics(df: pd.DataFrame, alert_col: str, spike_mgdl: float = 180,
                        lookback_min: int = 120, step: int = 5) -> dict:
    """Clinically meaningful alert metrics.

    An *event* is a crossing of ``spike_mgdl`` after at least 30 min below it. It counts as caught if an alert
    fired within the ``lookback_min`` before the crossing. Lead time is the time between the
    first such alert and the crossing. False alerts are alert episodes that are not followed
    by a crossing within ``lookback_min``.
    """
    caught, leads, n_events, false_alerts, days, alert_steps = 0, [], 0, 0, 0.0, 0
    k = int(lookback_min / step)
    for _, d in df.sort_values("timestamp").groupby("pid"):
        d = d.set_index("timestamp")[["glucose", alert_col]]
        d = d[~d.index.duplicated()].asfreq(f"{step}min")
        g = d["glucose"]
        alert = d[alert_col].astype("boolean").fillna(False).to_numpy(dtype=bool)
        below_30 = (g < spike_mgdl).astype(float).rolling(int(30 / step), min_periods=int(30 / step)).min()
        cross = ((g >= spike_mgdl) & (below_30.shift(1) == 1)).fillna(False).to_numpy(dtype=bool)
        days += len(d) * step / 1440
        alert_steps += int(alert.sum())
        idx = np.where(cross)[0]
        n_events += len(idx)
        for i in idx:
            window = alert[max(0, i - k):i]
            if window.any():
                caught += 1
                first = int(np.argmax(window))
                leads.append((len(window) - first) * step)
        starts = np.where(alert & ~np.r_[False, alert[:-1]])[0]
        for s0 in starts:
            if not cross[s0:s0 + k + 1].any():
                false_alerts += 1
    return {
        "events": n_events,
        "event_recall": caught / max(n_events, 1),
        "median_lead_time_min": float(np.median(leads)) if leads else None,
        "false_alerts_per_day": false_alerts / max(days, 1e-9),
        "alert_on_time_pct": 100 * alert_steps * step / 1440 / max(days, 1e-9),
    }


# ============================================================================ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def plot_clarke(ref: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(ref, pred, s=3, alpha=0.25, color=TEAL, linewidths=0)
    ax.plot([0, 400], [0, 400], color=GREY, lw=0.8, ls=":")
    lines = [([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]), ([70, 70], [84, 400]),
             ([0, 70], [180, 180]), ([70, 290], [180, 400]), ([70, 70], [0, 56]), ([70, 400], [56, 320]),
             ([180, 180], [0, 70]), ([180, 400], [70, 70]), ([240, 240], [70, 180]), ([240, 400], [180, 180]),
             ([130, 180], [0, 70])]
    for x, y in lines:
        ax.plot(x, y, color=INK, lw=0.9)
    for label, (x, y) in {"A": (30, 15), "B": (370, 260), "B ": (280, 370), "C": (160, 370),
                          "C ": (160, 15), "D": (30, 140), "D ": (370, 120), "E": (30, 370), "E ": (370, 15)}.items():
        ax.text(x, y, label.strip(), fontsize=13, color=INK, fontweight="bold")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Measured glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rmse_by_horizon(table: pd.DataFrame, path: Path, who: str = "unseen patients") -> None:
    """table: index = model name, columns = horizons, values = RMSE."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {0: GREY, 1: "#CBD5E1", 2: AMBER, 3: TEAL}
    width = 0.8 / len(table)
    x = np.arange(len(table.columns))
    for i, (name, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width, row.values, width, label=name, color=colors.get(i, INK))
    ax.set_xticks(x + width * (len(table) - 1) / 2)
    ax.set_xticklabels([f"{c} min" for c in table.columns])
    ax.set_ylabel("RMSE (mg/dL), lower is better")
    ax.set_title(f"Forecast error ({who})", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_personalization(before: dict, after: dict, path: Path) -> None:
    hs = list(before)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hs, [before[h] for h in hs], marker="o", color=GREY, label="Population model")
    ax.plot(hs, [after[h] for h in hs], marker="o", color=TEAL, label="Personalized twin")
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Calibrating the twin on a patient's first days", color=INK, loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(imp: pd.Series, path: Path) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color=TEAL)
    ax.set_xlabel("Mean |contribution| to 60-min forecast (mg/dL)")
    ax.set_title("What drives the twin's predictions", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example(d: pd.DataFrame, pred_col: str, lo_col: str, hi_col: str, h: int, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhspan(70, 180, color=TEAL, alpha=0.07)
    ax.plot(d["timestamp"], d["glucose"], color=INK, lw=1.4, label="Measured glucose")
    shifted = d["timestamp"] + pd.Timedelta(minutes=h)
    ax.plot(shifted, d[pred_col], color=AMBER, lw=1.2, ls="--", label=f"Forecast made {h} min earlier")
    ax.fill_between(shifted, d[lo_col], d[hi_col], color=AMBER, alpha=0.15, lw=0)
    meals = d[d["carbs_0_30"].diff() > 0]
    for t in meals["timestamp"]:
        ax.axvline(t, color=GREY, lw=0.6, ls=":")
    ax.set_ylabel("mg/dL")
    ax.set_title(title, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(y: np.ndarray, p: np.ndarray, path: Path, bins: int = 10) -> None:
    """Reliability diagram: does a predicted 70% risk come true about 70% of the time?"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    mp = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    my = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    ax.plot(mp, my, marker="o", color=TEAL)
    ax.set_xlabel("Predicted spike risk")
    ax.set_ylabel("Observed spike rate")
    ax.set_title("Calibration of spike risk (unseen patients)", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bootstrap_ci(frame: pd.DataFrame, horizons: list[int], n_boot: int = 1000, seed: int = 0,
                 pred_fmt: str = "pred_{h}") -> dict:
    """95% confidence intervals by resampling PATIENTS (not rows), the right unit of uncertainty
    when rows from one person are strongly correlated."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(frame["pid"].unique()))
    P = len(pids)
    if P < 3:
        return {}
    draws = rng.integers(0, P, size=(n_boot, P))
    W = np.stack([np.bincount(r, minlength=P) for r in draws])  # times each patient is drawn
    out: dict = {"rmse": {}, "n_patients": P}
    for h in horizons:
        y = frame["glucose"] + frame[f"target_delta_{h}"]
        e2 = (frame[pred_fmt.format(h=h)] - y) ** 2
        g = pd.DataFrame({"pid": frame["pid"], "e2": e2}).dropna().groupby("pid")["e2"].agg(["sum", "count"])
        g = g.reindex(pids).fillna(0.0)
        rmse = np.sqrt((W @ g["sum"].to_numpy()) / np.clip(W @ g["count"].to_numpy(), 1, None))
        out["rmse"][h] = (float(np.percentile(rmse, 2.5)), float(np.percentile(rmse, 97.5)))
    if "p_spike" in frame and frame["p_spike"].notna().any():
        d = frame[["pid", "label_spike", "p_spike"]].dropna()
        by_pid = {p: g for p, g in d.groupby("pid")}
        aucs = []
        for r in draws[: min(n_boot, 300)]:
            b = pd.concat([by_pid[pids[i]] for i in r if pids[i] in by_pid])
            if b["label_spike"].nunique() == 2:
                aucs.append(roc_auc_score(b["label_spike"], b["p_spike"]))
        if aucs:
            out["spike_auroc"] = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    return out
