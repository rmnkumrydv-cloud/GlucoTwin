"""Neural track:  python -m glucotwin.nn [--epochs 15] [--quick]

P-MRGF, Personalized Multi-Residual Gated Forecaster (research model, forecasting only):

* CGM backbone: causal dilated TCN over the last 6 h of glucose, conditioned on the patient's
  health record through FiLM (h' = gamma(z) * h + beta(z)), plus an MLP over engineered glucose
  features (the strongest backbone in recent work). It outputs a full 24-step (2 h) trajectory.
* Meal, activity and heart-rate branches each add a residual correction with a learned per-step
  gate: G = G_base + a*R_meal + b*R_activity + c*R_hr. Residual heads start at zero, so the model
  begins as the CGM-only forecaster and only uses a modality when it helps.
* Heads: quantile trajectory (10/50/90%), spike-risk logit (multi-task).
* Loss: zone-weighted Huber (q50) + pinball (q10, q90) + velocity (trajectory slope) + BCE (risk).

Uses only DEVELOPMENT patients and the same leave-patients-out folds as the LightGBM model, so
the two can be ensembled honestly (see ensemble.py). After cross-validation one network is trained
on all development patients (models/nn_dev.pt); ``python -m glucotwin.test`` scores it once on the
locked test patients. Normalisation statistics are computed on training rows only.
The LightGBM model stays the what-if engine because its monotone constraints guarantee
physiologically sensible counterfactuals; this network does not.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import evaluate as ev
from .config import load_config
from .features import feature_columns, patient_features
from .splits import load_splits

log = logging.getLogger("glucotwin.nn")

MEAL_CH = ["carbs", "protein", "fat", "fiber", "cob_g", "carb_rate", "fat_on_board"]
ACT_CH = ["mets", "act_kcal"]
HR_CH = ["hr_rel_rest", "hr_slope_30"]
QUANTILES = (0.1, 0.5, 0.9)


# ============================================================================ data
def build_arrays(cfg) -> dict:
    """Full-grid per-step channels, engineered features and targets for every patient."""
    proc = cfg.path("processed_dir")
    streams = pd.read_parquet(proc / "wearable_stream.parquet")
    static = pd.read_csv(proc / "static_features.csv", index_col=0)
    ds_cols = feature_columns(pd.read_parquet(proc / "dataset.parquet").head(1),
                              cfg["features"].get("exclude_prefixes"))
    eng_cols = [c for c in ds_cols if c.startswith(("glucose", "time_above"))]
    ehr_cols = [c for c in static.columns if c in ds_cols]
    step = int(cfg["data"]["resample_minutes"])
    L = int(cfg["nn"]["context_min"] // step)
    H = int(cfg["nn"]["horizon_min"] // step)
    parts, offset = [], 0
    for pid, s in streams.groupby("pid", sort=True):
        f = patient_features(s, cfg, static.loc[pid].to_dict() if pid in static.index else {}, with_targets=True)
        f = f.reset_index(drop=True)
        n = len(f)
        pad_front, pad_back = L, H
        g = f["glucose"].to_numpy(float)
        parts.append({"pid": pid, "frame": f, "raw": s.sort_values("timestamp").reset_index(drop=True),
                      "g": g, "n": n, "start": offset + pad_front})
        offset += pad_front + n + pad_back
    total = offset
    C = {"cgm": 4, "meal": len(MEAL_CH), "act": len(ACT_CH) + 1, "hr": len(HR_CH) + 1}
    seq = {k: np.zeros((total, c), np.float32) for k, c in C.items()}
    gl = np.full(total, np.nan, np.float32)
    eng = np.zeros((total, len(eng_cols)), np.float32)
    ehr = np.zeros((total, len(ehr_cols)), np.float32)
    rows = []
    for p in parts:
        f, a, n = p["frame"], p["start"], p["n"]
        g = p["g"]
        sl = slice(a, a + n)
        gl[sl] = g
        seq["cgm"][sl, 0] = np.nan_to_num((g - 140.0) / 50.0)
        seq["cgm"][sl, 1] = np.isnan(g)
        seq["cgm"][sl, 2] = f["tod_sin"].to_numpy()
        seq["cgm"][sl, 3] = f["tod_cos"].to_numpy()
        for i, c in enumerate(MEAL_CH):
            v = f[c].to_numpy(float) if c in f else p["raw"][c].to_numpy(float)
            seq["meal"][sl, i] = np.log1p(np.clip(np.nan_to_num(v), 0, None) * (10 if c == "carb_rate" else 1))
        mets = f["mets_mean_15"].to_numpy(float) if "mets_mean_15" in f else np.full(n, np.nan)
        seq["act"][sl, 0] = np.nan_to_num(mets - 1.0)
        seq["act"][sl, 1] = np.log1p(np.nan_to_num(f["act_kcal_60"].to_numpy(float)))
        seq["act"][sl, 2] = np.isnan(mets)
        for i, c in enumerate(HR_CH):
            seq["hr"][sl, i] = np.nan_to_num(f[c].to_numpy(float) / 20.0)
        seq["hr"][sl, 2] = f["hr_missing_60"].to_numpy(float)
        eng[sl] = f.reindex(columns=eng_cols).to_numpy(np.float32)
        ehr[sl] = f.reindex(columns=ehr_cols).to_numpy(np.float32)
        ok = ~np.isnan(g)
        idx = np.flatnonzero(ok)
        rows.append(pd.DataFrame({"pos": a + idx, "pid": p["pid"], "timestamp": f["timestamp"].to_numpy()[idx],
                                  "label_spike": f["label_spike"].to_numpy()[idx]}))
    return {"seq": seq, "glucose": gl, "eng": eng, "ehr": ehr, "rows": pd.concat(rows, ignore_index=True),
            "L": L, "H": H, "eng_cols": eng_cols, "ehr_cols": ehr_cols}


class Windows(torch.utils.data.Dataset):
    def __init__(self, A: dict, pos: np.ndarray, labels: np.ndarray, stats: dict):
        self.A, self.pos, self.labels, self.st = A, pos, labels, stats
        self.offs = np.arange(-A["L"] + 1, 1)
        self.fut = np.arange(1, A["H"] + 1)

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, i: int):
        p = self.pos[i]
        w = p + self.offs
        A, st = self.A, self.st
        g0 = A["glucose"][p]
        fut = A["glucose"][p + self.fut]
        y = (fut - g0) / 50.0
        return (
            {k: torch.from_numpy(v[w]) for k, v in A["seq"].items()},
            torch.from_numpy(np.nan_to_num((A["eng"][p] - st["eng_mu"]) / st["eng_sd"])),
            torch.from_numpy(np.nan_to_num((A["ehr"][p] - st["ehr_mu"]) / st["ehr_sd"])),
            torch.from_numpy(np.nan_to_num(y).astype(np.float32)),
            torch.from_numpy((~np.isnan(y)).astype(np.float32)),
            torch.tensor(np.nan_to_num(fut, nan=0.0), dtype=torch.float32),
            torch.tensor(np.nan_to_num(self.labels[i], nan=0.0), dtype=torch.float32),
            torch.tensor(0.0 if np.isnan(self.labels[i]) else 1.0),
        )


# ============================================================================ model
class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3, film_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
        self.film = nn.Linear(film_dim, 2 * ch) if film_dim else None

    def forward(self, x, z=None):
        h = self.act(self.c1(nn.functional.pad(x, (self.pad, 0))))
        if self.film is not None and z is not None:
            gamma, beta = self.film(z).chunk(2, dim=-1)
            h = (1 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)
        h = self.drop(self.act(self.c2(nn.functional.pad(h, (self.pad, 0)))))
        return x + h


class Branch(nn.Module):
    def __init__(self, c_in: int, ch: int, dilations: tuple, film_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.blocks = nn.ModuleList([TCNBlock(ch, d, film_dim=film_dim, dropout=dropout) for d in dilations])

    def forward(self, x, z=None):                 # x: (B, L, C)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h, z)
        return h[:, :, -1]                         # representation at "now"


class PMRGF(nn.Module):
    def __init__(self, n_eng: int, n_ehr: int, H: int, ch: int = 32, dropout: float = 0.1,
                 use: tuple = ("meal", "act", "hr")):
        super().__init__()
        self.H, self.use = H, use
        zdim = 16
        self.static = nn.Sequential(nn.Linear(max(n_ehr, 1), 32), nn.GELU(), nn.Linear(32, zdim))
        self.cgm = Branch(4, ch, (1, 2, 4, 8, 16), film_dim=zdim, dropout=dropout)
        self.eng = nn.Sequential(nn.Linear(n_eng, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.GELU())
        core = ch + 64 + zdim
        self.base = nn.Sequential(nn.Linear(core, 128), nn.GELU(), nn.Linear(128, 3 * H))
        small = max(ch // 2, 8)
        self.branches = nn.ModuleDict()
        self.resid = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for name, c_in in (("meal", len(MEAL_CH)), ("act", len(ACT_CH) + 1), ("hr", len(HR_CH) + 1)):
            if name not in use:
                continue
            self.branches[name] = Branch(c_in, small, (1, 2, 4, 8), dropout=dropout)
            head = nn.Linear(small + core, H)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)            # zero-initialised residual: starts as CGM-only model
            self.resid[name] = head
            self.gates[name] = nn.Linear(small + core, H)
        self.risk = nn.Sequential(nn.Linear(core + small * len(self.branches), 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, seq, eng, ehr):
        z = self.static(ehr if ehr.shape[1] else torch.zeros(len(eng), 1))
        h = torch.cat([self.cgm(seq["cgm"], z), self.eng(eng), z], dim=-1)
        q = self.base(h).view(-1, 3, self.H)          # (B, quantile, step) of normalised delta
        extras, correction = [], 0.0
        for name, br in self.branches.items():
            hb = br(seq[name])
            u = torch.cat([hb, h], dim=-1)
            correction = correction + torch.sigmoid(self.gates[name](u)) * self.resid[name](u)
            extras.append(hb)
        q = q + (correction.unsqueeze(1) if torch.is_tensor(correction) else 0.0)
        q_sorted, _ = torch.sort(q, dim=1)            # keep q10 <= q50 <= q90
        logit = self.risk(torch.cat([h] + extras, dim=-1)).squeeze(-1)
        return q_sorted, logit


def loss_fn(q, logit, y, ym, fut_abs, lab, lab_m, cfg) -> torch.Tensor:
    w = torch.ones_like(y)
    for thr, weight in sorted((float(k), float(v)) for k, v in (cfg["nn"].get("zone_weights") or {}).items()):
        w = torch.where(fut_abs >= thr, torch.full_like(w, weight), w)
    w = w * ym
    med = q[:, 1]
    huber = (nn.functional.smooth_l1_loss(med, y, reduction="none", beta=0.2) * w).sum() / w.sum().clamp(min=1)
    pin = 0.0
    for i, tau in ((0, 0.1), (2, 0.9)):
        e = y - q[:, i]
        pin = pin + (torch.maximum(tau * e, (tau - 1) * e) * ym).sum() / ym.sum().clamp(min=1)
    vm = ym[:, 1:] * ym[:, :-1]
    vel = (nn.functional.smooth_l1_loss(med.diff(dim=1), y.diff(dim=1), reduction="none", beta=0.05) * vm).sum() \
        / vm.sum().clamp(min=1)
    pos = cfg["nn"]["risk_pos_weight"]
    bce = (nn.functional.binary_cross_entropy_with_logits(
        logit, lab, pos_weight=torch.tensor(pos), reduction="none") * lab_m).sum() / lab_m.sum().clamp(min=1)
    lw = cfg["nn"]["loss_weights"]
    return lw["forecast"] * huber + lw["quantile"] * pin + lw["velocity"] * vel + lw["risk"] * bce


# ============================================================================ training
def _stats(A: dict, pos: np.ndarray) -> dict:
    e, s = A["eng"][pos], A["ehr"][pos]
    return {"eng_mu": np.nanmean(e, 0), "eng_sd": np.nan_to_num(np.nanstd(e, 0), nan=1.0) + 1e-3,
            "ehr_mu": np.nan_to_num(np.nanmean(s, 0)) if s.shape[1] else np.zeros(0, np.float32),
            "ehr_sd": (np.nan_to_num(np.nanstd(s, 0), nan=1.0) + 1e-3) if s.shape[1] else np.ones(0, np.float32)}


def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    qs, ps = [], []
    with torch.no_grad():
        for seq, eng, ehr, *_ in loader:
            q, logit = model(seq, eng, ehr)
            qs.append(q.numpy())
            ps.append(torch.sigmoid(logit).numpy())
    return np.concatenate(qs), np.concatenate(ps)


def fit_fold(A: dict, rows: pd.DataFrame, train_pids, cfg, rng) -> tuple[PMRGF, dict, dict]:
    ncfg = cfg["nn"]
    pids = np.array(sorted(train_pids))
    n_val = max(1, int(round(len(pids) * cfg["model"]["early_stopping_val_frac"])))
    val_pids = set(rng.choice(pids, size=min(n_val, len(pids) - 1), replace=False))
    tr = rows[rows["pid"].isin(pids) & ~rows["pid"].isin(val_pids)]
    va = rows[rows["pid"].isin(val_pids)]
    tr = tr.iloc[::ncfg["train_stride"]]
    st = _stats(A, tr["pos"].to_numpy())
    mk = lambda d, sh: torch.utils.data.DataLoader(  # noqa: E731
        Windows(A, d["pos"].to_numpy(), d["label_spike"].to_numpy(float), st),
        batch_size=ncfg["batch_size"], shuffle=sh, num_workers=0)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    model = PMRGF(len(A["eng_cols"]), len(A["ehr_cols"]), A["H"], ch=ncfg["channels"], dropout=ncfg["dropout"],
                  use=tuple(ncfg["branches"]))
    opt = torch.optim.AdamW(model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ncfg["lr"], total_steps=ncfg["epochs"] * len(dl_tr))
    yv = np.stack([dl_va.dataset[i][3].numpy() for i in range(len(dl_va.dataset))])
    mv = np.stack([dl_va.dataset[i][4].numpy() for i in range(len(dl_va.dataset))])
    best, best_state, bad = np.inf, None, 0
    for ep in range(ncfg["epochs"]):
        model.train()
        t0 = time.time()
        for seq, eng, ehr, y, ym, fut, lab, lab_m in dl_tr:
            q, logit = model(seq, eng, ehr)
            loss = loss_fn(q, logit, y, ym, fut, lab, lab_m, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        q, _ = _predict(model, dl_va)
        mae = float((np.abs(q[:, 1] - yv) * mv).sum() / mv.sum() * 50)
        log.info("epoch %d: val MAE (all steps) %.2f mg/dL, %.0fs", ep + 1, mae, time.time() - t0)
        if mae < best - 0.01:
            best, bad = mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ncfg["patience"]:
                break
    model.load_state_dict(best_state)
    # conformal widening of the band on validation patients (same idea as the LightGBM model)
    q, _ = _predict(model, dl_va)
    lo, hi = q[:, 0], q[:, 2]
    score = np.where(mv > 0, np.maximum(lo - yv, yv - hi), np.nan)
    adj = np.nan_to_num(np.nanquantile(score, 0.8, axis=0)).clip(min=0)
    return model, st, {"band_adjust": adj, "val_mae": best}


def _model_from(ckpt: dict) -> PMRGF:
    hp = ckpt["hparams"]
    model = PMRGF(hp["n_eng"], hp["n_ehr"], hp["H"], ch=hp["channels"], dropout=hp["dropout"],
                  use=tuple(hp["branches"]))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def _checkpoint(model: PMRGF, st: dict, extra: dict, A: dict, cfg, train_pids) -> dict:
    return {"state": model.state_dict(), "stats": st, "extra": extra, "eng_cols": A["eng_cols"],
            "ehr_cols": A["ehr_cols"], "train_pids": sorted(int(p) for p in train_pids),
            "hparams": {"n_eng": len(A["eng_cols"]), "n_ehr": len(A["ehr_cols"]), "H": A["H"],
                        "channels": cfg["nn"]["channels"], "dropout": cfg["nn"]["dropout"],
                        "branches": list(cfg["nn"]["branches"])}}


def _predict_rows(model: PMRGF, A: dict, st: dict, extra: dict, te: pd.DataFrame, horizons, step) -> pd.DataFrame:
    dl = torch.utils.data.DataLoader(Windows(A, te["pos"].to_numpy(), te["label_spike"].to_numpy(float), st),
                                     batch_size=512, shuffle=False)
    q, p = _predict(model, dl)
    g0 = A["glucose"][te["pos"].to_numpy()]
    res = te[["pid", "timestamp"]].copy()
    for h in horizons:
        j = h // step - 1
        res[f"pred_nn_{h}"] = g0 + 50 * q[:, 1, j]
        res[f"lo_nn_{h}"] = g0 + 50 * (q[:, 0, j] - extra["band_adjust"][j])
        res[f"hi_nn_{h}"] = g0 + 50 * (q[:, 2, j] + extra["band_adjust"][j])
    res["p_spike_nn"] = p
    return res


def _horizons(cfg) -> list[int]:
    return [h for h in cfg["task"]["horizons_min"] if h <= cfg["nn"]["horizon_min"]]


def nn_metrics(frame: pd.DataFrame, cfg) -> dict:
    ds = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    horizons = _horizons(cfg)
    keep = ["pid", "timestamp", "glucose", "label_spike"] + [f"target_delta_{h}" for h in horizons]
    m = ds[keep].merge(frame, on=["pid", "timestamp"], how="inner")
    metrics = {"regression": {}}
    for h in horizons:
        y = m["glucose"] + m[f"target_delta_{h}"]
        metrics["regression"][h] = ev.regression_metrics(y, m[f"pred_nn_{h}"], m["glucose"])
        inside = (y >= m[f"lo_nn_{h}"]) & (y <= m[f"hi_nn_{h}"])
        metrics["regression"][h]["band_coverage_pct"] = float(inside[y.notna()].mean() * 100)
    metrics["spike"] = ev.classification_metrics(m["label_spike"], m["p_spike_nn"], 0.5)
    return metrics


def run(cfg, quick: bool = False) -> dict:
    """Cross-validation over development patients, then one network on all of them."""
    t_start = time.time()
    torch.manual_seed(cfg["model"]["seed"])
    if quick:
        cfg["nn"].update({"epochs": 2, "channels": 16, "train_stride": 4})
    splits = load_splits(cfg)
    A = build_arrays(cfg)
    rows = A["rows"]
    dev_rows = rows[rows["pid"].isin(splits["dev"])]
    assert not set(dev_rows["pid"]) & set(splits["test"])
    art = cfg.path("artifacts_dir")
    (art / "models").mkdir(parents=True, exist_ok=True)
    (art / "cv").mkdir(parents=True, exist_ok=True)
    horizons, step = _horizons(cfg), int(cfg["data"]["resample_minutes"])
    rng = np.random.default_rng(cfg["model"]["seed"])
    out = []
    for k, held in enumerate(splits["folds"]):
        t0 = time.time()
        train_pids = [p for p in splits["dev"] if p not in set(held)]
        model, st, extra = fit_fold(A, dev_rows, train_pids, cfg, rng)
        res = _predict_rows(model, A, st, extra, dev_rows[dev_rows["pid"].isin(held)], horizons, step)
        res["fold"] = k
        out.append(res)
        torch.save(_checkpoint(model, st, extra, A, cfg, train_pids), art / "models" / f"nn_fold_{k}.pt")
        log.info("fold %d/%d done in %.0fs", k + 1, len(splits["folds"]), time.time() - t0)
    oof = pd.concat(out, ignore_index=True)
    oof.to_parquet(art / "cv" / "oof_nn.parquet", index=False)
    metrics = nn_metrics(oof, cfg)

    t0 = time.time()
    model, st, extra = fit_fold(A, dev_rows, splits["dev"], cfg, rng)
    torch.save(_checkpoint(model, st, extra, A, cfg, splits["dev"]), art / "models" / "nn_dev.pt")
    metrics["timing_s"] = {"cross_validation": round(t0 - t_start), "development_model": round(time.time() - t0)}
    (art / "cv" / "metrics_nn.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("NN done: CV RMSE %s", {h: round(r["rmse"], 1) for h, r in metrics["regression"].items()})
    return metrics


def predict_test(cfg) -> pd.DataFrame | None:
    """Score the locked test patients with the development network (called by glucotwin.test)."""
    path = cfg.path("artifacts_dir") / "models" / "nn_dev.pt"
    if not path.exists():
        return None
    splits = load_splits(cfg)
    ckpt = torch.load(path, weights_only=False)
    assert not set(ckpt["train_pids"]) & set(splits["test"]), "network was trained on a test patient"
    A = build_arrays(cfg)
    te = A["rows"][A["rows"]["pid"].isin(splits["test"])]
    return _predict_rows(_model_from(ckpt), A, ckpt["stats"], ckpt["extra"], te, _horizons(cfg),
                         int(cfg["data"]["resample_minutes"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    if args.epochs:
        cfg["nn"]["epochs"] = args.epochs
    m = run(cfg, quick=args.quick)
    print(json.dumps({"cv_rmse": {h: round(r["rmse"], 1) for h, r in m["regression"].items()},
                      "cv_spike_auroc": m["spike"].get("auroc")}, indent=2, default=float))


if __name__ == "__main__":
    main()
