"""GlucoTwin doctor dashboard.

    streamlit run app/dashboard.py                 # your trained models in artifacts/
    streamlit run app/dashboard.py -- --demo       # the exported bundle in demo/ (as hosted online)

The dashboard "replays" the monitoring period: pick a patient, a day and a time, and the twin
shows what it would have predicted at that moment, using a model that never saw the patient.
Live mode plays the data forward one CGM reading at a time, as if it were arriving now.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from glucotwin.meals import INDIAN_MEALS, compose_meal  # noqa: E402

INK, TEAL, AMBER, RED, SLATE = "#1B2A41", "#0F766E", "#B45309", "#B42318", "#64748B"
RANGE_FILL = "rgba(15,118,110,0.08)"
FONT = "Atkinson Hyperlegible, system-ui, sans-serif"
SPLIT_TEXT = {"test": "locked test patient", "development": "development patient", "new": "new patient"}
GROUP_TEXT = {"healthy": "No diabetes", "prediabetes": "Prediabetes", "t2d": "Type 2 diabetes", "unknown": "Unknown"}
SPEEDS = [1, 2, 3, 5]

RUN_START = time.time()
st.set_page_config(page_title="GlucoTwin", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, .stMarkdown, button, input, textarea, select,
[class*="st-"]:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]) {
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
/* Streamlit draws its icons (sidebar arrows, expanders, toasts) as ligature text such as
   "keyboard_double_arrow_left". They must keep the icon font, or that text shows instead. */
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *,
[class*="material-symbols"], [class*="material-icons"] {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  font-feature-settings: 'liga' !important;
  -webkit-font-feature-settings: 'liga' !important;
  white-space: nowrap; overflow: hidden;
}
[data-testid="stSidebar"] { overflow-x: hidden; }
h1 { font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0; }
.gt-risk { border-left: 6px solid var(--c); padding: 0.7rem 1rem; background: #F8FAFB; margin-bottom: 0.8rem; }
.gt-risk h3 { margin: 0 0 0.3rem 0; color: var(--c); font-size: 1.25rem; }
.gt-risk p { margin: 0.15rem 0; }
.gt-muted { color: #64748B; font-size: 0.88rem; }
.gt-live { display: inline-block; font-weight: 700; color: #B42318; letter-spacing: 0.04em;
           border: 1.5px solid #B42318; border-radius: 4px; padding: 0 0.45rem; margin-right: 0.5rem; }
.gt-live::before { content: ""; display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%;
                   background: #B42318; margin-right: 0.35rem; animation: gt-pulse 1.2s ease-in-out infinite; }
@keyframes gt-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
@media (prefers-reduced-motion: reduce) { .gt-live::before { animation: none; } }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================ data
def local_data_ready() -> bool:
    """True when this copy holds the prepared data and trained models (your own computer). A fresh
    clone, such as Streamlit Community Cloud, has neither: the dataset is never committed."""
    return ((ROOT / "data" / "processed" / "dataset.parquet").exists()
            and (ROOT / "artifacts" / "models" / "model_dev.joblib").exists())


def config_path() -> str | None:
    """None = the normal project config (data/ and artifacts/). The demo bundle is used when asked
    for, or automatically when this copy has no prepared data but has an exported bundle."""
    demo = ROOT / "demo" / "config.yaml"
    if "--demo" in sys.argv[1:] or os.environ.get("GLUCOTWIN_DEMO") == "1":
        return str(demo)
    if os.environ.get("GLUCOTWIN_CONFIG"):
        return os.environ["GLUCOTWIN_CONFIG"]
    if not local_data_ready() and demo.exists():
        return str(demo)
    return None


CONFIG = config_path()
DEMO = CONFIG is not None and Path(CONFIG).resolve() == (ROOT / "demo" / "config.yaml").resolve()


@st.cache_resource(show_spinner="Loading the twin...")
def load_service(path: str | None):
    from glucotwin import service
    return service.TwinService(service.load_config(path) if path else None)


def missing_artifacts_screen(err: Exception) -> None:
    st.title("GlucoTwin")
    if DEMO or not (ROOT / "data" / "processed").exists() or not local_data_ready():
        # Typical on Streamlit Community Cloud: a fresh clone has no data/ (never committed).
        st.error("This copy of the repository has no prepared data and no demo bundle, so there is "
                 "nothing to show yet.")
        st.markdown(
            "**Hosting the dashboard online (Streamlit Community Cloud).** The dataset is never in the "
            "repository, so the hosted app runs on a small demo bundle. On the computer that trained the "
            "model, from the project folder:\n\n"
            "```bash\n"
            "python scripts/export_demo.py\n"
            "git add demo packages.txt\n"
            "git commit -m \"Add demo bundle for the live dashboard\"\n"
            "git push\n"
            "```\n"
            "Then reboot the app (Manage app, then Reboot). It picks up `demo/` by itself.\n\n"
            "**On your own computer**, run the pipeline first (README, steps 5 to 10): "
            "`prepare`, `train`, `test`."
        )
    else:
        st.error("The twin has not been trained yet, so there is nothing to show.")
        st.markdown(
            "Run these commands from the repository root, then reload this page:\n\n"
            "```bash\n"
            "python scripts/download_data.py      # or demo: python scripts/make_synthetic_data.py --out data/raw/synthetic\n"
            "python -m glucotwin.prepare          # demo: python -m glucotwin.prepare --raw data/raw/synthetic\n"
            "python -m glucotwin.train\n"
            "python -m glucotwin.test\n"
            "```"
        )
    st.caption(f"Details: {err}")
    st.stop()


try:
    svc = load_service(CONFIG)
except FileNotFoundError as exc:
    missing_artifacts_screen(exc)


@st.cache_data(show_spinner=False)
def patient_table(cfg_key: str | None) -> pd.DataFrame:
    return svc.patients()


patients = patient_table(CONFIG)
STEP = pd.Timedelta(minutes=svc.step)
THR_MGDL = float(svc.cfg["task"]["spike_abs_mgdl"])
RISE_MGDL = float(svc.cfg["task"].get("spike_rise_mgdl", 50))
SPIKE_DEF = str(svc.cfg["task"].get("spike_definition", "absolute"))
SPIKE_TEXT = {"absolute": f"glucose reaching {THR_MGDL:.0f} mg/dL",
              "rise": f"glucose rising {RISE_MGDL:.0f} mg/dL",
              "either": f"glucose reaching {THR_MGDL:.0f} mg/dL or rising {RISE_MGDL:.0f} mg/dL"}.get(
    SPIKE_DEF, f"glucose reaching {THR_MGDL:.0f} mg/dL")
SYNTH = bool(getattr(svc, "synthetic", False))


def patient_label(pid: int) -> str:
    r = patients.set_index("pid").loc[pid]
    a1c = f" · HbA1c {r['hba1c']:.1f}%" if pd.notna(r["hba1c"]) else ""
    short = {"healthy": "No diabetes", "prediabetes": "Prediabetes", "t2d": "Type 2"}
    return f"{pid:03d} · {short.get(r['group'], GROUP_TEXT.get(r['group'], r['group']))}{a1c}"


# ============================================================================ sidebar: patient
default_pid = int(patients.sort_values("time_above_180_pct", ascending=False)["pid"].iloc[0])
pid = st.sidebar.selectbox("Patient", patients["pid"].tolist(), index=patients["pid"].tolist().index(default_pid),
                           format_func=patient_label)
tl = svc.timeline(pid)
t_first, t_last = tl["timestamp"].min(), tl["timestamp"].max()
day0 = t_first.normalize()
max_day = int(tl["day"].max())
cal_days = int(svc.cfg["personalization"]["calibration_days"])
meals = svc.patient_meals(pid)


# ============================================================================ the replay clock
# The current time is held in two widget states (day slider and clock). Buttons and live mode
# change it by setting those states at the top of a run, before the widgets are drawn.
DAY_KEY = f"day_{pid}"


def clock_key(d: int) -> str:
    return f"clock_{pid}_{d}"


def meals_on(d: int) -> pd.DataFrame:
    return meals[(meals["timestamp"] >= day0 + pd.Timedelta(days=d)) &
                 (meals["timestamp"] < day0 + pd.Timedelta(days=d + 1))]


def lunch_on(d: int):
    dm = meals_on(d)
    lunch = dm[dm["meal_type"].astype(str).str.lower().str.startswith("lunch")]
    return lunch.iloc[0] if len(lunch) else (dm.iloc[0] if len(dm) else None)


def default_clock(d: int) -> dtime:
    pick = lunch_on(d)
    return (pick["timestamp"] + pd.Timedelta(minutes=5)).time() if pick is not None else dtime(13, 0)


def current_time() -> pd.Timestamp:
    d = st.session_state.get(DAY_KEY, min(cal_days + 1, max_day))
    c = st.session_state.get(clock_key(d), default_clock(d))
    return day0 + pd.Timedelta(days=d, hours=c.hour, minutes=c.minute)


def set_time(t: pd.Timestamp) -> None:
    t = min(max(pd.Timestamp(t), t_first), t_last).floor(STEP)
    d = int((t.normalize() - day0).days)
    st.session_state[DAY_KEY] = d
    st.session_state[clock_key(d)] = t.time()


def request_advance() -> None:
    st.session_state["_advance"] = True


def go_before_lunch() -> None:
    pick = lunch_on(st.session_state.get(DAY_KEY, 0))
    if pick is not None:
        set_time(pick["timestamp"] - pd.Timedelta(minutes=60))


if st.session_state.pop("_advance", False):
    nxt = current_time() + STEP
    if nxt > t_last:
        st.session_state["live"] = False
        st.session_state["_live_ended"] = True
    else:
        set_time(nxt)

live = bool(st.session_state.get("live", False))
st.session_state.setdefault(DAY_KEY, min(cal_days + 1, max_day))
day = st.sidebar.slider("Day of monitoring", 0, max_day, key=DAY_KEY, disabled=live)
st.session_state.setdefault(clock_key(day), default_clock(day))
clock = st.sidebar.time_input("Time now", step=300, key=clock_key(day), disabled=live)
c_next, c_lunch = st.sidebar.columns(2)
c_next.button("Next reading", on_click=request_advance, disabled=live,
              help="Moves time forward one CGM reading (5 minutes) so you can watch the twin update.")
c_lunch.button("Before lunch", on_click=go_before_lunch, disabled=live or lunch_on(day) is None,
               help="Jumps to one hour before this day's lunch: a good starting point for live mode.")
day_meals = meals_on(day)
if len(day_meals):
    st.sidebar.caption("Meals logged this day: " + "; ".join(
        f"{t:%H:%M} {str(m).lower()} ({c:.0f} g carbs)"
        for t, m, c in zip(day_meals["timestamp"], day_meals["meal_type"].fillna("meal"), day_meals["carbs"])))
now = day0 + pd.Timedelta(days=day, hours=clock.hour, minutes=clock.minute)

st.sidebar.subheader("Live replay")
st.sidebar.toggle("Live mode", key="live", on_change=lambda: st.session_state.update(_live_started=True),
                  help="Plays the data forward one CGM reading at a time, as if it were arriving now. "
                       "The twin only ever sees readings up to the current time.")
speed = st.sidebar.select_slider("Speed", options=SPEEDS, value=2, key="live_speed",
                                 format_func=lambda s: f"1 reading every {s} s")
if st.session_state.pop("_live_ended", False):
    st.sidebar.info("Reached the end of this patient's data.")

st.sidebar.subheader("View")
personalized = st.sidebar.toggle("Personalized twin", value=True,
                                 help=f"Corrects the model using this patient's first {cal_days} days.")
reveal = st.sidebar.toggle("Show what actually happened", value=False,
                           help="Overlay the measured glucose after the current time (hidden in a real clinic).")
if SYNTH:
    note = ("Demo on <b>synthetic patients</b> in the CGMacros format: not real people. The model was "
            "trained on the same synthetic data, so its forecasts here are not scientific results. ")
elif DEMO:
    note = "Public demo on the locked test patients of CGMacros (PhysioNet, CC BY-NC-SA 4.0). "
else:
    note = "Built on de-identified public data (CGMacros, PhysioNet). "
st.sidebar.markdown(
    f"<p class='gt-muted'>{note}Research prototype, not a medical device and not for clinical decisions.</p>",
    unsafe_allow_html=True)

# ============================================================================ header
rec = svc.ehr(pid)["summary"]
row = patients.set_index("pid").loc[pid]
bits = [f"Patient {pid:03d}"]
if rec.get("age"):
    bits.append(f"{rec['age']}-year-old {rec.get('gender', '')}".strip())
st.title("GlucoTwin")
if SYNTH:
    st.warning("Synthetic demo data: these are not real patients, and the forecasts on this page are not "
               "scientific results. The team's results on real patients are in the Model evidence tab.")
st.markdown(f"**{', '.join(bits)}.** {GROUP_TEXT.get(row['group'])}"
            + (f" (HbA1c {row['hba1c']:.1f}%)" if pd.notna(row["hba1c"]) else "")
            + (f", BMI {row['bmi']:.1f}" if pd.notna(row["bmi"]) else "") + ". "
            + (f"Medications: {', '.join(rec['medications'])}." if rec["medications"] else "No medications recorded.")
            + f" <span class='gt-muted'>({SPLIT_TEXT[svc.split_of(pid)]}; forecasts come from a model that "
            "never saw this patient)</span>", unsafe_allow_html=True)
if live:
    st.markdown(f"<span class='gt-live'>LIVE</span> Day {day}, {now:%H:%M}. "
                f"<span class='gt-muted'>A new CGM reading (5 minutes of data) arrives every {speed} s.</span>",
                unsafe_allow_html=True)

tab_now, tab_whatif, tab_record, tab_clinic, tab_model = st.tabs(
    ["Forecast", "What-if", "Health record", "Clinic overview", "Model evidence"])


def base_layout(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=dict(family=FONT, color=INK),
                      plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EEF2F5", zeroline=False)
    return fig


def add_now_line(fig: go.Figure, t: pd.Timestamp) -> None:
    fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper", line=dict(color=INK, width=1, dash="dot"))
    fig.add_annotation(x=t, y=1, yref="paper", text="now", showarrow=False, yanchor="bottom",
                       font=dict(color=INK, size=12))


def alert_outcome(a: dict, t_now: pd.Timestamp) -> str:
    """What happened after an alert, using only readings up to now (nothing from the future)."""
    raised = a["raised"]
    if SPIKE_DEF == "absolute" and a["glucose"] >= THR_MGDL:
        return "Was already above"
    end = raised + pd.Timedelta(minutes=120)
    w = tl[(tl["timestamp"] > raised) & (tl["timestamp"] <= min(end, t_now))]
    reached = pd.Series(False, index=w.index)
    if SPIKE_DEF != "rise":
        reached = reached | (w["glucose"] >= THR_MGDL)
    if SPIKE_DEF in ("rise", "either"):
        reached = reached | (w["glucose"] >= a["glucose"] + RISE_MGDL)
    hit = w[reached]
    if len(hit):
        t_hit = hit["timestamp"].iloc[0]
        return f"Yes, after {int((t_hit - raised).total_seconds() // 60)} min"
    if t_now >= end:
        return "No (false alert)"
    return "Watching"


# ============================================================================ forecast tab
with tab_now:
    fc = svc.forecast(pid, now, personalized)
    t_now = pd.Timestamp(fc["at"])
    fc_prev = svc.forecast(pid, t_now - STEP, personalized)

    # Alert log: an alert is "raised" when the score crosses the threshold between two readings.
    log_key = f"alerts_{pid}"
    # An alert is logged once per episode: when it switches on between two readings, or, if it is
    # already on, when live mode starts. Jumping around with the clock logs nothing.
    prev = st.session_state.get("_prev_eval")                    # (pid, time, alert on, logged)
    same = bool(prev) and prev[0] == pid and prev[1] == t_now
    contiguous = bool(prev) and prev[0] == pid and prev[1] == t_now - STEP
    started = st.session_state.pop("_live_started", False) and live
    if same:
        logged = prev[3]
    elif contiguous:
        logged = prev[3] if (fc["alert"] and prev[2]) else False
    else:
        logged = False
    if fc["alert"] and not logged and (contiguous or started or (live and not same)):
        st.session_state.setdefault(log_key, []).append(
            {"raised": t_now, "risk": fc["p_spike"], "peak": max(fc["forecast"]),
             "glucose": fc["glucose_now"], "at_start": not (contiguous and not prev[2])})
        st.toast(f"Spike alert for patient {pid:03d} at {t_now:%H:%M}: risk {fc['p_spike']:.0%}, "
                 f"forecast peak {max(fc['forecast']):.0f} mg/dL", icon=":material/notifications_active:")
        logged = True
    st.session_state["_prev_eval"] = (pid, t_now, fc["alert"], logged)

    left, right = st.columns([2.3, 1])
    with left:
        past = tl[(tl["timestamp"] >= t_now - pd.Timedelta(hours=8)) & (tl["timestamp"] <= t_now)]
        fut = tl[(tl["timestamp"] > t_now) & (tl["timestamp"] <= t_now + pd.Timedelta(minutes=130))]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
        fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_FILL, line_width=0, row=1, col=1)
        fig.add_hline(y=180, line=dict(color=RED, width=0.8, dash="dash"), row=1, col=1)
        fig.add_trace(go.Scatter(x=past["timestamp"], y=past["glucose"], name="Measured", mode="lines",
                                 line=dict(color=INK, width=2)), row=1, col=1)
        times = pd.to_datetime(fc["times"])
        fig.add_trace(go.Scatter(x=list(times) + list(times[::-1]), y=fc["hi"] + fc["lo"][::-1], fill="toself",
                                 fillcolor="rgba(180,83,9,0.15)", line=dict(width=0), hoverinfo="skip",
                                 name="Likely range (80%)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=pd.to_datetime(fc_prev["times"]), y=fc_prev["forecast"],
                                 name="Forecast 5 min ago", mode="lines",
                                 line=dict(color=AMBER, width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=fc["forecast"], name="Twin forecast", mode="lines+markers",
                                 line=dict(color=AMBER, width=2.5), marker=dict(size=6)), row=1, col=1)
        if reveal and len(fut):
            fig.add_trace(go.Scatter(x=fut["timestamp"], y=fut["glucose"], name="What actually happened",
                                     mode="lines", line=dict(color=SLATE, width=1.5, dash="dot")), row=1, col=1)
        wm = meals[(meals["timestamp"] >= t_now - pd.Timedelta(hours=8)) & (meals["timestamp"] <= t_now)]
        if len(wm):
            fig.add_trace(go.Scatter(x=wm["timestamp"], y=[52] * len(wm), mode="markers+text", name="Meal",
                                     marker=dict(symbol="triangle-up", size=12, color=TEAL),
                                     text=[f"{c:.0f} g" for c in wm["carbs"]], textposition="top center",
                                     textfont=dict(color=TEAL)), row=1, col=1)
        for a in st.session_state.get(log_key, []):
            if t_now - pd.Timedelta(hours=8) <= a["raised"] <= t_now:
                fig.add_trace(go.Scatter(x=[a["raised"]], y=[a["glucose"]], mode="markers", name="Alert raised",
                                         marker=dict(symbol="diamond", size=11, color=RED,
                                                     line=dict(color="white", width=1)),
                                         showlegend=False, hovertemplate="Alert raised %{x|%H:%M}<extra></extra>"),
                              row=1, col=1)
        act = tl[(tl["timestamp"] >= t_now - pd.Timedelta(hours=8)) & (tl["timestamp"] <= t_now)]
        fig.add_trace(go.Bar(x=act["timestamp"], y=act["mets_mean_15"], name="Activity (METs)",
                             marker_color="rgba(15,118,110,0.45)", showlegend=False), row=2, col=1)
        add_now_line(fig, t_now)
        ymax = max(260, float(np.nanmax(fc["hi"])) + 20, float(past["glucose"].max() or 0) + 20)
        fig.update_yaxes(title_text="mg/dL", range=[40, ymax], row=1, col=1)
        fig.update_yaxes(title_text="METs", row=2, col=1)
        fig.update_xaxes(range=[t_now - pd.Timedelta(hours=8), t_now + pd.Timedelta(minutes=135)])
        st.plotly_chart(base_layout(fig), key="fc_chart")
    with right:
        peak_i = int(np.argmax(fc["forecast"]))
        peak_t = pd.Timestamp(fc["times"][peak_i])
        color = RED if fc["alert"] else TEAL
        head = "Spike likely within 2 hours" if fc["alert"] else "No spike expected in the next 2 hours"
        spike_note = f"<p class='gt-muted'>Spike = {SPIKE_TEXT}.</p>"
        st.markdown(
            f"<div class='gt-risk' style='--c:{color}'><h3>{head}</h3>"
            f"<p>Glucose now {fc['glucose_now']:.0f} mg/dL. Forecast peak {fc['forecast'][peak_i]:.0f} mg/dL "
            f"around {peak_t:%H:%M}.</p>"
            f"<p>Risk score {fc['p_spike']:.0%} (alert above {fc['spike_threshold']:.0%}).</p>" + spike_note
            + (f"<p>Expected to cross 180 mg/dL in about {fc['expected_crossing_min']} minutes.</p>"
               if fc.get("expected_crossing_min") is not None else "")
            + f"<p class='gt-muted'>Latest reading updated the 2-hour forecast from "
              f"{fc_prev['forecast'][-1]:.0f} to {fc['forecast'][-1]:.0f} mg/dL.</p>"
            + ("<p class='gt-muted'>Personalized to this patient.</p>" if fc["personalized"] else
               "<p class='gt-muted'>Population model (personalization starts after the calibration days).</p>")
            + "</div>", unsafe_allow_html=True)
        if fc.get("hypo_alert"):
            st.warning("Low glucose (below 70 mg/dL) is possible within the next hour.")
        alerts = st.session_state.get(log_key, [])
        if alerts:
            st.markdown("**Alerts raised in this replay**")
            st.dataframe(pd.DataFrame([{
                "Raised": f"Day {int((a['raised'].normalize() - day0).days)}, {a['raised']:%H:%M}"
                          + ("*" if a.get("at_start") else ""),
                "Risk": f"{a['risk']:.0%}", "Spike happened?": alert_outcome(a, t_now)} for a in alerts[::-1][:6]]),
                hide_index=True)
            st.caption(f"Spike = {SPIKE_TEXT} within 2 hours. Checked only against readings up to the "
                       "current time."
                       + (" * Already on when the replay started." if any(a.get("at_start") for a in alerts) else ""))
        d = pd.DataFrame(fc["drivers_60min"])
        if len(d):
            d = d.iloc[::-1]
            bar = go.Figure(go.Bar(x=d["effect_mgdl"], y=d["factor"], orientation="h",
                                   marker_color=[RED if v > 0 else TEAL for v in d["effect_mgdl"]],
                                   text=[f"{v:+.0f}" for v in d["effect_mgdl"]], textposition="outside"))
            bar.update_layout(title=dict(text="What moves the 60-minute forecast", font=dict(size=14)))
            lo_x, hi_x = min(0.0, float(d["effect_mgdl"].min())), max(0.0, float(d["effect_mgdl"].max()))
            pad = 0.2 * max(hi_x - lo_x, 1.0)
            bar.update_xaxes(title_text="mg/dL", range=[lo_x - (pad if lo_x < 0 else 0), hi_x + pad])
            st.plotly_chart(base_layout(bar, 280), key="drivers")
            st.caption("Red raises the forecast, teal lowers it, relative to the model's average.")
        if fc["alert"] and not live:
            st.info("Test advice before giving it: open the What-if tab to compare a smaller portion or a "
                    "short walk after the meal.")

# ============================================================================ what-if tab
with tab_whatif:
    if live:
        st.info("Live mode is running. Switch it off in the sidebar to test a meal or a walk at the current moment.")
    else:
        st.markdown(f"Simulate a meal eaten at **{t_now:%H:%M} on day {day}** and compare the twin's forecasts. "
                    "No other food is eaten for the next 2 hours, and activity stays at this patient's usual "
                    "resting level except for the walk you add.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            mode = st.radio("Meal", ["Keep the logged meal", "Build an Indian meal", "Enter macros"], horizontal=True)
            meal = None
            if mode == "Build an Indian meal":
                dishes = st.multiselect("Dishes", sorted(INDIAN_MEALS), default=["White rice (cooked)", "Sambar",
                                                                                "Mixed vegetable sabzi"])
                servings = {}
                cols = st.columns(3)
                for i, dname in enumerate(dishes):
                    with cols[i % 3]:
                        servings[dname] = st.number_input(f"{dname} ({INDIAN_MEALS[dname][0]})", 0.0, 5.0, 1.0,
                                                          0.5, key=f"sv_{dname}")
                meal = compose_meal(servings) if servings else None
                if meal:
                    st.caption(f"{meal['carbs']:.0f} g carbs, {meal['protein']:.0f} g protein, "
                               f"{meal['fat']:.0f} g fat, {meal['fiber']:.0f} g fiber, {meal['meal_kcal']:.0f} kcal "
                               "(approximate values).")
            elif mode == "Enter macros":
                m1, m2, m3, m4 = st.columns(4)
                meal = {"carbs": m1.number_input("Carbs (g)", 0.0, 300.0, 60.0, 5.0),
                        "protein": m2.number_input("Protein (g)", 0.0, 200.0, 20.0, 5.0),
                        "fat": m3.number_input("Fat (g)", 0.0, 200.0, 15.0, 5.0),
                        "fiber": m4.number_input("Fiber (g)", 0.0, 60.0, 5.0, 1.0)}
                meal["meal_kcal"] = 4 * meal["carbs"] + 4 * meal["protein"] + 9 * meal["fat"]
        with c2:
            walk = st.slider("Walk after eating (minutes)", 0, 45, 15, 5)
            walk_start = st.slider("Start the walk this many minutes after eating", 0, 60, 10, 5)
            walk_mets = st.select_slider("Walking pace", options=[2.5, 3.5, 4.5], value=3.5,
                                         format_func=lambda v: {2.5: "Slow", 3.5: "Brisk", 4.5: "Very brisk"}[v])
        label_meal = "Logged meal" if meal is None else "This meal"
        scenarios = [{"label": "Logged meal, no walk", "meal": None}]
        if meal is not None:
            scenarios.append({"label": "This meal, no walk", "meal": meal})
        if walk > 0:
            scenarios.append({"label": f"{label_meal} + {walk}-min walk", "meal": meal, "walk_minutes": walk,
                              "walk_start_min": walk_start, "walk_mets": walk_mets})
        results = svc.whatif(pid, t_now, scenarios, personalized)
        styles = [dict(color=SLATE, dash="dot"), dict(color=AMBER, dash="solid"), dict(color=TEAL, dash="solid")]
        fig = go.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_FILL, line_width=0)
        fig.add_hline(y=180, line=dict(color=RED, width=0.8, dash="dash"))
        past = tl[(tl["timestamp"] >= t_now - pd.Timedelta(hours=3)) & (tl["timestamp"] <= t_now)]
        fig.add_trace(go.Scatter(x=past["timestamp"], y=past["glucose"], name="Measured",
                                 line=dict(color=INK, width=2)))
        for r, stl in zip(results, styles):
            fig.add_trace(go.Scatter(x=pd.to_datetime(r["times"]), y=r["forecast"], name=r["label"],
                                     mode="lines+markers", line=dict(width=2.5, **stl)))
        add_now_line(fig, t_now)
        fig.update_yaxes(title_text="mg/dL")
        st.plotly_chart(base_layout(fig, 420), key="whatif_chart")
        base_peak = results[0]["peak_forecast"]
        table = pd.DataFrame([{
            "Scenario": r["label"], "Forecast peak (mg/dL)": round(r["peak_forecast"]),
            "Change vs logged meal": f"{r['peak_forecast'] - base_peak:+.0f}",
            "Spike risk": f"{r['p_spike']:.0%}",
        } for r in results])
        st.dataframe(table, hide_index=True)
        st.warning("Simulated scenarios from a research model. They are not treatment recommendations.")
        st.caption("The twin respects physiology by design: more carbohydrate can never lower its forecast, and "
                   "more planned activity can never raise it. Wrist-worn activity carried little signal in this "
                   "dataset, so the effect of a walk is small. Absolute values are estimates with the uncertainty "
                   "shown in the Forecast tab.")

# ============================================================================ record tab
with tab_record:
    s = svc.ehr(pid)["summary"]
    a, b = st.columns(2)
    with a:
        st.subheader("Conditions")
        st.write("\n".join(f"- {c}" for c in s["conditions"]) or "None recorded")
        st.subheader("Medications")
        st.write("\n".join(f"- {m}" for m in s["medications"]) or "None recorded")
    with b:
        st.subheader("Baseline measurements")
        st.dataframe(pd.DataFrame({"Measurement": list(s["labs"].keys()), "Value": list(s["labs"].values())}),
                     hide_index=True)
    used = bool(svc.cfg["ehr"].get("use_synthetic_fields_as_features", False))
    st.caption("Fields marked synthetic were generated for this prototype (the public dataset has no diagnoses, "
               "medications or genotypes). " + ("They are used as model inputs in this configuration."
                                                 if used else "They are shown for context only and are never "
                                                 "used as model inputs."))
    with st.expander("FHIR bundle (the record exactly as the twin reads it)"):
        st.json(svc.ehr(pid)["fhir_bundle"], expanded=False)

# ============================================================================ clinic overview
with tab_clinic:
    st.markdown(f"Every patient at **day {day}, {clock:%H:%M}**, ranked by spike risk. "
                "Scores come from models that never saw the patient.")

    @st.cache_data(show_spinner=False)
    def overview(day_: int, minute_: int, cfg_key: str | None) -> pd.DataFrame:
        if svc.oof is None:
            return pd.DataFrame()
        groups = patients.set_index("pid")["group"]
        rows = []
        for p, d in svc.oof.groupby("pid"):
            t = d["timestamp"].min().normalize() + pd.Timedelta(days=day_, minutes=minute_)
            d = d[d["timestamp"] <= t]
            if d.empty:
                continue
            r = d.iloc[-1]
            rows.append({"Patient": f"{int(p):03d}",
                         "Group": GROUP_TEXT.get(groups.get(p, ""), ""),
                         "Glucose now": round(r["glucose"]),
                         "Forecast in 60 min": round(r.get("pred_60", np.nan)) if pd.notna(r.get("pred_60")) else None,
                         "Spike risk": float(r["p_spike"]) if pd.notna(r.get("p_spike")) else 0.0,
                         "Alert": bool(r.get("p_spike", 0) >= r.get("thr_spike", 1))})
        return pd.DataFrame(rows).sort_values("Spike risk", ascending=False) if rows else pd.DataFrame()

    ov = overview(day, clock.hour * 60 + clock.minute, CONFIG)
    if ov.empty:
        st.info("Run training first to see the clinic overview.")
    else:
        st.dataframe(ov, hide_index=True, column_config={
            "Spike risk": st.column_config.ProgressColumn("Spike risk", min_value=0.0, max_value=1.0, format="percent"),
        })

# ============================================================================ model evidence
with tab_model:
    which = "test" if svc.metrics_test else "cv"
    m = svc.metrics
    if not m:
        st.info("Run training to generate model evidence.")
    else:
        if which == "test":
            st.markdown(f"Results on the **{m.get('patients', '')} locked test patients**, set aside before any "
                        "training or tuning. Every model, threshold and weight was chosen on development patients.")
            if SYNTH:
                st.info("These are the team's results on the real CGMacros data, computed on the team's "
                        "computer. The patients elsewhere in this demo are synthetic.")
        else:
            st.markdown("Development results from **leave-patients-out cross-validation** (each patient scored by "
                        "a model trained without them). Run `python -m glucotwin.test` for the final test results.")
        hs = [str(h) for h in svc.horizons]
        reg = m["regression"]
        rows = [{"Model": b, **{f"{h} min": round(m["baselines"][h][b]["rmse"], 1) for h in hs}}
                for b in m["baselines"][hs[0]]]
        rows.append({"Model": "GlucoTwin", **{f"{h} min": round(reg[h]["rmse"], 1) for h in hs}})
        if m.get("personalization"):
            rows.append({"Model": "GlucoTwin, personalized",
                         **{f"{h} min": round(m["personalization"][h]["personalized_rmse"], 1)
                            for h in hs if h in m["personalization"]}})
        st.subheader("Forecast error (RMSE, mg/dL)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        ci = m.get("bootstrap_ci_95", {}).get("rmse", {})
        if ci:
            st.caption("95% intervals from resampling patients: " + "; ".join(
                f"{h} min {v[0]:.1f} to {v[1]:.1f}" for h, v in ci.items()))
        c = m.get("classification", {}).get("spike")
        e = m.get("event_level_spike", {})
        if c and "auroc" in c:
            st.subheader("Spike alerts")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AUROC", f"{c['auroc']:.2f}")
            k2.metric("Spikes caught in advance", f"{100 * e.get('event_recall', 0):.0f}%")
            lead = e.get("median_lead_time_min")
            k3.metric("Median warning", f"{lead:.0f} min" if lead else "n/a")
            k4.metric("False alerts per day", f"{e.get('false_alerts_per_day', 0):.1f}")
        figs = svc.figure_dirs[which]
        imgs = [p for p in ["rmse_by_horizon.png", "clarke_60min.png", "personalization.png",
                            "spike_calibration.png", "example_forecast.png"] if (figs / p).exists()]
        imp = svc.figure_dirs["cv"] / "feature_groups.png"
        cols = st.columns(2)
        for i, p in enumerate(imgs):
            cols[i % 2].image(str(figs / p))
        if imp.exists():
            cols[len(imgs) % 2].image(str(imp))

# ============================================================================ live ticker
# A fragment that reruns itself every `speed` seconds; each time it asks the whole page to move
# one reading forward. It is only drawn while live mode is on, so switching it off stops the clock.
if live:
    # Subtract the time this run took, so a reading arrives about every `speed` seconds.
    wait = max(0.3, float(speed) - (time.time() - RUN_START))
    st.session_state["_last_tick"] = time.time()

    @st.fragment(run_every=wait)
    def _live_ticker() -> None:
        if time.time() - st.session_state.get("_last_tick", 0.0) >= 0.8 * wait:
            st.session_state["_advance"] = True
            st.rerun()

    _live_ticker()
