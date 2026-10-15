"""Draw the GlucoTwin architecture diagram (docs/architecture.pdf + .png).

    python docs/make_architecture.py

Edit the text below to match your final system, then re-run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, TEAL, BLUE, AMBER, SLATE = "#1B2A41", "#0F766E", "#2F5D8A", "#B45309", "#64748B"
FILL = {TEAL: "#E6F2F1", BLUE: "#E8EEF6", AMBER: "#FBF0E6", SLATE: "#F1F4F6", INK: "#EEF1F5"}
OUT = Path(__file__).resolve().parent

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def box(x, y, w, h, title, body, color, title_size=10.5, body_size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                linewidth=1.4, edgecolor=color, facecolor=FILL[color]))
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=title_size, fontweight="bold", color=color, va="top")
    ax.text(x + 1.2, y + h - 5.0, body, fontsize=body_size, color=INK, va="top", linespacing=1.45)


def arrow(x1, y1, x2, y2, color=SLATE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color, shrinkA=2, shrinkB=2))


def column_title(x, text):
    ax.text(x, 83.5, text, fontsize=11, color=SLATE, fontweight="bold")


ax.text(3, 88, "GlucoTwin architecture", fontsize=20, fontweight="bold", color=INK, va="center")
ax.text(52, 88, "a digital twin that forecasts post-meal glucose spikes 2 hours ahead", fontsize=12,
        color=SLATE, va="center")

# columns: x positions
X = [3, 34, 65, 96, 128]
W = [27, 27, 27, 28, 29]
for x, t in zip(X, ["1  Open data", "2  Two streams", "3  Fusion layer", "4  Twin engine", "5  Doctor-facing"]):
    column_title(x, t)

# 1 data sources
box(X[0], 58, W[0], 23, "CGMacros bio.csv",
    "45 adults: healthy, prediabetes,\ntype 2 diabetes\nAge, sex, BMI, HbA1c,\nfasting glucose and insulin,\nlipid panel", BLUE)
box(X[0], 30, W[0], 25, "CGMacros minute data",
    "Dexcom G6 + Libre Pro CGM\nFitbit heart rate, METs,\nactivity calories\nLogged meals: carbs, protein,\nfat, fiber (10 days each)", TEAL)
box(X[0], 12, W[0], 15, "Synthetic EHR fields",
    "Flagged in every record:\ndiagnoses, medications,\nTCF7L2 genotype", SLATE)

# 2 streams
box(X[1], 50, W[1], 31, "Static stream: EHR",
    "One FHIR R4 bundle per patient\n(ABDM-compatible format)\n\nPatient, Observation (LOINC),\nCondition (SNOMED, ICD-10),\nMedicationStatement\n\nParsed back into\n~16 static features", BLUE)
box(X[1], 12, W[1], 35, "Dynamic stream: wearables",
    "Causal 5-minute grid: the row at t\nuses data up to t only\n\nOriginal CGM readings recovered\n(CGMacros interpolated to 1 min)\nLast reading held <=10 min;\nlonger gaps stay missing\nHR and METs averaged\nQuality report per patient\n\nLeakage tests on raw data", TEAL)

# 3 fusion
box(X[2], 12, W[2], 69, "Feature engineering",
    "Glucose now, lags, trend,\nacceleration, 1-6 h stats, EWMA,\n24 h baseline, time above 180\n\nCarbs on board and absorption\nrate (gamma absorption model)\nMeal macros in time windows,\ntime since meal, fat on board\n\nActivity: METs windows,\nactive minutes, heart rate\nrelative to resting\n\nTime of day, missingness flags\n\nEHR features: HbA1c, HOMA-IR,\nTG/HDL, BMI, age, diagnosis\n\nPlan inputs (what-if):\nplanned carbs, planned activity\nhidden at random in training\n\n93 features per step, plus\n6 h sequences for the network", INK)

# 4 engine
box(X[3], 62, W[3], 19, "LightGBM engine",
    "Glucose at 30/60/90/120 min\nSpike and hypo classifiers\nMonotone constraints: the\nsafe what-if engine", AMBER)
box(X[3], 40, W[3], 19, "P-MRGF neural engine",
    "TCN on 6 h CGM, FiLM from EHR\nGated residuals: meals,\nactivity, heart rate\n24-step trajectory + risk head", AMBER)
box(X[3], 26, W[3], 11, "Honest ensemble + band",
    "Per-horizon weights fixed on\ndevelopment patients; 80% band", AMBER)
box(X[3], 12, W[3], 11, "Personalization",
    "Per-patient residual model\nfrom the first days", AMBER)

# 5 interface
box(X[4], 62, W[4], 19, "What-if simulator",
    "Swap the meal, change portion,\nadd a walk and compare curves\nIndian meal library\n(values to verify vs IFCT 2017)", TEAL)
box(X[4], 45, W[4], 14, "Explanations",
    "TreeSHAP contributions grouped\ninto plain-language drivers", TEAL)
box(X[4], 26, W[4], 16, "Doctor dashboard (Streamlit)",
    "Forecast, what-if, health record,\nclinic overview ranked by risk,\nmodel evidence", TEAL)
box(X[4], 12, W[4], 11, "REST API (FastAPI)",
    "/forecast  /whatif  /ehr\nready for EHR/app integration", TEAL)

# arrows
arrow(X[0] + W[0], 69, X[1], 66, BLUE)
arrow(X[0] + W[0], 19, X[1], 58, SLATE)
arrow(X[0] + W[0], 42, X[1], 30, TEAL)
arrow(X[1] + W[1], 65, X[2], 60, BLUE)
arrow(X[1] + W[1], 29, X[2], 34, TEAL)
for y in (71, 49, 31, 17):
    arrow(X[2] + W[2], 46, X[3], y, INK)
for y1, y2 in ((71, 71), (49, 52), (31, 34), (17, 17)):
    arrow(X[3] + W[3], y1, X[4], y2, AMBER)

# validation bar
ax.add_patch(FancyBboxPatch((3, 1.5), 154, 7, boxstyle="round,pad=0.3,rounding_size=1.2",
                            linewidth=1.2, edgecolor=SLATE, facecolor="#FFFFFF"))
ax.text(4.5, 5, "Validation", fontsize=10.5, fontweight="bold", color=SLATE, va="center")
ax.text(19, 5, "9 locked test patients scored once  |  5-fold patient CV on 36 development patients  |  purged "
               "personalization  |  baselines  |  RMSE, R2 of change, Clarke grid, AUROC, AUPRC, Brier, false alerts/day",
        fontsize=9.2, color=INK, va="center")

fig.savefig(OUT / "architecture.pdf")
fig.savefig(OUT / "architecture.png", dpi=150)
print("wrote", OUT / "architecture.pdf", "and .png")
