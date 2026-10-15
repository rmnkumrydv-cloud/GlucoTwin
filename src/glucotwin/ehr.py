"""Static stream: build a FHIR R4-style Bundle per patient and read features back out of it.

Why FHIR? India's Ayushman Bharat Digital Mission (ABDM) exchanges health records as FHIR
resources, so a twin that consumes FHIR-shaped EHRs is structurally ready to plug in.

Real CGMacros values (demographics, HbA1c, fasting labs) become Patient + Observation
resources. The challenge brief also asks for past diagnoses, medications and genetic markers,
which CGMacros does not contain; ``add_synthetic_fields`` generates plausible ones and marks
every such resource with a ``synthetic`` extension so nobody mistakes them for real data.
By default the model does NOT learn from synthetic fields (they are random, so they carry
no real signal and would only add noise or fake "insights").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_EXT = {"url": "https://glucotwin.local/fhir/synthetic", "valueBoolean": True}

# (feature name, LOINC code, display, unit)
LAB_OBSERVATIONS = [
    ("hba1c", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    ("fasting_glucose", "1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("fasting_insulin", "20448-7", "Insulin [Units/volume] in Serum or Plasma", "uU/mL"),
    ("triglycerides", "2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("cholesterol", "2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("hdl", "2085-9", "HDL Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("ldl", "13457-7", "LDL Cholesterol (calculated)", "mg/dL"),
    ("bmi", "39156-5", "Body mass index", "kg/m2"),
    ("weight_kg", "29463-7", "Body weight", "kg"),
    ("height_cm", "8302-2", "Body height", "cm"),
]
_LOINC_TO_FEATURE = {code: name for name, code, _, _ in LAB_OBSERVATIONS}

DIAGNOSIS = {
    "healthy": None,
    "prediabetes": ("714628002", "Prediabetes", "R73.03"),
    "t2d": ("44054006", "Diabetes mellitus type 2", "E11.9"),
}


def diagnosis_from_hba1c(a1c: float) -> str:
    """ADA thresholds, the same ones CGMacros used to group participants."""
    if a1c is None or np.isnan(a1c):
        return "unknown"
    if a1c < 5.7:
        return "healthy"
    return "prediabetes" if a1c < 6.5 else "t2d"


def _obs(pid: int, name: str, code: str, display: str, unit: str, value: float) -> dict:
    return {
        "resourceType": "Observation",
        "id": f"{pid}-{name}",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{pid}"},
        "valueQuantity": {"value": round(float(value), 2), "unit": unit},
    }


def build_ehr_record(row: pd.Series, rng: np.random.Generator, add_synthetic: bool = True) -> dict:
    pid = int(row["pid"])
    entries: list[dict] = []
    patient = {
        "resourceType": "Patient",
        "id": str(pid),
        "gender": row.get("sex") if isinstance(row.get("sex"), str) else "unknown",
        "extension": [{"url": "https://glucotwin.local/fhir/age-years", "valueInteger": int(row["age"])}]
        if pd.notna(row.get("age")) else [],
    }
    entries.append(patient)

    values = dict(row)
    if pd.notna(values.get("weight_lb")):
        values["weight_kg"] = values["weight_lb"] * 0.453592
    if pd.notna(values.get("height_in")):
        values["height_cm"] = values["height_in"] * 2.54
    for name, code, display, unit in LAB_OBSERVATIONS:
        v = values.get(name)
        if v is not None and pd.notna(v):
            entries.append(_obs(pid, name, code, display, unit, v))

    dx = diagnosis_from_hba1c(values.get("hba1c", np.nan))
    if DIAGNOSIS.get(dx):
        snomed, text, icd10 = DIAGNOSIS[dx]
        entries.append({
            "resourceType": "Condition", "id": f"{pid}-glycemic",
            "subject": {"reference": f"Patient/{pid}"},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": text},
                                {"system": "http://hl7.org/fhir/sid/icd-10", "code": icd10}],
                     "text": text},
            "note": [{"text": "Derived from baseline HbA1c using ADA thresholds."}],
        })

    if add_synthetic:
        entries.extend(_synthetic_resources(pid, values, dx, rng))

    return {"resourceType": "Bundle", "type": "collection", "id": f"ehr-{pid}",
            "entry": [{"resource": r} for r in entries]}


def _synthetic_resources(pid: int, v: dict, dx: str, rng: np.random.Generator) -> list[dict]:
    """Plausible, clearly flagged synthetic fields requested by the challenge brief."""
    out: list[dict] = []
    age = v.get("age") if pd.notna(v.get("age")) else 45
    bmi = v.get("bmi") if pd.notna(v.get("bmi")) else 27
    p_htn = float(np.clip(0.1 + 0.01 * (age - 30) + 0.02 * (bmi - 25), 0.05, 0.8))
    if rng.random() < p_htn:
        out.append({"resourceType": "Condition", "id": f"{pid}-htn", "extension": [SYNTHETIC_EXT],
                    "subject": {"reference": f"Patient/{pid}"},
                    "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003",
                                         "display": "Hypertensive disorder"}], "text": "Hypertension"}})
    meds = []
    if dx == "t2d":
        meds.append(("metformin", "Metformin 500 mg twice daily"))
        if rng.random() < 0.3:
            meds.append(("glimepiride", "Glimepiride 1 mg once daily (sulfonylurea)"))
    elif dx == "prediabetes" and rng.random() < 0.25:
        meds.append(("metformin", "Metformin 500 mg once daily"))
    for key, text in meds:
        out.append({"resourceType": "MedicationStatement", "id": f"{pid}-{key}", "extension": [SYNTHETIC_EXT],
                    "status": "active", "subject": {"reference": f"Patient/{pid}"},
                    "medicationCodeableConcept": {"text": text}})
    # TCF7L2 rs7903146: the strongest common genetic T2D risk variant; risk-allele count 0/1/2.
    risk_alleles = int(rng.choice([0, 1, 2], p=[0.55, 0.37, 0.08]))
    out.append({"resourceType": "Observation", "id": f"{pid}-tcf7l2", "extension": [SYNTHETIC_EXT],
                "status": "final", "subject": {"reference": f"Patient/{pid}"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "69548-6",
                                     "display": "Genetic variant assessment"}],
                         "text": "TCF7L2 rs7903146 risk-allele count"},
                "valueInteger": risk_alleles})
    return out


def build_all_ehr(bio: pd.DataFrame, cfg: dict, out_dir: Path) -> dict[int, dict]:
    ecfg = cfg["ehr"]
    rng = np.random.default_rng(ecfg["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for _, row in bio.iterrows():
        rec = build_ehr_record(row, rng, ecfg["add_synthetic_fields"])
        records[int(row["pid"])] = rec
        (out_dir / f"patient_{int(row['pid']):03d}.json").write_text(json.dumps(rec, indent=2))
    return records


def load_ehr_records(ehr_dir: Path) -> dict[int, dict]:
    recs = {}
    for p in sorted(Path(ehr_dir).glob("patient_*.json")):
        rec = json.loads(p.read_text())
        recs[int(rec["id"].split("-")[1])] = rec
    return recs


# ----------------------------------------------------------------------------- EHR -> features
def ehr_to_features(bundle: dict, use_synthetic: bool = False) -> dict:
    """Parse a FHIR-style bundle into a flat static feature vector for the model."""
    f: dict = {}
    for e in bundle["entry"]:
        r = e["resource"]
        synthetic = any(x.get("url") == SYNTHETIC_EXT["url"] for x in r.get("extension", []))
        if synthetic and not use_synthetic:
            continue
        rt = r["resourceType"]
        if rt == "Patient":
            f["sex_male"] = 1.0 if r.get("gender") == "male" else 0.0
            for x in r.get("extension", []):
                if x["url"].endswith("age-years"):
                    f["age"] = float(x["valueInteger"])
        elif rt == "Observation":
            code = r["code"]["coding"][0]["code"]
            if code in _LOINC_TO_FEATURE and "valueQuantity" in r:
                f[_LOINC_TO_FEATURE[code]] = float(r["valueQuantity"]["value"])
            elif code == "69548-6":
                f["tcf7l2_risk_alleles"] = float(r["valueInteger"])
        elif rt == "Condition":
            text = r["code"].get("text", "").lower()
            if "type 2" in text:
                f["dx_t2d"] = 1.0
            elif "prediabetes" in text:
                f["dx_prediabetes"] = 1.0
            elif "hypertens" in text:
                f["dx_hypertension"] = 1.0
        elif rt == "MedicationStatement":
            text = r["medicationCodeableConcept"]["text"].lower()
            if "metformin" in text:
                f["med_metformin"] = 1.0
            if "glimepiride" in text or "sulfonylurea" in text:
                f["med_sulfonylurea"] = 1.0
    f.setdefault("dx_t2d", 0.0)
    f.setdefault("dx_prediabetes", 0.0)
    if "fasting_glucose" in f and "fasting_insulin" in f:
        f["homa_ir"] = f["fasting_glucose"] * f["fasting_insulin"] / 405.0
    if "triglycerides" in f and "hdl" in f and f["hdl"] > 0:
        f["tg_hdl_ratio"] = f["triglycerides"] / f["hdl"]
    return f


def static_feature_table(records: dict[int, dict], use_synthetic: bool = False) -> pd.DataFrame:
    rows = []
    for pid, rec in records.items():
        f = ehr_to_features(rec, use_synthetic)
        f["pid"] = pid
        rows.append(f)
    df = pd.DataFrame(rows).set_index("pid").sort_index()
    df.columns = [f"ehr_{c}" for c in df.columns]
    return df


def ehr_summary(bundle: dict) -> dict:
    """Human-readable summary for the dashboard."""
    s = {"conditions": [], "medications": [], "labs": {}, "synthetic_fields": []}
    for e in bundle["entry"]:
        r = e["resource"]
        synthetic = any(x.get("url") == SYNTHETIC_EXT["url"] for x in r.get("extension", []))
        rt = r["resourceType"]
        if rt == "Patient":
            s["gender"] = r.get("gender")
            for x in r.get("extension", []):
                if x["url"].endswith("age-years"):
                    s["age"] = x["valueInteger"]
        elif rt == "Condition":
            s["conditions"].append(r["code"]["text"] + (" (synthetic)" if synthetic else ""))
        elif rt == "MedicationStatement":
            s["medications"].append(r["medicationCodeableConcept"]["text"] + (" (synthetic)" if synthetic else ""))
        elif rt == "Observation":
            label = r["code"].get("text") or r["code"]["coding"][0]["display"]
            if "valueQuantity" in r:
                s["labs"][label] = f"{r['valueQuantity']['value']} {r['valueQuantity']['unit']}"
            elif "valueInteger" in r:
                s["labs"][label] = str(r["valueInteger"]) + (" (synthetic)" if synthetic else "")
        if synthetic:
            s["synthetic_fields"].append(r["id"])
    return s
