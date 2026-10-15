"""GlucoTwin REST API.

    uvicorn app.api:app --reload        # then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from glucotwin.meals import compose_meal, meal_table  # noqa: E402
from glucotwin.service import get_service  # noqa: E402

app = FastAPI(title="GlucoTwin API", version="0.1.0",
              description="Digital twin that forecasts post-meal glucose and simulates what-if scenarios. "
                          "Research prototype: not a medical device.")


class Macros(BaseModel):
    carbs: float = Field(0, ge=0, le=400)
    protein: float = Field(0, ge=0, le=300)
    fat: float = Field(0, ge=0, le=300)
    fiber: float = Field(0, ge=0, le=100)
    meal_kcal: float = Field(0, ge=0, le=4000)


class Scenario(BaseModel):
    label: str = "Scenario"
    meal: Optional[Macros] = None                 # None keeps the logged meal
    indian_meal: Optional[dict[str, float]] = None  # {"Idli": 2, "Sambar": 1}
    walk_minutes: int = Field(0, ge=0, le=120)
    walk_mets: float = Field(3.5, ge=1.5, le=10)
    walk_start_min: int = Field(0, ge=0, le=90)


class WhatIfRequest(BaseModel):
    at: str
    personalized: bool = False
    scenarios: list[Scenario]


def _check(pid: int):
    svc = get_service()
    if pid not in set(svc.dataset["pid"].unique()):
        raise HTTPException(404, f"Unknown patient {pid}")
    return svc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/patients")
def patients() -> list[dict]:
    return get_service().patients().fillna(0).to_dict(orient="records")


@app.get("/patients/{pid}/ehr")
def ehr(pid: int) -> dict:
    return _check(pid).ehr(pid)


@app.get("/patients/{pid}/timeline")
def timeline(pid: int, start: Optional[str] = None, end: Optional[str] = None) -> list[dict]:
    d = _check(pid).timeline(pid, start, end)
    d["timestamp"] = d["timestamp"].astype(str)
    return d.astype(object).where(d.notna(), None).to_dict(orient="records")


@app.get("/patients/{pid}/forecast")
def forecast(pid: int, at: Optional[str] = None, personalized: bool = False) -> dict:
    return _check(pid).forecast(pid, at, personalized)


@app.post("/patients/{pid}/whatif")
def whatif(pid: int, req: WhatIfRequest) -> list[dict]:
    svc = _check(pid)
    scenarios = []
    for sc in req.scenarios:
        meal = sc.meal.model_dump() if sc.meal else None
        if sc.indian_meal:
            try:
                meal = compose_meal(sc.indian_meal)
            except KeyError as exc:
                raise HTTPException(422, str(exc)) from exc
        scenarios.append({"label": sc.label, "meal": meal, "walk_minutes": sc.walk_minutes,
                          "walk_mets": sc.walk_mets, "walk_start_min": sc.walk_start_min})
    return svc.whatif(pid, req.at, scenarios, req.personalized)


@app.get("/meals/indian")
def indian_meals() -> list[dict]:
    return meal_table()


@app.get("/model/metrics")
def model_metrics() -> dict:
    return get_service().metrics
