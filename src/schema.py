# src/schema.py
from pydantic import BaseModel
from typing import Optional, List


class RandomRequest(BaseModel):
    location: str
    date: str

    mean_temperature_c: float
    maximum_temperature_c: float
    minimum_temperature_c: float
    mean_wind_speed_kmh: float
    max_wind_speed_kmh: float

    highest_30_min_rainfall_mm: float
    highest_60_min_rainfall_mm: float
    highest_120_min_rainfall_mm: float


class ForecastRequest(BaseModel):
    location: str
    date: str


class EvaluationRequest(BaseModel):
    location: str
    date: str


class PredictionResponse(BaseModel):
    mode: str
    input: dict
    prediction: dict
    comparison: Optional[dict]
    meta: dict