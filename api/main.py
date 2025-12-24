# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.app_service import (
    run_random_mode,
    run_forecast_mode,
    run_evaluation_mode
)

from src.schema import (
    RandomRequest,
    ForecastRequest,
    EvaluationRequest,
    PredictionResponse
)

from src.external import build_external_features
from src.config import PROJECT_ROOT, RAW_DIR, PROCESS_DIR


# =====================================================
# APP INIT
# =====================================================

app = FastAPI(title="Rainfall Forecasting API")

# =====================================================
# LOAD EXTERNAL FEATURES ONCE
# =====================================================

# contoh: kamu load CSV eksternal di sini
external_sources = {
    "oni": pd.read_csv(RAW_DIR / "Data Eksternal/OceanicNinoIndex (ONI).csv"),
    "dmi": pd.read_csv(RAW_DIR / "Data Eksternal/Dipole Mode Index (DMI).csv"),
    "aqi": pd.read_csv(RAW_DIR / "Data Eksternal/AirQualityIndex_Google Trends.csv"),
    "rh":  pd.read_csv(RAW_DIR / "Data Eksternal/RelativeHumidityMonthlyMean.csv"),
}

external_df = build_external_features(external_sources)
train = pd.read_csv(PROCESS_DIR/'train.csv')
test = pd.read_csv(PROCESS_DIR/'test.csv')

train_min_date = train["date"].min()
train_max_date = train["date"].max()

# =====================================================
# REQUEST SCHEMAS
# =====================================================

class RandomRequest(BaseModel):
    features: dict


class ForecastRequest(BaseModel):
    location: str
    date: str


class EvaluationRequest(BaseModel):
    location: str
    date: str


# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Rainfall API is running"}


@app.post("/random", response_model=PredictionResponse)
def random_mode(req: RandomRequest):
    return run_random_mode(
        user_input=req.features,
        external_df=external_df
    )


@app.post("/forecast", response_model=PredictionResponse)
def forecast_mode(req: ForecastRequest):
    return run_forecast_mode(
        location=req.location,
        date=req.date,
        external_df=external_df
    )


@app.post("/evaluate", response_model=PredictionResponse)
def evaluation_mode(req: EvaluationRequest):
    return run_evaluation_mode(
        location=req.location,
        date=req.date,
        external_df=external_df,
        train_df=train,
        test_df=test
    )

@app.get("/validity")
def get_validity():
    return {
        "locations": sorted(list(known_locations)),
        "evaluation": {
            "min_date": "2009-01-01",
            "max_date": "2025-06-30"
        },
        "forecast": {
            "min_date": today,
            "max_date": today + 14
        }
    }