from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from app.services.app_service import (
    run_random_mode,
    run_forecast_mode,
    run_evaluation_mode,
    load_model
)

from src.schema import (
    RandomRequest,
    ForecastRequest,
    EvaluationRequest,
    PredictionResponse
)

from src.config import config

# ===================================== APP INIT =====================================

app = FastAPI(title="Rainfall Forecasting API")
_model = load_model(model_path=config.paths.models/'cv_model_v2.pkl')

# ===================================== LOAD EXTERNAL FEATURES ONCE =====================================

external_df = pd.read_csv(config.paths.clean/'external_features.csv')
train = pd.read_csv(config.paths.processed/'train.csv')
test = pd.read_csv(config.paths.processed/'test.csv')

train_min_date = train["date"].min()
train_max_date = train["date"].max()

# ===================================== REQUEST SCHEMAS =====================================

class RandomRequest(BaseModel):
    features: dict


class ForecastRequest(BaseModel):
    location: str
    date: str


class EvaluationRequest(BaseModel):
    location: str
    date: str

# ===================================== ROUTES =====================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Rainfall API is running"}


@app.post("/random", response_model=PredictionResponse)
def random_mode(req: RandomRequest):
    return run_random_mode(
        model=_model,
        user_input=req.features,
    )


@app.post("/forecast", response_model=PredictionResponse)
def forecast_mode(req: ForecastRequest):
    return run_forecast_mode(
        model=_model,
        location=req.location,
        date=req.date,
    )


@app.post("/evaluate", response_model=PredictionResponse)
def evaluation_mode(req: EvaluationRequest):
    return run_evaluation_mode(
        model=_model,
        location=req.location,
        date=req.date,
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