import pandas as pd
import joblib
from pathlib import Path
from fastapi import HTTPException

from app.services.pipeline import build_features_from_api
from src.model import inference_data
from app.services.observed import get_observed_daily_rainfall

from datetime import (
    datetime,
    timedelta
)
from zoneinfo import ZoneInfo

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    return model


def get_last_observed_date_sg():
    now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
    return (now_sg.date() - timedelta(days=1)).strftime("%Y-%m-%d")


def run_random_mode(
    model,
    user_input: dict,
) -> pd.DataFrame:
    if "date" not in user_input:
        raise ValueError("Random mode requires 'date' for external features.")

    date = user_input["date"]
    
    X = pd.DataFrame([user_input])
    result = inference_data(model, X)
    result["feature_source"] = "random_user_input"

    return format_response(
        mode="random",
        location=user_input['location'],
        date=date,
        pred_mm=result["predicted_daily_rainfall_mm"].iloc[0],
        feature_source="random_user_input",
        notes=["Scenario-based prediction"]
    )


def validate_forecast_date(date_str: str):
    today = datetime.now(ZoneInfo("Asia/Singapore")).date()
    max_forecast = today + timedelta(days=14)

    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if date < today or date > max_forecast:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Forecast date must be between "
                f"{today} and {max_forecast}"
            )
        )


def run_forecast_mode(
    model,
    location: str,
    date: str,
) -> pd.DataFrame:
    validate_forecast_date(date)

    X = build_features_from_api(location, date)
    result = inference_data(model, X)
    return format_response(
        mode="forecast",
        location=location,
        date=date,
        pred_mm=result["predicted_daily_rainfall_mm"].iloc[0],
        feature_source="open_meteo",
        notes=["Forecast limited to short-term weather window"]
    )


def run_evaluation_mode(
    model,
    location: str,
    date: str,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_source = None

    if train_df is not None:
        row = train_df.loc[
            (train_df["location"] == location) &
            (train_df["date"] == date)
        ]
        if not row.empty:
            X = row.drop(columns=["daily_rainfall_total_mm"]).copy()
            feature_source = "train_dataset"


    if feature_source is None and test_df is not None:
        row = test_df.loc[
            (test_df["location"] == location) &
            (test_df["date"] == date)
        ]
        if not row.empty:
            X = row.drop(columns=["daily_rainfall_total_mm"]).copy()
            feature_source = "train_dataset"


    if feature_source is None:
        X = build_features_from_api(location, date)
        feature_source = "open_meteo"

    pred_df = inference_data(model, X)
    observed_df = get_observed_daily_rainfall(location, date)
    if observed_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Observed rainfall not available for this date"
        )

    result = pred_df.copy()
    result["observed_daily_rainfall_mm"] = (
        observed_df["daily_rainfall_total_mm"].iloc[0]
    )
    result["error_mm"] = (
        result["predicted_daily_rainfall_mm"]
        - result["observed_daily_rainfall_mm"]
    )
    result["feature_source"] = feature_source

    pred_mm = pred_df["predicted_daily_rainfall_mm"].iloc[0]
    obs_mm = observed_df["daily_rainfall_total_mm"].iloc[0]

    return format_response(
        mode="evaluation",
        location=location,
        date=date,
        pred_mm=pred_mm,
        observed_mm=obs_mm,
        error_mm=pred_mm - obs_mm,
        feature_source=feature_source,
        notes=[
            "Historical evaluation using best available features"
            if feature_source != "open_meteo"
            else "Recent evaluation using Open-Meteo features"
        ]
    )


def format_response(
    mode,
    location,
    date,
    pred_mm,
    observed_mm=None,
    error_mm=None,
    feature_source=None,
    notes=None,
):
    return {
        "mode": mode,
        "input": {
            "location": location,
            "date": date
        },
        "prediction": {
            "daily_rainfall_mm": round(float(pred_mm), 3)
        },
        "comparison": None if observed_mm is None else {
            "observed_daily_rainfall_mm": round(float(observed_mm), 3),
            "error_mm": round(float(error_mm), 3)
        },
        "meta": {
            "feature_source": feature_source,
            "notes": notes or []
        }
    }