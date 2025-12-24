import pandas as pd
import joblib
from pathlib import Path
from fastapi import HTTPException

from src.pipeline import build_features_from_api
from src.external import get_external_features_for_date, build_external_features
from src.model import inference_data
from src.observed import get_observed_daily_rainfall

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import MODEL_DIR, RAW_DIR, PROCESS_DIR
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    return model

_model = load_model(MODEL_PATH)


def get_last_observed_date_sg():
    now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
    return (now_sg.date() - timedelta(days=1)).strftime("%Y-%m-%d")


def run_random_mode(
    user_input: dict,
    external_df: pd.DataFrame
) -> pd.DataFrame:
    if "date" not in user_input:
        raise ValueError("Random mode requires 'date' for external features.")

    date = user_input["date"]

    X = pd.DataFrame([user_input])

    external_feats = get_external_features_for_date(
        external_df=external_df,
        date=date
    )
    X = X.assign(**external_feats)

    result = inference_data(_model, X)
    result["feature_source"] = "random_user_input"

    return format_response(
        mode="random",
        location=user_input['location'],
        date=date,
        pred_mm=result["predicted_daily_rainfall_mm"].iloc[0],
        feature_source="random_user_input",
        external_month_used=external_feats.get("external_month"),
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
    location: str,
    date: str,
    external_df: pd.DataFrame
) -> pd.DataFrame:
    validate_forecast_date(date)
    
    X = build_features_from_api(location, date)

    external_feats = get_external_features_for_date(
        external_df=external_df,
        date=date
    )

    X = X.assign(**external_feats)

    result = inference_data(_model, X)
    return format_response(
        mode="forecast",
        location=location,
        date=date,
        pred_mm=result["predicted_daily_rainfall_mm"].iloc[0],
        feature_source="open_meteo",
        external_month_used=external_feats.get("external_month"),
        notes=["Forecast limited to short-term weather window"]
    )


def run_evaluation_mode(
    location: str,
    date: str,
    external_df: pd.DataFrame,
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

            external_feats = get_external_features_for_date(
                external_df=external_df,
                date=date
            )
            X = X.assign(**external_feats)

            feature_source = "train_dataset"


    if feature_source is None and test_df is not None:
        row = test_df.loc[
            (test_df["location"] == location) &
            (test_df["date"] == date)
        ]
        if not row.empty:
            X = row.drop(columns=["daily_rainfall_total_mm"]).copy()

            external_feats = get_external_features_for_date(
                external_df=external_df,
                date=date
            )
            X = X.assign(**external_feats)

            feature_source = "train_dataset"


    if feature_source is None:
        X = build_features_from_api(location, date)

        external_feats = get_external_features_for_date(
            external_df=external_df,
            date=date
        )
        X = X.assign(**external_feats)
        feature_source = "open_meteo"

    pred_df = inference_data(_model, X)
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
        external_month_used=external_feats.get("external_month"),
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
    external_month_used=None,
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
            "external_month_used": external_month_used,
            "notes": notes or []
        }
    }


if __name__ == '__main__':
    dmi = pd.read_csv(RAW_DIR/'Data Eksternal/Dipole Mode Index (DMI).csv')
    aqi = pd.read_csv(RAW_DIR/'Data Eksternal/AirQualityIndex_Google Trends.csv')
    oni = pd.read_csv(RAW_DIR/'Data Eksternal/OceanicNinoIndex (ONI).csv')
    rh = pd.read_csv(RAW_DIR/'Data Eksternal/RelativeHumidityMonthlyMean.csv')

    external_df = build_external_features({
        "dmi": dmi,
        "aqi": aqi,
        "oni": oni,
        "rh": rh,
    })

    train = pd.read_csv(PROCESS_DIR/'train.csv')
    test = pd.read_csv(PROCESS_DIR/'test.csv')

    print(run_forecast_mode("Admiralty", "2025-09-23", external_df))
    print(run_evaluation_mode("Admiralty", "2025-01-10",
                              external_df, train_df=train, test_df=test))
    random_input = {
        "location": "Admiralty",
        "date": "2025-01-15",

        "mean_temperature_c": 27.5,
        "maximum_temperature_c": 30.1,
        "minimum_temperature_c": 24.3,
        "mean_wind_speed_kmh": 8.2,
        "max_wind_speed_kmh": 15.0,

        "highest_30_min_rainfall_mm": 10.0,
        "highest_60_min_rainfall_mm": 18.0,
        "highest_120_min_rainfall_mm": 25.0,
        }

    print(run_random_mode(random_input, external_df))