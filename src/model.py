from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import (
    CLEAN_DIR,
    MODEL_DIR,
    PROCESS_DIR,
    RAIN_EXTREME_COLUMNS,
    METEOROGICAL_COLUMNS
)

from src.features import (
    TimeFeatures,
    TemperatureFeatures,
    WindRainFeatures,
    LagFeatures,
    RollingStatsFeatures,
    CyclicalInteractionFeatures,
    DropFeatures,
    DebugTransformer,
    StructuralWeatherImputer
)


def build_feature_pipeline():
    return Pipeline(steps=[
        ("structural_imputer", StructuralWeatherImputer()),
        ("time", TimeFeatures()),
        ("temp", TemperatureFeatures()),
        ("wind_rain", WindRainFeatures()),
        ("lag", LagFeatures(lag_days=1)),
        ("rolling", RollingStatsFeatures()),
        ("cyclical", CyclicalInteractionFeatures()),
        ("drop", DropFeatures(['date']))
    ])


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ['location']),
            ("num", "passthrough",
             METEOROGICAL_COLUMNS + RAIN_EXTREME_COLUMNS),
        ],
        remainder="drop"
    )


def build_model(xgb_params=None, transform_target=False):
    default_params = dict(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    if xgb_params:
        default_params.update(xgb_params)

    model = XGBRegressor(**default_params)

    if transform_target:
        model = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return model


def build_pipeline(xgb_params=None):
    return Pipeline(steps=[
        ("features", build_feature_pipeline()),
        ("preprocess", build_preprocessor()),
        ("model", build_model(xgb_params))
    ])


def inference_data(
    model,
    data: pd.DataFrame,
    clip_negative: bool = True,
    return_dataframe: bool = True
):
    preds = model.predict(data)

    if clip_negative:
        preds = np.clip(preds, 0, None)

    if return_dataframe:
        result = data.copy()
        result["predicted_daily_rainfall_mm"] = preds
        return result

    return preds

if __name__ == '__main__':
    print("Loading data...")
    train = pd.read_csv(CLEAN_DIR / "train.csv")

    train.sort_values(["date", "location"], inplace=True)

    X = train.drop(columns=["daily_rainfall_total_mm"])
    y = train["daily_rainfall_total_mm"].fillna(0)

    pipe = build_pipeline()

    tscv = TimeSeriesSplit(n_splits=5)

    print("Cross validating...")
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    print("MSE per fold:", -scores)
    print("Mean MSE:", -scores.mean())

    print("Fitting final model...")
    pipe.fit(X, y)
