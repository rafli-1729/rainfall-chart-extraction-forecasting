from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import config
from src.dataset_builder import convert_numeric
from src.model import (
    build_feature_pipeline,
    build_preprocessor,
    build_model,
    build_pipeline
)

if __name__ == "__main__":
    print("Loading data...")
    train = pd.read_csv(config.paths.clean / "train_v2.csv")
    external_df = pd.read_csv(config.paths.clean/ "external_features.csv")

    train.sort_values(["date", "location"], inplace=True)

    feature_cols = (
        config.features.meteorogical_columns +
        config.features.rain_extreme_columns
    )

    mask_all_features_missing = train[feature_cols].isna().all(axis=1)
    mask_target_present = train["daily_rainfall_total_mm"].notna()

    mask_drop = mask_all_features_missing & mask_target_present

    train = train.loc[~mask_drop].reset_index(drop=True)

    X = train.drop(columns=["daily_rainfall_total_mm"])
    y = train["daily_rainfall_total_mm"].fillna(0)

    pipe = build_pipeline(external_df=external_df,
                        model_type='two_stage')

    tscv = TimeSeriesSplit(n_splits=5)

    print("Cross validating...")
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    print("MAE per fold:", -scores)
    print("Mean MAE:", -scores.mean())

    print("Fitting final model...")
    pipe.fit(X, y)

    y_pred = pipe.predict(X)

    print("Mean Absolute Error :", mean_absolute_error(y, y_pred))
    print("Mean Squared Error  :", mean_squared_error(y, y_pred))

    joblib.dump(pipe, config.paths.models/'cv_model_v2.pkl')
    print("Model saved!")