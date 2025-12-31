import pandas as pd
import numpy as np
import pandas
import joblib

from src.config import config

def main():
    cv_model = joblib.load(config.paths.models/"cv_model.pkl")
    obs_model = joblib.load(config.paths.models/"obs_model.pkl")

    train = pd.read_csv(config.paths.clean/"train.csv")
    wss = pd.read_csv(config.paths.clean/"wss.csv")
    wss = wss[["date", "location", "daily_rainfall_total_mm"]]

    train["cv_predicted_mm"] = cv_model.predict(train.drop(columns=["daily_rainfall_total_mm"]))
    train["obs_predicted_mm"] = obs_model.predict(train.drop(columns=["daily_rainfall_total_mm"]))

    feature_cols = (
        config.features.meteorogical_columns +
        config.features.rain_extreme_columns
    )

    mask_all_features_missing = train[feature_cols].isna().all(axis=1)
    mask_target_present = train["daily_rainfall_total_mm"].notna()

    mask_set_nan = mask_all_features_missing & mask_target_present
    train.loc[mask_set_nan, "cv_predicted_mm"] = np.nan
    train.loc[mask_set_nan, "obs_predicted_mm"] = np.nan

    train["date"] = pd.to_datetime(train["date"])
    wss["date"]   = pd.to_datetime(wss["date"])

    train.rename(columns={
        "daily_rainfall_total_mm": "extracted_mm"
    }, inplace=True)

    wss.rename(columns={
        "daily_rainfall_total_mm": "observed_mm"
    }, inplace=True)

    train["location"] = train["location"].str.replace("_", " ")
    train_locations = train["location"].unique()
    wss = wss[wss["location"].isin(train_locations)]

    daily = (
        train[["date", "location", "extracted_mm", "cv_predicted_mm", "obs_predicted_mm"]]
        .merge(
            wss[["date", "location", "observed_mm"]],
            on=["date", "location"],
            how="left"
        )
    )

    daily.sort_values(['location', 'date'], inplace=True)
    daily.dropna(subset=["observed_mm"]).to_csv(config.paths.clean/"daily.csv", index=False)


if __name__ == "__main__":
    main()