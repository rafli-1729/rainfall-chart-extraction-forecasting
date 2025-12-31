import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class StructuralWeatherImputer(BaseEstimator, TransformerMixin):
    def __init__(self, location_col="location"):
        self.location_col = location_col

    def fit(self, X: pd.DataFrame, y=None):
        self.loc_median_ = (
            X.groupby(self.location_col)
             .median(numeric_only=True)
        )

        self.global_median_ = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in self.global_median_.index:
            if col not in df.columns:
                continue

            # Location-based fill
            df[col] = df[col].fillna(
                df[self.location_col].map(self.loc_median_[col])
            )

            # Global fallback
            df[col] = df[col].fillna(self.global_median_[col])

        return df


class AddExternalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, external_df: pd.DataFrame, on="date"):
        self.external_df = external_df
        self.on = on

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        X_ = X_.merge(
            self.external_df,
            on=self.on,
            how="left",
            validate="m:1"
        )

        return X_


class LocationMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, location_col='location'):
        self.cols = cols
        self.location_col = location_col

    def fit(self, X, y=None):
        X_ = X[[self.location_col] + self.cols]
        self.medians_ = (
            X_
            .groupby(self.location_col)[self.cols]
            .median()
        )
        self.global_median_ = X_[self.cols].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(
                X[self.location_col].map(self.medians_[col])
            )

            X[col] = X[col].fillna(self.global_median_[col])
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("Int64")

        return df


class TemperatureFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["temp_range_c"] = (
            df["maximum_temperature_c"] - df["minimum_temperature_c"]
        )
        return df


class WindRainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        gust = df["max_wind_speed_kmh"] / df["mean_wind_speed_kmh"]
        df["wind_gust_factor"] = gust.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1)

        ratio = (
            df["highest_60_min_rainfall_mm"]
            / df["highest_30_min_rainfall_mm"]
        )
        df["rain_intensity_ratio"] = ratio.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1)

        return df


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lag_days: int = 1):
        self.lag_days = lag_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in [
            "mean_temperature_c",
            "highest_60_min_rainfall_mm",
            "mean_wind_speed_kmh",
        ]:
            df[f"{col}_lag{self.lag_days}"] = df[col].shift(self.lag_days)
            df[f"{col}_lag{self.lag_days}"] = df[f"{col}_lag{self.lag_days}"].fillna(df[col])

        return df


class RollingStatsFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["mean_temp_roll_7d"] = (
            df["mean_temperature_c"].rolling(7, min_periods=1).mean()
        )
        df["max_rain_roll_3d"] = (
            df["highest_60_min_rainfall_mm"].rolling(3, min_periods=1).max()
        )
        df["mean_wind_roll_7d"] = (
            df["mean_wind_speed_kmh"].rolling(7, min_periods=1).mean()
        )

        return df


class CyclicalInteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        days = 366

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["day_of_year_sin"] = np.sin(
            2 * np.pi * df["day_of_year"] / days
        )
        df["day_of_year_cos"] = np.cos(
            2 * np.pi * df["day_of_year"] / days
        )

        df["wind_x_rain"] = (
            df["mean_wind_speed_kmh"]
            * df["highest_60_min_rainfall_mm"]
        )

        df["ONI_x_temp"] = df["ONI"] * df["mean_temperature_c"]
        df["DMI_x_rainfall"] = (
            df["DMI"] * df["highest_60_min_rainfall_mm"]
        )
        df["heat_index_proxy"] = (
            df["RH"] * df["mean_temperature_c"]
        )
        df["aqi_x_temp_range"] = (
            df["AQI"] * df["temp_range_c"]
        )

        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        return df.drop(columns=self.cols, errors="ignore")


class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.name, type(X), getattr(X, "shape", None))
        return X