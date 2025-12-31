import pandas as pd
import numpy as np
import os

from src.config import config
from src.dataset_builder import align_to_daily

external_folder = config.paths.raw/'Data Eksternal'
external_csv = list(external_folder.glob("*.csv"))

expanded_dfs = []

for path in external_csv:
    df = pd.read_csv(path)

    feature = os.path.splitext(os.path.basename(path))[0]
    df.columns = ["date", feature]

    dt_format = "%d/%m/%Y" if "ONI" in feature else None
    df["date"] = pd.to_datetime(df["date"], format=dt_format)

    expanded_dfs.append(df)

expanded_dfs = [align_to_daily(df) for df in expanded_dfs]
all_dates = pd.concat(
    [df[["date"]] for df in expanded_dfs]
)["date"].drop_duplicates().sort_values()

external_df = pd.DataFrame({"date": all_dates})
for df in expanded_dfs:
    external_df = external_df.merge(df, on="date", how="left")

external_df['date'] = pd.to_datetime(external_df['date'], format="mixed").dt.strftime("%Y-%m-%d")
feature_map = {
    "date": "date",
    "AirQualityIndex_Google Trends" : "AQI",
    "Dipole Mode Index (DMI)" : "DMI",
    "OceanicNinoIndex (ONI)" : "ONI",
    "RelativeHumidityMonthlyMean" : "RH"
}

external_df.columns = external_df.columns.map(feature_map)
external_df = (external_df
               .replace(-9999, np.nan)
               .bfill()
               .ffill()
)

for column in list(feature_map.values())[1:]:
    for lag in [1, 2, 3, 6]:
        external_df[f'{column}_lag_{lag}'] = external_df[column].shift(lag)
    for window in [3, 6, 12]:
        external_df[f'{column}_rolling_mean_{window}'] = external_df[column].rolling(window, min_periods=1).mean()

external_df = external_df.bfill().ffill()
external_df.to_csv(config.paths.clean/"external_features.csv", index=False)

assert external_df["date"].is_unique
assert external_df["date"].is_monotonic_increasing

print("External features built:")
print(external_df.shape)
print(external_df.head())
