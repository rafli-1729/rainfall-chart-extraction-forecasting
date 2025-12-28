import pandas as pd
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

feature_map = {
    "date": "date",
    "AirQualityIndex_Google Trends" : "AQI",
    "Dipole Mode Index (DMI)" : "DMI",
    "OceanicNinoIndex (ONI)" : "ONI",
    "RelativeHumidityMonthlyMean" : "RH"
}

external_df.columns = external_df.columns.map(feature_map)
external_df.to_csv(config.paths.processed/"external_features.csv", index=False)

assert external_df["date"].is_unique
assert external_df["date"].is_monotonic_increasing

print("External features built:")
print(external_df.shape)
print(external_df.head())
