from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np

from src.config import RAW_DIR

def build_external_features(external_features):
    external_df = None
    for idx, dataset in external_features.items():
        dataset=dataset.copy()
        dataset.columns = ['external_date', f'feature_{idx}']

        dt_format=None
        if idx == 'oni':
            dt_format = '%d/%m/%Y' # Special format

        dataset['external_date'] = pd.to_datetime(dataset['external_date'], format=dt_format).dt.strftime("%Y-%m")
        dataset[f'feature_{idx}'].replace(-9999, np.nan, inplace=True)

        dataset[f'feature_{idx}'] = dataset[f'feature_{idx}'].ffill()

        for lag in [1, 2, 3, 6]:
            dataset[f'{idx}_lag_{lag}'] = dataset[f'feature_{idx}'].shift(lag)
        for window in [3, 6, 12]:
            dataset[f'{idx}_rolling_mean_{window}'] = dataset[f'feature_{idx}'].rolling(window, min_periods=1).mean()

        if external_df is None:
            external_df = dataset
        else:
            external_df = external_df.merge(
                dataset,
                on='external_date',
                how='right'
            )

    return external_df

def get_external_features_for_date(
    external_df: pd.DataFrame,
    date: str
) -> dict:
    target_ym = (
        pd.to_datetime(date)
        - pd.offsets.MonthBegin(1)
    ).strftime("%Y-%m")

    if target_ym in external_df["external_date"].values:
        row = external_df.loc[external_df["external_date"] == target_ym]
        used_ym = target_ym
    else:
        # fallback to last available month
        last_ym = external_df["external_date"].max()
        row = external_df.loc[external_df["external_date"] == last_ym]
        used_ym = last_ym

    if row.empty:
        raise ValueError("External feature table is empty.")

    features = (
        row
        .drop(columns=["external_date"])
        .iloc[0]
        .to_dict()
    )

    features["_external_month_used"] = used_ym
    return features


if __name__ == '__main__':
    dmi = pd.read_csv(RAW_DIR/'Data Eksternal/Dipole Mode Index (DMI).csv')
    aqi = pd.read_csv(RAW_DIR/'Data Eksternal/AirQualityIndex_Google Trends.csv')
    oni = pd.read_csv(RAW_DIR/'Data Eksternal/OceanicNinoIndex (ONI).csv')
    rh = pd.read_csv(RAW_DIR/'Data Eksternal/RelativeHumidityMonthlyMean.csv')

    external_features = build_external_features({
        "dmi": dmi,
        "aqi": aqi,
        "oni": oni,
        "rh": rh,
    })

    print(get_external_features_for_date(
        external_features,
        '2025-01-01'
    ))