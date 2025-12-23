import pandas as pd
import numpy as np


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