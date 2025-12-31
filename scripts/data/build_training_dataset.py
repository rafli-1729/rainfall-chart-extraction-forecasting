import pandas as pd
import numpy as np

from src.config import config
from src.dataset_builder import convert_numeric

rain_col = config.features.rain_extreme_columns
met_col = config.features.meteorogical_columns

def main():
    train = pd.read_csv(config.paths.processed/'train.csv')
    test = pd.read_csv(config.paths.processed/'test.csv')

    target = pd.read_csv(config.paths.processed/'train_target.csv')
    external_features = pd.read_csv(config.paths.processed/'external_features.csv')

    train = pd.merge(train, target, on=['location', 'date'], how='outer')
    # train = pd.merge(train, external_features, on='date', how='left')

    train = convert_numeric(train, columns=rain_col+met_col)
    train.drop(columns='csv_files', inplace=True)
    train.to_csv(config.paths.clean/"train.csv", index=False)

    test = pd.merge(test, external_features, on='date', how='left')
    test = convert_numeric(test, columns=rain_col+met_col)
    test.drop(columns='csv_files', inplace=True)
    test.to_csv(config.paths.clean/'test.csv', index=False)

if __name__ == "__main__":
    main()