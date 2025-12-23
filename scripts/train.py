from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = Path(PROJECT_ROOT/'data/clean')
INFERENCE_PATH = Path(PROJECT_ROOT/'/data/inference/')

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

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

print('Loading Train and Test data . . .')
train = pd.read_csv(CLEAN_PATH/'train_engineered.csv')
test = pd.read_csv(CLEAN_PATH/'test_engineered.csv')

rain_extreme_cols = [
    'highest_30_min_rainfall_mm',
    'highest_60_min_rainfall_mm',
    'highest_120_min_rainfall_mm'
]

num_cols = [
    'mean_temperature_c',
    'maximum_temperature_c',
    'minimum_temperature_c',
    'mean_wind_speed_kmh',
    'max_wind_speed_kmh',
]

cat_cols = ['location']

print('Building pipeline . . .')
preprocess = ColumnTransformer(
    transformers=[
        ('rain_zero', SimpleImputer(strategy='constant', keep_empty_features=True, fill_value=0), rain_extreme_cols),
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

pipe = Pipeline(
    steps=[
        ('preprocess', preprocess),
        ('model', XGBRegressor(
            objective='reg:squarederror',

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
        ))
    ]
)

tscv = TimeSeriesSplit(n_splits=5)

train.sort_values(['date', 'location'], inplace=True)

X = train.drop(columns=['daily_rainfall_total_mm', 'date'])
y = train['daily_rainfall_total_mm']

print('Train XGBoost model . . .')
scores = cross_val_score(
    pipe,
    X,
    y,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

print("MSE per fold:", -scores)
print("Mean MSE:", -scores.mean())