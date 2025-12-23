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

from src.config import CLEAN_DIR, MODEL_DIR, INFERENCE_DIR, RAIN_EXTREME_COLUMNS, METEOROGICAL_COLUMNS


print('Loading Train and Test data . . .')
train = pd.read_csv(CLEAN_DIR/'train.csv')
test = pd.read_csv(CLEAN_DIR/'test.csv')


print('Building pipeline . . .')
preprocess = ColumnTransformer(
    transformers=[
        ('rain_zero', SimpleImputer(strategy='constant', keep_empty_features=True, fill_value=0), RAIN_EXTREME_COLUMNS),
        ('num', SimpleImputer(strategy='median'), METEOROGICAL_COLUMNS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
    ]
)

pipe = Pipeline(
    steps=[
        ("time", TimeFeatures()),
        ("temp", TemperatureFeatures()),
        ("wind_rain", WindRainFeatures()),
        ("lag", LagFeatures(lag_days=1)),
        ("rolling", RollingStatsFeatures()),
        ("cyclical", CyclicalInteractionFeatures()),
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

pipe.fit(X, y)

import joblib

joblib.dump(pipe, MODEL_DIR/'xgb_model.pkl')
print('XGB Pipeline succesfully saved!')