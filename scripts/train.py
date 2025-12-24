from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CLEAN_DIR, MODEL_DIR, INFERENCE_DIR, RAIN_EXTREME_COLUMNS, METEOROGICAL_COLUMNS
from src.model import build_feature_pipeline, build_preprocessor, build_model, build_pipeline

print("Loading data...")
train = pd.read_csv(CLEAN_DIR / "train.csv")

train.sort_values(["date", "location"], inplace=True)

X = train.drop(columns=["daily_rainfall_total_mm"])
y = train["daily_rainfall_total_mm"].fillna(0)

pipe = build_pipeline()

tscv = TimeSeriesSplit(n_splits=5)

print("Cross validating...")
scores = cross_val_score(
    pipe,
    X,
    y,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

print("MSE per fold:", -scores)
print("Mean MSE:", -scores.mean())

print("Fitting final model...")
pipe.fit(X, y)

joblib.dump(pipe, MODEL_DIR/'xgb_model.pkl')