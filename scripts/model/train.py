from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    CLEAN_DIR,
    INFERENCE_DIR,
)
from src.model import (
    build_feature_pipeline,
    build_preprocessor,
    build_model,
    build_pipeline
)

print("Loading data...")
train = pd.read_csv(CLEAN_DIR / "train_1226.csv")
print(train.shape)
print(len(train.location.unique()))

# train.dropna(inplace=True)
print(train.shape)
print(len(train.location.unique()))

train.sort_values(["date", "location"], inplace=True)

X = train.drop(columns=["daily_rainfall_total_mm"])
y = train["daily_rainfall_total_mm"].fillna(0)

pipe = build_pipeline(model_type='two_stage')

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

y_pred = pipe.predict(X)

print("Mean Absolute Error :", mean_absolute_error(y, y_pred))
print("Mean Squared Error  :", mean_squared_error(y, y_pred))

joblib.dump(pipe, MODEL_DIR/'xgb_model_1226.pkl')
print("Model saved!")