from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    CLEAN_DIR,
    MODEL_DIR,
    INFERENCE_DIR,
    RAIN_EXTREME_COLUMNS,
    METEOROGICAL_COLUMNS
)
from src.model import (
    build_feature_pipeline,
    build_preprocessor,
    build_model,
    build_pipeline
)

print("Loading data...")
train = pd.read_csv(CLEAN_DIR / "train_1226.csv")

train.sort_values(["date", "location"], inplace=True)

X = train.drop(columns=["daily_rainfall_total_mm"])
y = (train["daily_rainfall_total_mm"].fillna(0)>0.1).astype(int)

pipe = build_pipeline(model_type='classifier')

tscv = TimeSeriesSplit(n_splits=5)

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

param_grid = {
    "model__max_depth": [3, 4, 5],
    "model__min_child_weight": [3, 5, 10],
    "model__scale_pos_weight": [0.5, 0.75, 1.0, 1.25],
}

tscv = TimeSeriesSplit(n_splits=3)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="recall",
    cv=tscv,
    n_jobs=-1,
    verbose=3
)

gs.fit(X, y)
print(gs.best_params_)