import pandas as pd
from pathlib import Path
import joblib

from src.config import CLEAN_DIR, MODEL_DIR

train = pd.read_csv(CLEAN_DIR/'train_1226.csv')
nea = pd.read_csv(CLEAN_DIR/'nea.csv')

model = joblib.load(MODEL_DIR/'xgb_model_1226.pkl')
train['predicted_mm'] = model.predict(train.drop(columns=['daily_rainfall_total_mm']))

train = train[['date', 'location', 'daily_rainfall_total_mm', "predicted_mm"]]

train["date"] = pd.to_datetime(train["date"])
nea["date"]   = pd.to_datetime(nea["date"])

train = train.rename(columns={
    "daily_rainfall_total_mm": "extracted_mm"
})

daily = (
    train[["date", "location", "predicted_mm", "extracted_mm"]]
    .merge(
        nea[["date", "location", "observed_mm"]],
        on=["date", "location"],
        how="left"
    )
)

daily.dropna(subset=['observed_mm']).to_csv(CLEAN_DIR/'daily_1226-v1.csv', index=False)