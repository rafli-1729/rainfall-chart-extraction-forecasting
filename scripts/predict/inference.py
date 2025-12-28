import pandas as pd

from src.config import config
from src.model import

test = pd.read_csv(config.paths.clean / "test.csv")

test['prediksi'] = pipe.predict(test.drop(columns=['daily_rainfall_total_mm']))
test['date'] = pd.to_datetime(test['date'])
test['tahun'] = test['date'].dt.year
test['bulan'] = test['date'].dt.month
test['hari'] = test['date'].dt.day
test['ID (kota)'] = (
    test['location'].str.lower() + "_" + test['date'].dt.strftime("%Y_%m_%d")
)

test[['ID (kota)',
      'tahun', 'bulan', 'hari',
      'prediksi'
]].to_csv(config.paths.inference/'submission.csv', index=False)