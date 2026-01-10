import pandas as pd

from src.config import config
import joblib

if __name__ == "__main__":
    model = joblib.load(config.paths.models/"cv_model_v2.pkl")
    test = pd.read_csv(config.paths.clean / "test.csv")

    test['prediksi'] = model.predict(test)
    test['prediksi'] = test['prediksi'].clip(lower=0)

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
    ]].to_csv(config.paths.inference/'submission_v2.csv', index=False)