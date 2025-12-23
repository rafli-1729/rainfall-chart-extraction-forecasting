import logging
import pandas as pd
from pathlib import Path

from src.dataset import(
    merge_each_city,
    merge_all_cities,
    merge_dataset,
    build_training_dataset
)
from src.external import build_external_features

from src.config import (
    RAW_DIR,
    PROCESS_DIR,
    CLEAN_DIR,
    METEOROGICAL_COLUMNS,
    RAIN_EXTREME_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    verbose = True

    logger.info("Merging yearly data (train)...")
    merge_each_city(
        input_root=RAW_DIR/'train',
        output_dir=PROCESS_DIR/'merge'/'train',
        verbose=verbose
    )

    logger.info("Merging yearly data (test)...")
    merge_each_city(
        input_root=RAW_DIR/'test',
        output_dir=PROCESS_DIR/'merge'/'test',
        verbose=verbose
    )

    train = build_training_dataset(
        features_dir=PROCESS_DIR/'merge'/'train',
        targets_dir=PROCESS_DIR/'extract',
        output_csv=PROCESS_DIR/'train.csv',
        corrupt_col=METEOROGICAL_COLUMNS+RAIN_EXTREME_COLUMNS,
        verbose=True
    )

    logger.info("Merging all cities to single dataset (test)...")
    test = merge_all_cities(
        input_root=PROCESS_DIR/'merge'/'test',
        output_dir=PROCESS_DIR/'test.csv',
        corrupt_cols=METEOROGICAL_COLUMNS+RAIN_EXTREME_COLUMNS
    )

    dmi = pd.read_csv(RAW_DIR/'Data Eksternal/Dipole Mode Index (DMI).csv')
    aqi = pd.read_csv(RAW_DIR/'Data Eksternal/AirQualityIndex_Google Trends.csv')
    oni = pd.read_csv(RAW_DIR/'Data Eksternal/OceanicNinoIndex (ONI).csv')
    rh = pd.read_csv(RAW_DIR/'Data Eksternal/RelativeHumidityMonthlyMean.csv')

    external_features = build_external_features({
        "dmi": dmi,
        "aqi": aqi,
        "oni": oni,
        "rh": rh,
    })

    merge_dataset(
        train, external_features, 'date',
        save_path=CLEAN_DIR/'train.csv'
    )
    merge_dataset(
        test, external_features, 'date',
        save_path=CLEAN_DIR/'test.csv'
    )

if __name__ == '__main__':
    main()