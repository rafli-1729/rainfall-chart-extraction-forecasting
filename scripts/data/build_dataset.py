import logging
import pandas as pd
from pathlib import Path

from src.dataset_builder import(
    concatenate_csv_files,
    merge_all_cities,
    merge_dataframes,
    make_time_key,
    build_training_dataset,
    merge_csv
)
from src.external_features import build_external_features

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    verbose = True

    test_input_root = config.paths.raw/'Test'
    train_input_root = config.paths.raw/'Train'
    test_output_dir = config.paths.interim/'merged_by_cities/test'
    train_output_dir = config.paths.interim/'merged_by_cities/train'

    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)

    train_city_folders = [f.name for f in os.scandir(train_input_root) if f.is_dir()]
    test_city_folders = [f.name for f in os.scandir(test_input_root) if f.is_dir()]

    for city in train_city_folders:
        city_path = os.path.join(test_input_root, city)
        df = concatenate_csv_files(id=1, input_root=city_path)
        df.to_csv(f"{train_output_dir}/{city}.csv", index=False)

    train_output_file = config.paths.interim/'merged/train.csv'
    concatenate_csv_files(id=2,
                          input_root=train_output_dir,
                          output_dir=train_output_file)

    for city in test_city_folders:
        city_path = os.path.join(test_input_root, city)
        df = concatenate_csv_files(id=1, input_root=city_path)
        df.to_csv(f"{test_output_dir}/{city}.csv", index=False)

    test_output_file = config.paths.interim/'merged/test.csv'
    concatenate_csv_files(id=2,
                          input_root=test_output_dir,
                          output_dir=test_output_file)

    

    # logger.info("Merging yearly data (train)...")
    # concatenate_csv_files(
    #     input_root=config.paths.raw/'train',
    #     output_dir=config.paths.interim/'merge'/'train',
    #     verbose=verbose
    # )

    # logger.info("Merging yearly data (test)...")
    # concatenate_csv_files(
    #     input_root=config.paths.raw/'test',
    #     output_dir=config.paths.interim/'merge'/'test',
    #     verbose=verbose
    # )

    # train = build_training_dataset(
    #     features_dir=config.paths.interim/'merge'/'train',
    #     targets_dir=config.paths.interim/'extract',
    #     output_csv=config.paths.processed/'train.csv',
    #     corrupt_col=config.features.meteorogical_columns+
    #                 config.features.rain_extreme_columns,
    #     verbose=True
    # )

    # logger.info("Merging all cities to single dataset (test)...")
    # test = concatenate_csv_files(
    #     input_root=config.paths.interim/'merge'/'test',
    #     output_dir=config.paths.processed/'test.csv'
    # )

    # dmi = pd.read_csv(config.paths.raw/'Data Eksternal/Dipole Mode Index (DMI).csv')
    # aqi = pd.read_csv(config.paths.raw/'Data Eksternal/AirQualityIndex_Google Trends.csv')
    # oni = pd.read_csv(config.paths.raw/'Data Eksternal/OceanicNinoIndex (ONI).csv')
    # rh = pd.read_csv(config.paths.raw/'Data Eksternal/RelativeHumidityMonthlyMean.csv')

    # external_datasets = {
    #     "dmi": dmi,
    #     "aqi": aqi,
    #     "oni": oni,
    #     "rh": rh,
    # }

    # external_features = build_external_features(external_datasets)

    # params={
    #     "col":"date",
    #     "fmt":"%Y-%m",
    #     "out_col":"__time_key"
    # }

    # train = make_time_key(df=train, **params)
    # test = make_time_key(df=test, **params)

    # external_features = make_time_key(df=external_features,
    #                                   col='external_date',
    #                                   fmt='%Y-%m', out_col="__time_key")

    # merge_dataframes(
    #     left=train, right=external_features, on="__time_key",
    #     save_path=config.paths.clean/'train.csv', drop=True
    # )
    # merge_dataframes(
    #     left=test, right=external_features, on="__time_key",
    #     save_path=config.paths.clean/'test.csv', drop=True
    # )

if __name__ == '__main__':
    main()