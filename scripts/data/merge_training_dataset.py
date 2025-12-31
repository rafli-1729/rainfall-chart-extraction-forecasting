import os
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from src.config import config
from src.dataset_builder import(
    concatenate_csv_files,
    merge_dataframes
)

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

    # ================================ TRAIN ================================

    for city in train_city_folders:
        city_path = os.path.join(train_input_root, city)
        df = concatenate_csv_files(id="csv_files", input_root=city_path)
        df.to_csv(f"{train_output_dir}/{city}.csv", index=False)

    train_output_file = config.paths.processed/'train.csv'
    train_df = concatenate_csv_files(id="location",
                                     input_root=train_output_dir,
                                     output_dir=train_output_file)

    # ================================ TEST ================================

    for city in test_city_folders:
        city_path = os.path.join(test_input_root, city)
        df = concatenate_csv_files(id="csv_files", input_root=city_path)
        df.to_csv(f"{test_output_dir}/{city}.csv", index=False)

    test_output_file = config.paths.processed/'test.csv'
    test_df = concatenate_csv_files(id="location",
                                    input_root=test_output_dir)

    test_df['total_daily_rainfall_mm'] = np.nan
    test_df.to_csv(test_output_file, index=False)

    # ================================ Train Target ================================

    train_target_files = config.paths.interim/'extract'
    train_target_dir = config.paths.processed/'train_target.csv'

    train_target_df = concatenate_csv_files(id="location",
                                            input_root=train_target_files,
                                            output_dir=train_target_dir)

    # ================================ SUMMARY ================================

    logger.info(f"Train features shape    : {train_df.shape}")
    logger.info(f"Train target shape      : {train_target_df.shape}")
    logger.info(f"Test shape              : {test_df.shape}")

if __name__ == '__main__':
    main()