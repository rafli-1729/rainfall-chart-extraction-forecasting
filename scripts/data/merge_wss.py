from warnings import filterwarnings
filterwarnings('ignore')

import os

from src.config import config
from src.dataset_builder import (
    concatenate_csv_files,
    convert_numeric
)

wss_path = config.paths.raw/'wss'
wss_merged_path = config.paths.interim/'wss_merged'

rain_col = config.features.rain_extreme_columns
met_col = config.features.meteorogical_columns


def main():
    os.makedirs(wss_merged_path, exist_ok=True)

    # for folder in wss_path.iterdir():
    #     filename = os.path.splitext(os.path.basename(folder))[0]
    #     concatenate_csv_files(id='csv_files',
    #                           input_root=folder,
    #                           output_dir=f"{wss_merged_path}/{filename}.csv")

    wss_df = concatenate_csv_files(id='station_code',
                                      input_root=wss_merged_path)
    wss_df = convert_numeric(wss_df, columns=rain_col+met_col+['daily_rainfall_total_mm'])

    wss_df.drop(columns=['csv_files', 'station_code'], inplace=True)
    wss_df.sort_values(['location', 'date'], inplace=True)
    wss_df["location"] = (
        wss_df["location"]
        .str.replace(r"[()]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    wss_df.to_csv(config.paths.clean/'wss.csv', index=False)


if __name__ == "__main__":
    main()
