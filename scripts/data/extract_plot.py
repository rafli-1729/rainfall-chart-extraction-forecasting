import pandas as pd

import os
import glob

from tqdm import tqdm
import logging

from src.config import config
from src.plot_extraction import (
    count_total_rows,
    extract_rainfall_from_plot
)
from src.dataset_builder import clean_column_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    verbose = True

    input_root=config.paths.raw/"Train"
    output_dir=config.paths.interim/"extracted"

    os.makedirs(output_dir, exist_ok=True)

    total_rows = count_total_rows(input_root)

    with tqdm(
        total=total_rows, desc="Extracting rainfall", unit="rows"
    ) as pbar:

        for loc in os.scandir(input_root):
            if not loc.is_dir():
                continue

            csvs = sorted(glob.glob(f"{loc.path}/*.csv"))
            pngs = sorted(glob.glob(f"{loc.path}/*.png"))

            yearly_data = []

            for csv_file, png_file in zip(csvs, pngs):
                df = clean_column_names(pd.read_csv(csv_file))
                df["date"] = pd.to_datetime(df["date"])

                rainfall = extract_rainfall_from_plot(
                    png_file, len(df), verbose
                )

                df["daily_rainfall_total_mm"] = rainfall
                yearly_data.append(df[["date", "daily_rainfall_total_mm"]])

                pbar.update(len(df))

            if yearly_data:
                final_df = pd.concat(yearly_data, ignore_index=True)
                final_df.to_csv(
                    os.path.join(output_dir, f"{loc.name}.csv"),
                    index=False
                )

            break

if __name__ == "__main__":
    main()