import pandas as pd

import os
import glob
import sys

from tqdm import tqdm
import logging

from src.config import config
from src.plot_label_extractor import count_total_rows
from src.dots_extractor_v2.pipeline import extract_rainfall_from_plot
from src.dataset_builder import clean_column_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    verbose = False

    input_root = config.paths.raw / "Train"
    output_dir = config.paths.interim / "extractedv2"
    os.makedirs(output_dir, exist_ok=True)

    total_rows = count_total_rows(input_root)

    with tqdm(
        total=total_rows,
        desc="Extracting rainfall",
        unit="rows",
    ) as pbar:

        for loc in os.scandir(input_root):
            if not loc.is_dir():
                continue

            print(f"\nProcessing {loc.name}...")
            csvs = sorted(glob.glob(f"{loc.path}/*.csv"))
            pngs = sorted(glob.glob(f"{loc.path}/*.png"))

            yearly_data = []

            for csv_file, png_file in zip(csvs, pngs):
                df = clean_column_names(pd.read_csv(csv_file))
                df["date"] = pd.to_datetime(df["date"])
                df = (
                    df
                    .set_index("date")
                    .reindex(
                        pd.date_range(
                            start=df["date"].min(),
                            end=df["date"].max(),
                            freq="D",
                        )
                    )
                    .rename_axis("date")
                    .reset_index()
                )

                rainfall = extract_rainfall_from_plot(
                    png=png_file,
                    csv=csv_file,
                    verbose=verbose,
                )

                if len(rainfall) != len(df):
                    raise ValueError(
                        f"Length mismatch: rainfall={len(rainfall)} "
                        f"df={len(df)} for {png_file}"
                    )

                rainfall.index = pd.to_datetime(rainfall.index)
                rainfall.name = "daily_rainfall_total_mm"

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = df.join(rainfall, how="left")
                df = df.reset_index()

                yearly_data.append(
                    df[["date", "daily_rainfall_total_mm"]]
                )

                pbar.update(len(df))

            if yearly_data:
                final_df = pd.concat(yearly_data, ignore_index=True)
                final_df.to_csv(
                    os.path.join(output_dir, f"{loc.name}.csv"),
                    index=False,
                )


if __name__ == "__main__":
    main()