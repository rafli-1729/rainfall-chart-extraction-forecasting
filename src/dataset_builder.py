# Ignore warnings
from warnings import filterwarnings
filterwarnings('ignore')

# logging
import logging
logger = logging.getLogger(__name__)

# Core library
import pandas as pd
import numpy as np
import random
import re

# Pathing library
import os
import glob
from pathlib import Path

def convert_numeric(df: pd.DataFrame, columns: list[str]) -> list[str]:
    converted = []
    for col in columns:
        if col == 'date' or col == 'location':
            continue

        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)

    return df


def load_random_train_sample(train_root: Path, seed: int | None = None):
    csv_files = list(train_root.glob("*/*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in train directory")

    if seed is not None:
        random.seed(seed)

    sample_path = random.choice(csv_files)
    df = pd.read_csv(sample_path)

    location = sample_path.parent.name

    match = re.search(r"(\d{4})", sample_path.name)
    year = match.group(1) if match else "unknown"

    logger.info(
        "Random train sample loaded | "
        "Location=%s | Year=%s | File=%s | Shape=%s",
        location, year, sample_path.name, df.shape
    )

    logger.debug("DataFrame info:\n%s", df.info(buf=None))
    return df


def check_columns_consistency(root_dir):
    columns_map = {}

    csv_files = glob.glob(
        os.path.join(root_dir, "**", "*.csv"),
        recursive=True
    )

    if not csv_files:
        raise ValueError("No CSV files found")

    for f in csv_files:
        cols = pd.read_csv(f, nrows=0).columns

        normalized_cols = tuple(sorted(
            col.strip() for col in cols
        ))

        columns_map.setdefault(normalized_cols, []).append(f)

    return columns_map


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.replace("â", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[().%°/]", "", regex=True)
    )

    return df


def concatenate_csv_files(
    id: int | str ,
    input_root: str,
    output_dir: str = None,
) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(input_root, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found")

    dfs = []
    for f in csv_files:
        df = clean_column_names(pd.read_csv(f))
        filename = os.path.splitext(os.path.basename(f))[0]

        df[id] = filename
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    if output_dir:
        merged_df.to_csv(output_dir, index=False)

    return merged_df


def make_time_key(
    df: pd.DataFrame,
    col: str,
    fmt: str,
    out_col: str = "__time_key"
):
    df = df.copy()
    df[out_col] = pd.to_datetime(df[col], format='mixed').dt.strftime(fmt)
    return df


def align_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        d = row["date"]

        # monthly
        start = d.replace(day=1)
        end = start + pd.offsets.MonthEnd(0)

        for day in pd.date_range(start, end, freq="D"):
            r = row.copy()
            r["date"] = day
            records.append(r)

    return pd.DataFrame(records)


def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "left",
    save_path: str | Path = None,
    drop: bool = False
):
    left = left.copy()
    merged = pd.merge(left, right, on=on, how=how)

    if drop:
        merged = merged.drop(columns=[on])

    if save_path:
        merged.to_csv(save_path, index=False)

    return merged


if __name__ == '__main__':
    main3()