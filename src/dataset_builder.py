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
    input_root: str,
    output_dir: str,
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    try:
        city_folders = [f.name for f in os.scandir(input_root) if f.is_dir()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    if not city_folders:
        if verbose:
            logger.warning("No city folders found.")
        return

    max_city_len = max(len(c) for c in city_folders)

    total_files = 0
    total_rows = 0
    processed_cities = 0

    if verbose:
        logger.info("Processing %d city folders...\n", len(city_folders))

    for city in city_folders:
        city_path = os.path.join(input_root, city)
        csv_files = sorted(glob.glob(os.path.join(city_path, "*.csv")))

        concatenated_df = concatenate_csv_files(csv_path, id=1)

        output_path = os.path.join(output_dir, f"{city}.csv")
        concatenated_df.to_csv(output_path, index=False)

        files_count = len(csv_files)
        rows_count = concatenated_df.shape[0]

        total_files += files_count
        total_rows += rows_count
        processed_cities += 1

        if verbose:
            logger.info(
                "%-*s | %4d files | %9d rows | %2d cols",
                max_city_len,
                city,
                files_count,
                rows_count,
                merged_df.shape[1],
            )

    if verbose:
        sep = "-" * (max_city_len + 45)
        logger.info(
            "\n%s\n"
            "[SUMMARY] Cities processed : %d\n"
            "[SUMMARY] Total files      : %d\n"
            "[SUMMARY] Total rows       : %d\n"
            "%s",
            sep,
            processed_cities,
            total_files,
            total_rows,
            sep,
        )

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
        city = os.path.splitext(os.path.basename(f))[0]

        df[f'source_{id}'] = city
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    if output_dir:
        merged_df.to_csv(output_dir, index=False)

    return merged_df


def build_training_dataset(
    features_dir: str | Path,
    targets_dir: str | Path,
    output_csv: str | Path,
    corrupt_col: list,
    verbose: bool = True
) -> pd.DataFrame:
    feature_files = sorted(features_dir.glob("*.csv"))

    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")

    merged_frames = []
    for feature_path in feature_files:
        location = feature_path.stem
        target_path = targets_dir / f"{location}.csv"

        if not target_path.exists():
            if verbose:
                logger.warning(f"⚠️ Skipped {location}: target file not found")
            continue

        df_feat = pd.read_csv(feature_path, parse_dates=["date"])
        df_tgt = pd.read_csv(target_path, parse_dates=["date"])

        df_merged = pd.merge(
            df_feat.sort_values("date"),
            df_tgt.sort_values("date"),
            on="date",
            how="inner",
            validate="one_to_one"
        )

        df_merged["location"] = location
        merged_frames.append(df_merged)

    if not merged_frames:
        raise RuntimeError("No datasets were merged successfully")

    final_df = pd.concat(merged_frames, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df = convert_numeric(final_df, corrupt_col)
    final_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n✅ Final training dataset saved to: {output_csv}")
        print(f"   Rows: {len(final_df):,}")
        print(f"   Columns: {final_df.shape[1]}")

    return final_df


def make_time_key(
    df: pd.DataFrame,
    col: str,
    fmt: str,
    out_col: str = "__time_key"
):
    df = df.copy()
    df[out_col] = pd.to_datetime(df[col], format='mixed').dt.strftime(fmt)
    return df


def infer_time_resolution(series: pd.Series) -> str:
    s = pd.to_datetime(series, errors="coerce").dropna()

    if s.dt.day.nunique() > 1:
        return "daily"
    if s.dt.month.nunique() > 1:
        return "monthly"
    return "yearly"


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