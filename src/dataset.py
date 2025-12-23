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
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[().%°/]", "", regex=True)
    )

    return df


def merge_each_city(
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

        if not csv_files:
            if verbose:
                logger.warning("%-*s | no CSV files found", max_city_len, city)
            continue

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df = clean_column_names(df)
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)

        output_path = os.path.join(output_dir, f"{city}.csv")
        merged_df.to_csv(output_path, index=False)

        files_count = len(csv_files)
        rows_count = merged_df.shape[0]

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


def merge_all_cities(input_root: str, output_dir: str, corrupt_cols) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(input_root, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found")

    dfs = []
    for f in csv_files:
        df = clean_column_names(pd.read_csv(f))
        df = convert_numeric(df, corrupt_cols)
        city = os.path.splitext(os.path.basename(f))[0]

        df['location'] = city
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['daily_rainfall_total_mm'] = np.nan

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
        location = feature_path.stem.replace("_merged", "")
        target_path = targets_dir / f"{location}.csv"

        if not target_path.exists():
            if verbose:
                tqdm.write(f"⚠️ Skipped {location}: target file not found")
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


def merge_dataset(df, external_df, date_col, save_path):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format = 'mixed')
    df['external_date'] = df[date_col].dt.strftime('%Y-%m')

    df['external_date'] = pd.to_datetime(df['external_date'])
    external_df['external_date'] = pd.to_datetime(external_df['external_date'])

    df = pd.merge(df, external_df, on = 'external_date', how = 'left')

    df = df.drop(columns=['external_date'])
    df.sort_values([date_col, 'location'])

    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    from config import RAW_DIR

    verbose=True
    train_sample = load_random_train_sample(
        RAW_DIR / "train",
    )

    # print(train_sample.iloc[1,1])

    train_columns = check_columns_consistency(RAW_DIR / "train")
    print(f"\nFound {len(train_columns)} unique column structures in raw train\n")

    for i, (cols, files) in enumerate(train_columns.items(), 1):
        print(f"[Structure {i}]")
        print(f"Columns ({len(cols)}): {cols}")
        print(f"Used by {len(files)} files")
        print("-" * 90, end='\n\n')

    test_columns = check_columns_consistency(RAW_DIR/'test')
    print(f"Found {len(test_columns)} unique column structures in raw test\n")

    for i, (cols, files) in enumerate(test_columns.items(), 1):
        print(f"[Structure {i}]")
        print(f"Columns ({len(cols)}): {cols}")
        print(f"Used by {len(files)} files")
        print("-" * 90)