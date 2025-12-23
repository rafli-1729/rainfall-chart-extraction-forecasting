# Ignore warnings
from warnings import filterwarnings
filterwarnings('ignore')

# Core library
import pandas as pd
import numpy as np
import random
import re

# Pathing library
import os
import glob
from pathlib import Path

# Defaults
filterwarnings('ignore')
np.set_printoptions(legacy='1.25')

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)

RAW_PATH = Path(PROJECT_ROOT/'data/raw')

PROCESS_PATH = Path(PROJECT_ROOT/'data/process')
MERGE_PATH = PROCESS_PATH/'merge'


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

    print(
        f"Random train sample loaded\n"
        f"Location : {location}\n"
        f"Year     : {year}\n"
        f"File     : {sample_path.name}\n"
        f"Shape    : {df.shape}\n"
        f"Info     :",
    )

    df.info()
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
            print("[WARN] No city folders found.")
        return

    max_city_len = max(len(c) for c in city_folders)

    total_files = 0
    total_rows = 0
    processed_cities = 0

    if verbose:
        print(f"[INFO] Processing {len(city_folders)} city folders...\n")

    for city in city_folders:
        city_path = os.path.join(input_root, city)
        csv_files = sorted(glob.glob(os.path.join(city_path, "*.csv")))

        if not csv_files:
            if verbose:
                print(f"[WARN] {city:<{max_city_len}} | no CSV files found")
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
            print(
                f"[OK] {city:<{max_city_len}} | "
                f"{files_count:>4} files | "
                f"{rows_count:>9,} rows | "
                f"{merged_df.shape[1]:>2} cols"
            )

    if verbose:
        print("\n" + "-" * (max_city_len + 45))
        print(
            f"[SUMMARY] Cities processed : {processed_cities}\n"
            f"[SUMMARY] Total files      : {total_files:,}\n"
            f"[SUMMARY] Total rows       : {total_rows:,}"
        )
        print("-" * (max_city_len + 45))


def merge_all_cities(input_root: str, output_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(input_root, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found")

    dfs = []
    for f in csv_files:
        df = clean_column_names(pd.read_csv(f))
        city = os.path.splitext(os.path.basename(f))[0]

        df['location'] = city
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['daily_rainfall_total_mm'] = np.nan

    merged_df.to_csv(output_dir, index=False)

    return merged_df


if __name__ == '__main__':
    train_sample = load_random_train_sample(
        RAW_PATH / "train",
    )

    # print(train_sample.iloc[1,1])

    train_columns = check_columns_consistency(RAW_PATH / "train")
    print(f"\nFound {len(train_columns)} unique column structures in raw train\n")

    for i, (cols, files) in enumerate(train_columns.items(), 1):
        print(f"[Structure {i}]")
        print(f"Columns ({len(cols)}): {cols}")
        print(f"Used by {len(files)} files")
        print("-" * 90, end='\n\n')

    test_columns = check_columns_consistency(RAW_PATH/'test')
    print(f"Found {len(test_columns)} unique column structures in raw test\n")

    for i, (cols, files) in enumerate(test_columns.items(), 1):
        print(f"[Structure {i}]")
        print(f"Columns ({len(cols)}): {cols}")
        print(f"Used by {len(files)} files")
        print("-" * 90)

    print('Merge yearly data . . .')
    merge_each_city(
        input_root=RAW_PATH/'train',
        output_dir=MERGE_PATH/'train'
    )

    merge_each_city(
        input_root=RAW_PATH/'test',
        output_dir=MERGE_PATH/'test'
    )

    print('Merge all cities to single data . . .')
    merge_all_cities(
        input_root=MERGE_PATH/'train',
        output_dir=PROCESS_PATH/'train.csv'
    )

    merge_all_cities(
        input_root=MERGE_PATH/'test',
        output_dir=PROCESS_PATH/'test.csv'
    )

    print('Success!')