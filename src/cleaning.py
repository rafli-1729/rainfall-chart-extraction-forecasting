import pandas as pd
import numpy as np
import glob
import re
import os


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names:
    - strip whitespace
    - lowercase
    - replace spaces with underscores
    - remove special characters
    """
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
    verbose: bool = True
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