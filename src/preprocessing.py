import pandas as pd
import numpy as np

from src import cleaning

def convert_numeric(df: pd.DataFrame, columns: list[str]) -> list[str]:
    converted = []
    for col in columns:
        if col == 'date':
            continue

        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)

    return df