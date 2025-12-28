import duckdb
import pandas as pd
from pathlib import Path

from src.config import CLEAN_DIR, DATABASE_DIR

def load_raw_data():
    df = pd.read_csv(CLEAN_DIR / "daily_1226-v1.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_weekly_aggregates(df):
    return (
        df
        .assign(year=df["date"].dt.year)
        .set_index("date")
        .groupby(["location", "year"])
        .resample("W")
        .mean()
        .reset_index()
    )

df = load_raw_data()
weekly = build_weekly_aggregates(df)

con = duckdb.connect(DATABASE_DIR/'rainfall.duckdb')

con.execute("DROP TABLE IF EXISTS daily_rainfall;")
con.execute("CREATE TABLE weekly_rainfall AS SELECT * FROM df")

con.execute("DROP TABLE IF EXISTS weekly_rainfall;")
con.execute("CREATE TABLE weekly_rainfall AS SELECT * FROM weekly")

con.close()