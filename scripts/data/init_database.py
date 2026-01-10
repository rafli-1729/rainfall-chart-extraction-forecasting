import duckdb
import pandas as pd
from pathlib import Path

from src.config import config

def load_raw_data():
    df = pd.read_csv(config.paths.clean / "daily_v2.csv")
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

con = duckdb.connect(config.paths.database/'rainfall.duckdb')

con.execute("DROP TABLE IF EXISTS daily_rainfall_v2;")
con.execute("CREATE TABLE daily_rainfall_v2 AS SELECT * FROM df")

con.execute("DROP TABLE IF EXISTS weekly_rainfall_v2;")
con.execute("CREATE TABLE weekly_rainfall_v2 AS SELECT * FROM weekly")

con.close()