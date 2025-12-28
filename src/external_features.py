import pandas as pd

def get_external_features_for_date(
    external_df: pd.DataFrame,
    date: str
) -> dict:
    target_ym = (
        pd.to_datetime(date)
        - pd.offsets.MonthBegin(1)
    ).strftime("%Y-%m")

    if target_ym in external_df["external_date"].values:
        row = external_df.loc[external_df["external_date"] == target_ym]
        used_ym = target_ym
    else:
        last_ym = external_df["external_date"].max()
        row = external_df.loc[external_df["external_date"] == last_ym]
        used_ym = last_ym

    if row.empty:
        raise ValueError("External feature table is empty.")

    features = (
        row
        .drop(columns=["external_date"])
        .iloc[0]
        .to_dict()
    )

    features["_external_month_used"] = used_ym
    return features