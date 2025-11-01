import os
import pandas as pd
from datetime import datetime, timezone
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def load_latest_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def build_features():
    ow_file = "data/raw_openweather/latest_openweather.parquet"
    aqicn_file = "data/raw_aqicn/latest_aqicn.parquet"
    out_dir = "data/features"
    os.makedirs(out_dir, exist_ok=True)

    df_ow = load_latest_parquet(ow_file).reset_index(drop=True)
    df_aq = load_latest_parquet(aqicn_file).reset_index(drop=True)

    df_ow = df_ow.rename(columns={"timestamp_utc": "ow_timestamp"})
    df_aq = df_aq.rename(columns={"timestamp_utc": "aqicn_timestamp"})

    if df_aq["aqicn_timestamp"].isnull().any():
        df_aq["aqicn_timestamp"] = datetime.now(timezone.utc).isoformat()

    df = pd.concat([df_ow, df_aq], axis=1)
    df["timestamp_utc"] = pd.to_datetime(df["ow_timestamp"], utc=True)
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["month"] = df["timestamp_utc"].dt.month
    df["weekday"] = df["timestamp_utc"].dt.weekday

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_file = os.path.join(out_dir, f"features_{ts}.parquet")
    latest_file = os.path.join(out_dir, "latest_features.parquet")

    df.to_parquet(ts_file, index=False)
    df.to_parquet(latest_file, index=False)

    logging.info(f"Saved {ts_file} and updated {latest_file}")
    print(df.head())

if __name__ == "__main__":
    build_features()
