import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")
if not TOKEN:
    raise ValueError("AQICN_TOKEN not found in environment (.env)")

BASE_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"

def fetch_aqicn(lat: float, lon: float) -> dict:
    """Fetch AQICN data for given lat/lon."""
    url = BASE_URL.format(lat=lat, lon=lon, token=TOKEN)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def parse_response(j: dict) -> dict:
    """Parse AQICN response into flat dict."""
    data = j.get("data", {})
    iaqi = data.get("iaqi", {})

    row = {
        "timestamp_utc": data.get("time", {}).get("utc"),
        "aqi_aqicn": data.get("aqi"),
    }
    for k, v in iaqi.items():
        row[f"aqicn_{k}"] = v.get("v")
    return row

if __name__ == "__main__":
    lat, lon = 24.8607, 67.0011

    raw = fetch_aqicn(lat, lon)
    row = parse_response(raw)
    df = pd.DataFrame([row])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = "data/raw_aqicn"
    os.makedirs(out_dir, exist_ok=True)

    ts_file = os.path.join(out_dir, f"aqicn_{ts}.parquet")
    latest_file = os.path.join(out_dir, "latest_aqicn.parquet")

    df.to_parquet(ts_file, index=False)
    df.to_parquet(latest_file, index=False)

    logging.info(f"Saved {ts_file} and updated {latest_file}")
    print(df.head())
