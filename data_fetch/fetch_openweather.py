#!/usr/bin/env python3
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_retry_session(retries=3, backoff=1, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def fetch_current(lat, lon, api_key, session=None, timeout=10):
    session = session or get_retry_session()
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def parse_response(j):
    if "list" not in j or not j["list"]:
        raise ValueError("OpenWeather response missing 'list'")
    rec = j["list"][0]
    ts = rec.get("dt")
    ts = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    main = rec.get("main", {})
    comps = rec.get("components", {})
    out = {
        "timestamp_utc": ts.isoformat(),
        "aqi_ow": main.get("aqi")
    }
    for k, v in comps.items():
        out[f"ow_{k}"] = v
    return out

def save(df: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = out_dir / f"openweather_{ts}.parquet"
    df.to_parquet(file_path, index=False)
    df.to_parquet(out_dir / "latest_openweather.parquet", index=False)
    logging.info("Saved %s and updated latest_openweather.parquet", file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=24.8607, help="Latitude (default: Karachi)")
    parser.add_argument("--lon", type=float, default=67.0011, help="Longitude (default: Karachi)")
    parser.add_argument("--out_dir", default="data/raw_openweather", help="Output directory")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise SystemExit("OPENWEATHER_API_KEY not found in environment (.env)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    session = get_retry_session()

    try:
        j = fetch_current(args.lat, args.lon, api_key, session=session)
        row = parse_response(j)
        df = pd.DataFrame([row])
        save(df, args.out_dir)
        print(df.to_string(index=False))
    except Exception as e:
        logging.exception("Failed to fetch OpenWeather: %s", e)
        raise

if __name__ == "__main__":
    main()
