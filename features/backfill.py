# features/backfill.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
import hopsworks  # Use hopsworks==4.4.2

# --- Load environment variables ---
load_dotenv()

OW_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")

CITY = "Karachi"
LAT, LON = 24.8607, 67.0011


# --- Fetch current data ---
def fetch_current_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OW_KEY}&units=metric"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("⚠️ Weather fetch error:", e)
        return {}


def fetch_current_aqi():
    try:
        url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception as e:
        print("⚠️ AQI fetch error:", e)
        return {}


# --- Combine into DataFrame ---
def fetch_real_data():
    print("🌍 Fetching real-time AQI + weather data...")
    weather = fetch_current_weather()
    aqi = fetch_current_aqi()

    if not weather or not aqi:
        print("⚠️ Missing data, skipping update.")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    main = weather.get("main", {})
    wind = weather.get("wind", {})
    clouds = weather.get("clouds", {})
    iaqi = aqi.get("iaqi", {})

    row = {
        "timestamp_utc": now,
        "ow_temp": main.get("temp"),
        "ow_pressure": main.get("pressure"),
        "ow_humidity": main.get("humidity"),
        "ow_wind_speed": wind.get("speed"),
        "ow_wind_deg": wind.get("deg"),
        "ow_clouds": clouds.get("all"),
        "ow_co": iaqi.get("co", {}).get("v"),
        "ow_no2": iaqi.get("no2", {}).get("v"),
        "ow_pm2_5": iaqi.get("pm25", {}).get("v"),
        "ow_pm10": iaqi.get("pm10", {}).get("v"),
        "aqi_aqicn": aqi.get("aqi"),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),
    }

    df = pd.DataFrame([row])
    for col in df.columns:
        if col != "timestamp_utc":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("✅ Real-time data fetched!")
    return df


# --- Full backfill and historic update ---
def backfill():
    if not HOPSWORKS_API_KEY:
        print("❌ Missing HOPSWORKS_API_KEY. Cannot upload.")
        return

    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # --- Load existing feature group if it exists ---
    try:
        fg = fs.get_feature_group(name="aqi_features", version=1)
        df_existing = fg.read()
        print(f"ℹ️ Loaded {len(df_existing)} existing rows from Hopsworks.")
    except:
        df_existing = pd.DataFrame()
        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            description="AQI + weather dataset for Karachi",
            primary_key=["timestamp_utc"],
            event_time="timestamp_utc",
        )
        print("ℹ️ Feature group created as it did not exist.")

    # --- Fetch new data ---
    df_new = fetch_real_data()

    # --- Combine old + new ---
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    if df_combined.empty:
        print("⚠️ No data to upload.")
        return

    # --- Convert timestamp and sort ---
    df_combined["timestamp_utc"] = pd.to_datetime(df_combined["timestamp_utc"], utc=True, errors="coerce")
    df_combined = df_combined.sort_values("timestamp_utc").reset_index(drop=True)

    # --- Forward-fill missing numeric values ---
    numeric_cols = [
        "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed",
        "ow_wind_deg", "ow_clouds", "ow_co", "ow_no2",
        "ow_pm2_5", "ow_pm10", "aqi_aqicn", "hour",
        "day", "month", "weekday"
    ]
    df_combined[numeric_cols] = df_combined[numeric_cols].ffill().fillna(0)

    # --- Ensure types ---
    int_cols = ["ow_pressure", "ow_humidity", "ow_wind_deg", "ow_clouds",
                "ow_pm2_5", "ow_pm10", "aqi_aqicn",
                "hour", "day", "month", "weekday"]
    float_cols = ["ow_temp", "ow_wind_speed", "ow_no2", "ow_co"]

    for col in int_cols:
        df_combined[col] = df_combined[col].astype("int64")
    for col in float_cols:
        df_combined[col] = df_combined[col].astype(float)

    # --- Deduplicate based on timestamp ---
    df_combined = df_combined.drop_duplicates(subset=["timestamp_utc"], keep="last").reset_index(drop=True)

    print("🧹 Cleaned & combined DataFrame info:")
    print(df_combined.info())

    # --- Upload entire cleaned dataset ---
    print(f"📤 Uploading {len(df_combined)} rows to Hopsworks...")
    fg.insert(df_combined, write_options={"wait_for_job": True})
    print("🚀 Backfill completed successfully! All historical & new data updated.")


# --- Entry Point ---
if __name__ == "__main__":
    print(f"🕐 Running backfill at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ...")
    if not OW_KEY:
        print("❌ Missing OpenWeather key!")
    elif not AQICN_TOKEN:
        print("❌ Missing AQICN token!")
    else:
        print(f"Using API keys: {OW_KEY[:6]}..., {AQICN_TOKEN[:6]}...")
        backfill()
