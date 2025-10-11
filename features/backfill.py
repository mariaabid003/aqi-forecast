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


# --- Push to Hopsworks ---
def backfill():
    df_new = fetch_real_data()
    if df_new.empty:
        print("⚠️ No new data fetched.")
        return

    if not HOPSWORKS_API_KEY:
        print("❌ Missing HOPSWORKS_API_KEY. Cannot upload.")
        return

    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # --- Data cleaning ---
    df_new["timestamp_utc"] = pd.to_datetime(df_new["timestamp_utc"], utc=True, errors="coerce")

    # Columns expected as integers (Hopsworks schema uses BIGINT)
    int_columns = [
        "ow_pressure", "ow_humidity", "ow_wind_deg", "ow_clouds",
        "ow_pm2_5", "ow_pm10", "aqi_aqicn",
        "hour", "day", "month", "weekday"
    ]
    for col in int_columns:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna(0).astype("int64")

    # Columns expected as floats
    float_columns = ["ow_temp", "ow_wind_speed", "ow_no2", "ow_co"]
    for col in float_columns:
        if col in df_new.columns:
            df_new[col] = df_new[col].fillna(0).astype(float)

    # Fill remaining NaNs if any
    df_new = df_new.fillna({
        col: 0 for col in df_new.select_dtypes(include=["number"]).columns
    })

    print("🧹 Cleaned DataFrame before upload:")
    print(df_new.info())

    # --- Upload to Feature Store ---
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI + weather dataset for Karachi",
        primary_key=["timestamp_utc"],
        event_time="timestamp_utc",
    )

    print(f"📤 Uploading {len(df_new)} new row(s) to Hopsworks...")
    fg.insert(df_new, write_options={"wait_for_job": True})
    print("🚀 New data pushed successfully!")


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
