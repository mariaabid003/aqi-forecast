import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
import hopsworks

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

LAT, LON = 24.8607, 67.0011

# Fetch current weather
def fetch_current_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}

# Fetch current AQI
def fetch_current_aqi():
    try:
        url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except:
        return {}

def fetch_real_data():
    weather = fetch_current_weather()
    aqi = fetch_current_aqi()
    if not weather or not aqi:
        return pd.DataFrame()

    # ✅ ensure timestamp is a proper UTC datetime
    now = datetime.now(timezone.utc)

    main = weather.get("main", {})
    wind = weather.get("wind", {})
    clouds = weather.get("clouds", {})
    iaqi = aqi.get("iaqi", {})

    row = {
        # ✅ store directly as datetime (not microseconds)
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
    # ✅ ensure consistent UTC type
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    return df

def preprocess_features(df):
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)

    df["lag_1"] = df["aqi_aqicn"].shift(1).fillna(method="ffill").fillna(0)
    df["lag_2"] = df["aqi_aqicn"].shift(2).fillna(method="ffill").fillna(0)
    df["rolling_mean_3"] = df["aqi_aqicn"].rolling(window=3, min_periods=1).mean()

    # Correct data types for Hopsworks schema
    float_cols = [
        "ow_temp", "ow_pressure", "ow_humidity",
        "ow_wind_speed", "ow_wind_deg", "ow_clouds",
        "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
        "aqi_aqicn", "lag_1", "lag_2", "rolling_mean_3"
    ]
    time_int_cols = ["hour", "day", "month", "weekday"]

    df[float_cols] = df[float_cols].astype("float64")
    df[time_int_cols] = df[time_int_cols].astype("int64")

    # ✅ make sure timestamps remain timezone-aware
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    return df

def backfill():
    if not HOPSWORKS_API_KEY:
        print("Missing Hopsworks API key.")
        return

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    try:
        fg = fs.get_feature_group("aqi_features", 1)
        df_existing = fg.read()
        print(f"Loaded {len(df_existing)} rows.")
    except Exception:
        df_existing = pd.DataFrame()
        fg = fs.create_feature_group(
            name="aqi_features",
            version=1,
            description="AQI plus weather data for Karachi",
            primary_key=["timestamp_utc"],
            event_time="timestamp_utc",
        )
        print("Created new feature group aqi_features v1.")

    df_new = fetch_real_data()
    if df_new.empty:
        print("Nothing to update.")
        return

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_clean = preprocess_features(df_combined)
    df_clean.drop_duplicates(subset=["timestamp_utc"], keep="last", inplace=True)

    print(f"Uploading {len(df_clean)} rows to Hopsworks.")
    fg.insert(df_clean, write_options={"wait_for_job": True})
    print("Backfill complete.")

if __name__ == "__main__":
    if not OPENWEATHER_API_KEY:
        print("Missing OpenWeather key.")
    elif not AQICN_TOKEN:
        print("Missing AQICN token.")
    else:
        print("Running real time backfill.")
        backfill()
