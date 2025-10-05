# features/backfill.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
import hopsworks

# --- Load environment variables ---
load_dotenv()

OW_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")

# --- Location ---
CITY = "Karachi"
LAT = 24.8607
LON = 67.0011


# --- Fetch current data from APIs ---
def fetch_current_weather():
    """Fetch current weather data from OpenWeather API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OW_KEY}&units=metric"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch weather: {resp.status_code}")
            return {}
        return resp.json()
    except Exception as e:
        print("⚠️ Error fetching weather:", e)
        return {}


def fetch_current_aqi():
    """Fetch current air quality data from AQICN API."""
    try:
        url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch AQI: {resp.status_code}")
            return {}
        return resp.json().get("data", {})
    except Exception as e:
        print("⚠️ Error fetching AQI:", e)
        return {}


# --- Build dataframe from real data ---
def fetch_real_data():
    """Fetch and structure live data from OpenWeather + AQICN."""
    print("🌍 Fetching real AQI and weather data...")
    weather = fetch_current_weather()
    aqi = fetch_current_aqi()

    if not weather or not aqi:
        print("⚠️ Missing weather or AQI data, skipping update.")
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
        "ow_no": iaqi.get("no", {}).get("v"),
        "ow_no2": iaqi.get("no2", {}).get("v"),
        "ow_o3": iaqi.get("o3", {}).get("v"),
        "ow_so2": iaqi.get("so2", {}).get("v"),
        "ow_pm2_5": iaqi.get("pm25", {}).get("v"),
        "ow_pm10": iaqi.get("pm10", {}).get("v"),
        "ow_nh3": iaqi.get("nh3", {}).get("v"),
        "aqi_aqicn": aqi.get("aqi"),
        "aqicn_co": iaqi.get("co", {}).get("v"),
        "aqicn_no2": iaqi.get("no2", {}).get("v"),
        "aqicn_pm25": iaqi.get("pm25", {}).get("v"),
        "aqicn_pm10": iaqi.get("pm10", {}).get("v"),
        "aqicn_o3": iaqi.get("o3", {}).get("v"),
        "aqicn_so2": iaqi.get("so2", {}).get("v"),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),
    }

    df = pd.DataFrame([row])

    # Clean numeric types
    for col in df.columns:
        if col != "timestamp_utc":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("✅ Real-time data fetched successfully!")
    print(df.head())
    return df


# --- Backfill and push to Hopsworks ---
def backfill(out_file="data/features/training_dataset"):
    os.makedirs("data/features", exist_ok=True)
    df_new = fetch_real_data()

    if df_new.empty:
        print("⚠️ No new data fetched. Aborting backfill.")
        return

    # Merge with previous local data
    if os.path.exists(out_file + ".parquet"):
        df_old = pd.read_parquet(out_file + ".parquet")
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
    else:
        df = df_new

    # Save locally
    df.to_parquet(out_file + ".parquet", index=False)
    df.to_csv(out_file + ".csv", index=False)

    print(f"✅ Added 1 new real-time record. Total rows: {len(df)}")

    # --- Upload to Hopsworks ---
    if HOPSWORKS_API_KEY:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            description="Real AQI + weather dataset for Karachi",
            primary_key=["timestamp_utc"],
            event_time="timestamp_utc",
        )

        # ✅ Ensure correct dtypes
        int_cols = [
            "ow_pressure", "ow_humidity", "ow_wind_deg", "ow_clouds",
            "hour", "day", "month", "weekday"
        ]
        for col in int_cols:
            if col in df_new.columns:
                df_new[col] = df_new[col].astype("int64", errors="ignore")

        float_cols = [
            "ow_temp", "ow_wind_speed", "ow_co", "ow_no", "ow_no2",
            "ow_o3", "ow_so2", "ow_pm2_5", "ow_pm10", "ow_nh3",
            "aqi_aqicn", "aqicn_co", "aqicn_no2", "aqicn_pm25",
            "aqicn_pm10", "aqicn_o3", "aqicn_so2"
        ]
        for col in float_cols:
            if col in df_new.columns:
                df_new[col] = df_new[col].astype("float64", errors="ignore")

        print("📤 Inserting data to Hopsworks Feature Store...")
        fg.insert(df_new)
        print("🚀 Real data successfully pushed to Hopsworks.")
    else:
        print("⚠️ Missing HOPSWORKS_API_KEY – skipping upload.")


if __name__ == "__main__":
    if not OW_KEY:
        print("❌ No OpenWeather key found!")
    elif not AQICN_TOKEN:
        print("❌ No AQICN token found!")
    else:
        print(f"Using OpenWeather key: {OW_KEY[:6]}... and AQICN token: {AQICN_TOKEN[:6]}...")
        backfill()
