# features/backfill.py
import os
import requests
import pandas as pd
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load env vars
load_dotenv()

OW_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

# Karachi coords
LAT = 24.8607
LON = 67.0011


def fetch_openweather_current():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OW_KEY}&units=metric"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_openweather_forecast():
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OW_KEY}&units=metric"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_openweather_air():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OW_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_aqicn_current():
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", {})


def backfill(out_file="data/features/training_dataset"):
    os.makedirs("data/features", exist_ok=True)
    rows = []

    # --- Current Weather ---
    weather = fetch_openweather_current()
    air = fetch_openweather_air()
    aqicn = fetch_aqicn_current()

    ts = datetime.fromtimestamp(weather["dt"], tz=timezone.utc)

    row = {
        "timestamp_utc": ts,
        "ow_temp": weather["main"]["temp"],
        "ow_pressure": weather["main"]["pressure"],
        "ow_humidity": weather["main"]["humidity"],
        "ow_wind_speed": weather["wind"]["speed"],
        "ow_wind_deg": weather["wind"]["deg"],
        "ow_clouds": weather["clouds"]["all"],
    }

    # OpenWeather air quality
    if air.get("list"):
        comps = air["list"][0]["components"]
        row.update({
            "ow_co": comps.get("co"),
            "ow_no": comps.get("no"),
            "ow_no2": comps.get("no2"),
            "ow_o3": comps.get("o3"),
            "ow_so2": comps.get("so2"),
            "ow_pm2_5": comps.get("pm2_5"),
            "ow_pm10": comps.get("pm10"),
            "ow_nh3": comps.get("nh3"),
        })

    # AQICN snapshot
    iaqi = aqicn.get("iaqi", {})
    row.update({
        "aqi_aqicn": aqicn.get("aqi"),
        "aqicn_co": iaqi.get("co", {}).get("v"),
        "aqicn_no2": iaqi.get("no2", {}).get("v"),
        "aqicn_pm25": iaqi.get("pm25", {}).get("v"),
        "aqicn_pm10": iaqi.get("pm10", {}).get("v"),
        "aqicn_o3": iaqi.get("o3", {}).get("v"),
        "aqicn_so2": iaqi.get("so2", {}).get("v"),
    })

    # Time-based features
    row["hour"] = ts.hour
    row["day"] = ts.day
    row["month"] = ts.month
    row["weekday"] = ts.weekday()

    rows.append(row)

    # --- Forecast Weather ---
    forecast = fetch_openweather_forecast()
    for entry in forecast["list"]:
        ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)

        row = {
            "timestamp_utc": ts,
            "ow_temp": entry["main"]["temp"],
            "ow_pressure": entry["main"]["pressure"],
            "ow_humidity": entry["main"]["humidity"],
            "ow_wind_speed": entry["wind"]["speed"],
            "ow_wind_deg": entry["wind"]["deg"],
            "ow_clouds": entry["clouds"]["all"],
            "hour": ts.hour,
            "day": ts.day,
            "month": ts.month,
            "weekday": ts.weekday(),
        }
        rows.append(row)

    # --- Build DataFrame ---
    df_new = pd.DataFrame(rows)

    # Append if file exists
    if os.path.exists(out_file + ".parquet"):
        df_old = pd.read_parquet(out_file + ".parquet")
        df = pd.concat([df_old, df_new], ignore_index=True)
        # Drop duplicates (by timestamp)
        df = df.drop_duplicates(subset=["timestamp_utc"])
    else:
        df = df_new

    # Save updated dataset
    df.to_parquet(out_file + ".parquet", index=False)
    df.to_csv(out_file + ".csv", index=False)

    print(f"✅ Dataset updated with {len(df_new)} new rows.")
    print(f"📊 Total dataset size: {len(df)} rows.")
    print(df.tail())


if __name__ == "__main__":
    if OW_KEY:
        print("Using OpenWeather key:", OW_KEY[:6] + "...")
    else:
        print("❌ No OpenWeather key found! Check your .env file.")

    backfill()
