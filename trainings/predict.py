import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
import logging
import glob
from datetime import datetime

# ─────────────────────────────────────────────
# 🪵 Logging Configuration
# ─────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────
# 🔐 Load Environment Variables
# ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

# ─────────────────────────────────────────────
# 🔗 Connect to Hopsworks
# ─────────────────────────────────────────────
logging.info("🚀 Starting AQI prediction pipeline...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()
logging.info("✅ Connected to Hopsworks Feature Store & Model Registry.")

# ─────────────────────────────────────────────
# 📥 Load Latest Feature Data
# ─────────────────────────────────────────────
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
logging.info(f"📊 Retrieved {df.shape[0]} rows from 'aqi_features'.")

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

# ─────────────────────────────────────────────
# 🎯 Select the Latest (Today’s) Row
# ─────────────────────────────────────────────
latest = df.sort_values(by="datetime", ascending=False).head(1) if "datetime" in df.columns else df.tail(1)
logging.info(f"📅 Using the latest record for prediction ({len(latest)} row).")

# ─────────────────────────────────────────────
# 🧩 Prepare Features
# ─────────────────────────────────────────────
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

latest = latest.dropna(subset=feature_cols)
X = latest[feature_cols].astype("float64")

# ─────────────────────────────────────────────
# 🤖 Load Latest Model & Scaler
# ─────────────────────────────────────────────
all_models = mr.get_models("rf_aqi_model")
latest_model = max(all_models, key=lambda m: m.version)
model_dir = latest_model.download()
logging.info(f"📦 Downloaded model version {latest_model.version} to: {model_dir}")

# Robust path search for model/scaler
model_path = glob.glob(os.path.join(model_dir, "**/model.pkl"), recursive=True)
scaler_path = glob.glob(os.path.join(model_dir, "**/scaler.pkl"), recursive=True)

if not model_path:
    raise FileNotFoundError("❌ model.pkl not found in downloaded directory.")

model = joblib.load(model_path[0])
scaler = joblib.load(scaler_path[0]) if scaler_path else None
logging.info("✅ Model and scaler loaded successfully.")

# ─────────────────────────────────────────────
# 🔮 Predict AQI for Today
# ─────────────────────────────────────────────
X_scaled = scaler.transform(X.values) if scaler else X.values
pred = model.predict(X_scaled)[0]
logging.info("✅ AQI prediction completed.")

# ─────────────────────────────────────────────
# 📊 Show Result
# ─────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n🌤️ Today's AQI Prediction:")
print(f"📅 Date: {today}")
print(f"💨 Predicted AQI: {pred:.2f}")
