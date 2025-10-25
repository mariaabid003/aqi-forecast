import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

# 🔐 Connect to Hopsworks
load_dotenv()
API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

print("✅ Connected to Hopsworks")

# 📥 Load latest feature data
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]
df.drop(columns=["timestamp_utc"], errors="ignore", inplace=True)
df.dropna(subset=FEATURE_COLS, inplace=True)

# 🤖 Load latest model and scaler
all_models = mr.get_models("rf_aqi_model")
latest_model = max(all_models, key=lambda m: m.version)
model_dir = latest_model.download()

model, scaler = None, None
for root, _, files in os.walk(model_dir):
    for f in files:
        if f == "model.pkl":
            model = joblib.load(os.path.join(root, f))
        elif f == "scaler.pkl":
            scaler = joblib.load(os.path.join(root, f))

print(f"✅ Loaded model version {latest_model.version}")

# 🔮 Predict AQI
X_scaled = scaler.transform(df[FEATURE_COLS].values)
preds = model.predict(X_scaled)

print("\n🌤️ Predicted AQI values (last 5):")
print(preds[-5:])
