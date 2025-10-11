import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
from datetime import timedelta
from dotenv import load_dotenv

# ----------------------------------
# CONFIGURATION
# ----------------------------------
MODEL_DIR = "models"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
TARGET_COL = "aqi_aqicn"
FORECAST_DAYS = 3

# The same 14 features used during training
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

# ----------------------------------
# LOAD ENV VARIABLES & CONNECT
# ----------------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")

print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ----------------------------------
# LOAD LATEST FEATURES
# ----------------------------------
print("📥 Loading latest feature data from Hopsworks...")
feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
)

df = feature_group.read()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

# Drop rows without target and sort
df = df.dropna(subset=[TARGET_COL])
df.sort_values("timestamp_utc", inplace=True)

# Get the latest row as base input
latest_row = df.iloc[-1:].copy()
print(f"✅ Latest record timestamp: {latest_row['timestamp_utc'].values[0]}")

# ----------------------------------
# LOAD MODEL AND SCALER
# ----------------------------------
scaler_path = os.path.join(MODEL_DIR, "rf_scaler.joblib")
model_path = os.path.join(MODEL_DIR, "rf_model.joblib")

if not os.path.exists(scaler_path) or not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model or scaler files not found in models/ folder.")

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Keep only the features used during training
X_latest = latest_row[FEATURE_COLS].copy()

# ----------------------------------
# FORECAST LOOP (3 sequential days)
# ----------------------------------
predictions = []
base_timestamp = pd.to_datetime(latest_row["timestamp_utc"].values[0])

print("\n🔮 Forecasting next 3 days of AQI...")

for i in range(FORECAST_DAYS):
    # Scale features
    X_scaled = scaler.transform(X_latest)

    # Predict AQI
    predicted_aqi = model.predict(X_scaled)[0]
    predictions.append(predicted_aqi)

    # Prepare next day's features (simple temporal shift)
    next_input = X_latest.copy()
    next_input["day"] = (next_input["day"] + 1).clip(upper=31)
    next_input["weekday"] = (next_input["weekday"] + 1) % 7
    next_input["month"] = base_timestamp.month  # keep same month

    # Simulate slight atmospheric progression
    next_input["ow_pm2_5"] *= 1.02
    next_input["ow_pm10"] *= 1.02
    next_input["ow_no2"] *= 1.01

    X_latest = next_input  # feed this to next iteration

# ----------------------------------
# STORE RESULTS
# ----------------------------------
future_dates = pd.date_range(
    start=base_timestamp + timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

forecast_df = pd.DataFrame({
    "forecast_date": future_dates,
    "predicted_aqi": np.round(predictions, 2)
})

# Save locally for dashboard
os.makedirs("data", exist_ok=True)
forecast_csv = os.path.join("data", "forecast_next_3_days.csv")
forecast_df.to_csv(forecast_csv, index=False)

print("\n🌤️ AQI Forecast for Next 3 Days:")
print(forecast_df)

# ----------------------------------
# OPTIONAL: SAVE FORECAST TO HOPSWORKS FEATURE GROUP
# ----------------------------------
try:
    forecast_fg = fs.get_or_create_feature_group(
        name="aqi_forecasts",
        version=1,
        primary_key=["forecast_date"],
        description="Predicted AQI values for next 3 days"
    )
    forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
    print("🚀 Forecast uploaded to Hopsworks Feature Store!")
except Exception as e:
    print(f"⚠️ Could not upload forecast to Feature Store: {e}")

print("\n🏁 3-Day Forecast Generation Complete!")
