import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
import tensorflow as tf
from datetime import datetime, timedelta

# -----------------------------
# 🔐 Connect to Hopsworks
# -----------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------------
# 📥 Load Feature Data
# -----------------------------
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[TARGET] + FEATURE_COLS)

# -----------------------------
# 🔄 Load Model & Scalers
# -----------------------------
model_path = "models/tf_lstm_model.keras"
scaler_X_path = "models/tf_scaler.joblib"
scaler_y_path = "models/tf_y_scaler.joblib"

model = tf.keras.models.load_model(model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# -----------------------------
# 🔁 Create Input Sequence (last 7 days)
# -----------------------------
SEQUENCE_LENGTH = 7
latest_seq = df[FEATURE_COLS].iloc[-SEQUENCE_LENGTH:].values
latest_seq = latest_seq.reshape((1, SEQUENCE_LENGTH, len(FEATURE_COLS)))

# Scale features
nsamples, ntimesteps, nfeatures = latest_seq.shape
latest_scaled = scaler_X.transform(latest_seq.reshape(nsamples*ntimesteps, nfeatures))
latest_scaled = latest_scaled.reshape((nsamples, ntimesteps, nfeatures))

# -----------------------------
# 🔮 Forecast next 3 days
# -----------------------------
FORECAST_DAYS = 3
preds = []
timestamps = []

base_ts = pd.to_datetime(df.index[-1] if df.index.dtype.kind == 'M' else datetime.utcnow())

for i in range(FORECAST_DAYS):
    pred_scaled = model.predict(latest_scaled, verbose=0)
    pred_aqi = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]  # inverse-transform
    preds.append(pred_aqi)
    next_ts = base_ts + pd.Timedelta(days=i+1)
    timestamps.append(next_ts)

    # Prepare next input sequence by appending predicted features
    # For simplicity, we just shift the window, repeat last known features
    next_input = latest_scaled[:, 1:, :]  # drop oldest
    last_features = latest_scaled[:, -1:, :]  # last timestep
    next_input = np.concatenate([next_input, last_features], axis=1)
    latest_scaled = next_input

# -----------------------------
# 📋 Display Forecast
# -----------------------------
forecast_df = pd.DataFrame({
    "forecast_date": timestamps,
    "predicted_aqi": preds
})

print("\n🌤️ AQI Forecast for Next 3 Days:")
print(forecast_df)
