#!/usr/bin/env python3
import os
import joblib
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

# -------------------------
# Logging
# -------------------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# -------------------------
# Load env
# -------------------------
load_dotenv()
API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    raise ValueError("Missing HOPSWORKS API key in env (aqi_forecast_api_key / HOPSWORKS_API_KEY)")

# -------------------------
# Feature columns (match train_sklearn & backfill)
# -------------------------
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

def find_artifact_files(model_dir, names=("model.pkl", "scaler.pkl")):
    found = {}
    for root, _, files in os.walk(model_dir):
        for fname in files:
            if fname in names:
                found[fname] = os.path.join(root, fname)
    return found

def load_latest_model_and_scaler(mr, model_name="rf_aqi_model"):
    models = mr.get_models(model_name)
    if not models:
        raise RuntimeError(f"No models found in registry for name '{model_name}'")
    latest = max(models, key=lambda m: m.version)
    log.info(f"📦 Downloading model '{model_name}' version {latest.version} ...")
    model_dir = latest.download()
    log.info(f"📦 Model downloaded to: {model_dir}")

    found = find_artifact_files(model_dir)
    model_path = found.get("model.pkl")
    scaler_path = found.get("scaler.pkl")

    if not model_path:
        raise FileNotFoundError("model.pkl not found inside downloaded model artifact")
    if not scaler_path:
        raise FileNotFoundError("scaler.pkl not found inside downloaded model artifact")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    log.info("✅ Loaded model and scaler.")
    return model, scaler, latest.version

def main():
    log.info("🚀 Starting AQI prediction (latest row) ...")

    # Connect
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    log.info("✅ Connected to Hopsworks Feature Store & Model Registry.")

    # Read feature group
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    log.info(f"📥 Read {len(df)} rows from 'aqi_features'")

    if df.empty:
        log.error("Feature group is empty — nothing to predict.")
        return

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[[-1]].copy()
    log.info(f"📅 Using latest row timestamp: {latest['timestamp_utc'].iloc[0]}")

    # Ensure all required features exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in latest.columns:
            log.warning(f"⚠️ Column '{col}' missing — filling with 0")
            latest[col] = 0

    X = latest[FEATURE_COLS].astype("float64")

    # Load model & scaler
    model, scaler, model_version = load_latest_model_and_scaler(mr, model_name="rf_aqi_model")

    # Scale & predict
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    actual = None
    if "aqi_aqicn" in latest.columns:
        actual = latest["aqi_aqicn"].iloc[0]

    out = {
        "predicted_aqi": float(pred),
        "predicted_at_utc": datetime.utcnow().replace(tzinfo=None).isoformat(sep=" "),
        "model_version": int(model_version),
        "observed_aqi": float(actual) if actual is not None and not pd.isna(actual) else None
    }

    msg = (
        f"💨 Predicted AQI: {out['predicted_aqi']:.2f} "
        f"(model v{out['model_version']}) | Observed AQI: {out['observed_aqi']}"
        if out["observed_aqi"] is not None else
        f"💨 Predicted AQI: {out['predicted_aqi']:.2f} (model v{out['model_version']})"
    )
    log.info(msg)
    log.info("✅ Prediction completed successfully.")

if __name__ == "__main__":
    main()
