#!/usr/bin/env python3
import os
import glob
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
# Feature columns (must match train)
# -------------------------
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

def find_artifact_files(model_dir, names=("model.pkl", "scaler.pkl")):
    """Return paths for model.pkl and scaler.pkl if present."""
    found = {}
    for root, _, files in os.walk(model_dir):
        for fname in files:
            if fname in names:
                found[fname] = os.path.join(root, fname)
    return found

def load_latest_model_and_scaler(mr, model_name="rf_aqi_model"):
    """Download latest model from registry and return (model, scaler)."""
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

    # Use timestamp_utc to get latest
    if "timestamp_utc" not in df.columns:
        log.error("Expected column 'timestamp_utc' not present in feature group.")
        return

    # Keep timezone-aware timestamps (assume already stored)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[[-1]].copy()  # preserve DataFrame

    log.info(f"📅 Using latest row timestamp: {latest['timestamp_utc'].iloc[0]}")

    # Ensure all required feature columns are present
    missing_cols = [c for c in FEATURE_COLS if c not in latest.columns]
    if missing_cols:
        log.error(f"Missing required feature columns for prediction: {missing_cols}")
        return

    # Select features (in exact order used in training)
    X = latest[FEATURE_COLS].astype("float64")

    # Load model & scaler
    model, scaler, model_version = load_latest_model_and_scaler(mr, model_name="rf_aqi_model")

    # Scale & predict
    X_scaled = scaler.transform(X.values)  # scaler expects same columns
    pred = model.predict(X_scaled)[0]

    # Observed AQI if available in feature group row
    actual = None
    if "aqi_aqicn" in latest.columns:
        actual = latest["aqi_aqicn"].iloc[0]

    # Output
    out = {
        "predicted_aqi": float(pred),
        "predicted_at_utc": datetime.utcnow().replace(tzinfo=None).isoformat(sep=" "),
        "model_version": int(model_version),
        "observed_aqi": float(actual) if actual is not None and not pd.isna(actual) else None
    }

    msg = (
        f"💨 Predicted AQI: {out['predicted_aqi']:.2f} "
        f"(model v{out['model_version']}) | "
        f"Observed AQI: {out['observed_aqi']}" if out["observed_aqi"] is not None else
        f"💨 Predicted AQI: {out['predicted_aqi']:.2f} (model v{out['model_version']})"
    )
    log.info(msg)

    # Close Hopsworks client cleanup
    log.info("Done.")

if __name__ == "__main__":
    main()
