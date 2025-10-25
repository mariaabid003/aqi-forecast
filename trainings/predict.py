import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime


# ========================================
# 🔐 Step 1: Connect to Hopsworks
# ========================================
def connect_hopsworks():
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")
    if not HOPSWORKS_API_KEY:
        raise ValueError("❌ Missing Hopsworks API key!")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    print("✅ Connected to Hopsworks!")
    return fs


# ========================================
# 📥 Step 2: Load and Clean Feature Data
# ========================================
def load_and_clean_data(fs):
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    print(f"📊 Loaded {df.shape[0]} rows from feature store")

    # Replace infinite values and fill missing data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    TARGET = "aqi_aqicn"
    FEATURE_COLS = [
        "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
        "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
        "hour", "day", "month", "weekday"
    ]

    # Clean columns
    df = df.drop(columns=["timestamp_utc"], errors="ignore")
    df = df.dropna(subset=[TARGET] + FEATURE_COLS)
    if df.empty:
        raise ValueError("🚨 Dataset is empty after cleaning!")

    return df, TARGET, FEATURE_COLS


# ========================================
# ⚙️ Step 3: Load Model and Scaler
# ========================================
def load_model_and_scaler():
    model_path = "models/rf_model.joblib"
    scaler_path = "models/rf_scaler.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Random Forest model not found!")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("❌ Scaler file not found!")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Random Forest model and scaler loaded successfully!")
    return model, scaler


# ========================================
# 🔮 Step 4: Make Predictions on All Data
# ========================================
def make_predictions(model, scaler, df, FEATURE_COLS):
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    forecast_df = pd.DataFrame({
        "timestamp": df.index if df.index.dtype.kind == 'M' else range(len(df)),
        "predicted_aqi": preds
    })

    print("✅ Predictions generated for all available data!")
    return forecast_df


# ========================================
# 💾 Step 5: Save or Upload Forecasts
# ========================================
def save_forecasts(forecast_df):
    output_dir = "data/predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "aqi_forecast_latest.csv")
    forecast_df.to_csv(output_path, index=False)
    print(f"💾 Forecast saved locally at: {output_path}")


# ========================================
# 🚀 Step 6: Main Flow
# ========================================
def main():
    print("🚀 Starting AQI Forecast Pipeline (Random Forest)...\n")

    fs = connect_hopsworks()
    df, TARGET, FEATURE_COLS = load_and_clean_data(fs)
    model, scaler = load_model_and_scaler()
    forecast_df = make_predictions(model, scaler, df, FEATURE_COLS)
    save_forecasts(forecast_df)

    print("\n🌤️ AQI Forecast completed successfully!")
    print(forecast_df.tail(5))  # Show last 5 predictions


if __name__ == "__main__":
    main()
