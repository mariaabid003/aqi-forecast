import os
import pandas as pd
import numpy as np
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# ============================================================
# 🔐 CONNECT TO HOPSWORKS
# ============================================================
print("🔐 Connecting to Hopsworks...")
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ============================================================
# 📡 LOAD DATA
# ============================================================
print("📡 Loading feature group...")
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
print(f"✅ Data loaded from Hopsworks. Shape: {df.shape}")

# ============================================================
# 🧹 CLEAN DATA
# ============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[TARGET] + feature_cols)
if df.empty:
    raise ValueError("🚨 Dataset empty after cleaning!")

# ============================================================
# ⚙️ Evaluate Random Forest Model
# ============================================================
print("\n🔍 Evaluating rf_aqi_model...")
try:
    rf_model_path = os.path.join("rf_aqi_model", "pipeline.pkl")
    rf_model = joblib.load(rf_model_path)

    X = df[feature_cols]
    y = df[TARGET]
    y_pred = rf_model.predict(X)

    rf_rmse = np.sqrt(mean_squared_error(y, y_pred))
    rf_mae = mean_absolute_error(y, y_pred)
    rf_r2 = r2_score(y, y_pred)

    print(f"✅ rf_aqi_model - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}")

except Exception as e:
    print(f"❌ Could not load or evaluate rf_aqi_model: {e}")

# ============================================================
# ⚙️ Evaluate TensorFlow LSTM Model
# ============================================================
print("\n🔍 Evaluating tf_lstm_aqi_model...")
try:
    SEQ_LEN = 7

    # Recreate sequences
    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(df)):
        X_seq.append(df[feature_cols].iloc[i - SEQ_LEN:i].values)
        y_seq.append(df[TARGET].iloc[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 80/20 split
    split_idx = int(len(X_seq) * 0.8)
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

    # Load saved scalers
    scaler_X = joblib.load("models/tf_scaler.joblib")
    scaler_y = joblib.load("models/tf_y_scaler.joblib")

    # Scale inputs (same way as training)
    ns, nt, nf = X_test.shape
    X_scaled = scaler_X.transform(X_test.reshape(ns * nt, nf)).reshape(ns, nt, nf)

    # Load model
    model = load_model("models/tf_lstm_model.keras")

    # Predict and inverse transform
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    lstm_mae = mean_absolute_error(y_test, y_pred)
    lstm_r2 = r2_score(y_test, y_pred)

    print(f"✅ tf_lstm_aqi_model - RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}, R²: {lstm_r2:.2f}")

except Exception as e:
    print(f"❌ Could not evaluate tf_lstm_aqi_model: {e}")

# ============================================================
# 🏁 DONE
# ============================================================
print("\n🏁 Evaluation completed successfully!")
