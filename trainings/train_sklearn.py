# trainings/train_sklearn.py

import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv

# -----------------------------
# ✅ Load environment variables
# -----------------------------
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------------
# Load feature group data
# -----------------------------
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()

print("✅ Data loaded from Hopsworks Feature Store!")
print("📊 Dataset shape:", df.shape)

# -----------------------------
# Data Preprocessing
# -----------------------------
df = df.dropna(subset=["aqi_aqicn"])
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]
target_col = "aqi_aqicn"

# Drop rows with missing features
df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df[target_col]

print("✅ Features shape:", X.shape)
print("✅ Target shape:", y.shape)

# -----------------------------
# Train/Test Split for evaluation only
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model Training on train split
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = rf_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🌲 Random Forest Evaluation (on test split):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# -----------------------------
# Refit on full dataset for deployment
# -----------------------------
print("🔄 Re-training on full dataset for deployment...")
X_scaled_full = scaler.fit_transform(X)
rf_model.fit(X_scaled_full, y)

# -----------------------------
# Save Model & Scaler
# -----------------------------
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/rf_model.joblib"
SCALER_PATH = "models/rf_scaler.joblib"

joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")

# -----------------------------
# Upload Model to Hopsworks Model Registry
# -----------------------------
mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="rf_aqi_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="Random Forest model for Karachi AQI forecasting (trained on full dataset)"
)
model_meta.save(MODEL_PATH)

print("🚀 Model uploaded to Hopsworks Model Registry!")
print("\n🏁 Training completed successfully!")
