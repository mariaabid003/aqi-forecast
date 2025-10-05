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

# -----------------------------
# Hopsworks Integration
# -----------------------------
print("🔐 Connecting to Hopsworks...")

# You can store your API key safely in hopsworks_api.key file OR in .env
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    # Fallback: direct string (your actual API key)
    HOPSWORKS_API_KEY = "n5hGwARKFatp9aKl.BxHt0Ymf1IZFSW87cmtIJbUqnvjyVO0a713g2RwvPvUfh04frFh7CYyWfwhAtqmV"

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Load feature group data from Hopsworks
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()  # Reads entire feature group as a DataFrame

print("✅ Data loaded from Hopsworks Feature Store!")
print("Dataset shape:", df.shape)

# -----------------------------
# Data Preprocessing
# -----------------------------
df = df.dropna(subset=["aqi_aqicn"])
df.fillna(method="ffill", inplace=True)  # forward-fill missing values

# -----------------------------
# Features & Target
# -----------------------------
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg", "ow_clouds",
    "ow_co", "ow_no", "ow_no2", "ow_o3", "ow_so2", "ow_pm2_5", "ow_pm10", "ow_nh3",
    "aqicn_co", "aqicn_no2", "aqicn_pm25", "aqicn_pm10", "aqicn_o3", "aqicn_so2",
    "hour", "day", "month", "weekday"
]
target_col = "aqi_aqicn"

X = df[feature_cols]
y = df[target_col]

print("✅ Features shape:", X.shape)
print("✅ Target shape:", y.shape)
print("✅ Missing target values:", y.isna().sum())
print(df[[target_col]].tail())


# -----------------------------
# Train/Test Split
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
# Model Training
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

print("\n🌲 Random Forest Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

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
    description="Random Forest model for Karachi AQI forecasting"
)
model_meta.save(MODEL_PATH)
print("🚀 Model uploaded to Hopsworks Model Registry!")

print("\n🏁 Training completed successfully!")
