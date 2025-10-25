import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dotenv import load_dotenv
import json
import logging

# ──────────────────────────────────────────────────────────────
# ✅ Setup Logging
# ──────────────────────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ──────────────────────────────────────────────────────────────
# ✅ Load Environment Variables
# ──────────────────────────────────────────────────────────────
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

# ──────────────────────────────────────────────────────────────
# 🔐 Connect to Hopsworks
# ──────────────────────────────────────────────────────────────
logging.info("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ──────────────────────────────────────────────────────────────
# 📥 Load Data from Feature Store
# ──────────────────────────────────────────────────────────────
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
logging.info(f"✅ Data loaded from Hopsworks Feature Store! Shape: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 🧹 Data Cleaning & Preprocessing
# ──────────────────────────────────────────────────────────────
df = df.dropna(subset=["aqi_aqicn"])
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# Sort by datetime if available
if "datetime" in df.columns:
    df = df.sort_values("datetime").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# 🧠 Feature Engineering: Add lags & rolling mean
# ──────────────────────────────────────────────────────────────
df["lag_1"] = df["aqi_aqicn"].shift(1)
df["lag_2"] = df["aqi_aqicn"].shift(2)
df["rolling_mean_3"] = df["aqi_aqicn"].rolling(window=3).mean()
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]
target_col = "aqi_aqicn"

df = df.dropna(subset=feature_cols)
X = df[feature_cols]
y = df[target_col]
logging.info(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")

# ──────────────────────────────────────────────────────────────
# 🧪 Train/Test Split
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
logging.info(f"✅ Split: Train={len(X_train)}, Test={len(X_test)}")

# ──────────────────────────────────────────────────────────────
# ⚖️ Feature Scaling
# ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────
# 🌲 Train Model (Optimized Random Forest)
# ──────────────────────────────────────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

cv_rmse = np.sqrt(-cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error"))
logging.info(f"📊 CV RMSE (train folds): {np.round(cv_rmse, 3)}")

rf_model.fit(X_train_scaled, y_train)

# ──────────────────────────────────────────────────────────────
# 📊 Model Evaluation
# ──────────────────────────────────────────────────────────────
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

logging.info(f"✅ Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
logging.info(f"✅ Test RMSE: {rmse:.3f} | MAE: {mae:.3f}")

# ──────────────────────────────────────────────────────────────
# 🔄 Re-train on Full Dataset
# ──────────────────────────────────────────────────────────────
logging.info("🔄 Re-training on full dataset for deployment...")
X_scaled_full = scaler.fit_transform(X)
rf_model.fit(X_scaled_full, y)

# ──────────────────────────────────────────────────────────────
# 💾 Save Model & Scaler Locally
# ──────────────────────────────────────────────────────────────
model_dir = "rf_aqi_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, f"{model_dir}/model.pkl")
joblib.dump(scaler, f"{model_dir}/scaler.pkl")

metadata = {
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "cv_rmse_mean": float(np.mean(cv_rmse))
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logging.info(f"✅ Model and scaler saved in '{model_dir}'")

# ──────────────────────────────────────────────────────────────
# 🚀 Upload to Hopsworks Model Registry
# ──────────────────────────────────────────────────────────────
mr = project.get_model_registry()
model = mr.python.create_model(
    name="rf_aqi_model",
    metrics=metadata,
    description="Optimized Random Forest for AQI prediction (lags + rolling mean)."
)
model.save(model_dir)

logging.info("🚀 Model successfully uploaded to Hopsworks Registry.")

