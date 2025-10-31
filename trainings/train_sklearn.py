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

# ─────────────────────────────────────────────
# 🪵 Logging Configuration
# ─────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────
# 🔐 Load Environment Variables
# ─────────────────────────────────────────────
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

# ─────────────────────────────────────────────
# 🧭 Connect to Hopsworks
# ─────────────────────────────────────────────
logging.info("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ─────────────────────────────────────────────
# 📥 Load Data from Feature Store
# ─────────────────────────────────────────────
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
logging.info(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 🧠 Prepare Features and Target (Forecast)
# ─────────────────────────────────────────────
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

# Predict NEXT AQI (shift -1)
df["target_aqi_next"] = df["aqi_aqicn"].shift(-1)
df.dropna(subset=feature_cols + ["target_aqi_next"], inplace=True)

X = df[feature_cols].astype("float64")
y = df["target_aqi_next"].astype("float64")

logging.info(f"✅ Data ready: {X.shape[0]} samples, {len(feature_cols)} features.")

# ─────────────────────────────────────────────
# ✂️ Train/Test Split (No Shuffle for Forecasting)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
logging.info(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# ⚖️ Standardize Features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 🌲 Train Random Forest Model
# ─────────────────────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

logging.info("🚀 Training Random Forest with 5-fold CV...")
cv_rmse = np.sqrt(
    -cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error")
)
logging.info(f"📊 CV RMSE (5-fold): {np.round(cv_rmse, 3)} | Mean: {np.mean(cv_rmse):.3f}")

rf_model.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# 📈 Evaluate Model
# ─────────────────────────────────────────────
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

logging.info(f"✅ Train R²: {train_r2:.3f}")
logging.info(f"✅ Test R²: {test_r2:.3f}")
logging.info(f"✅ Test RMSE: {rmse:.3f}")
logging.info(f"✅ Test MAE: {mae:.3f}")

# ─────────────────────────────────────────────
# 🔁 Retrain on Full Dataset for Deployment
# ─────────────────────────────────────────────
logging.info("🔁 Retraining model on full dataset for deployment...")
X_scaled_full = scaler.fit_transform(X)
rf_model.fit(X_scaled_full, y)

# ─────────────────────────────────────────────
# 💾 Save Model, Scaler, and Metadata
# ─────────────────────────────────────────────
model_dir = "rf_aqi_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, f"{model_dir}/model.pkl")
joblib.dump(scaler, f"{model_dir}/scaler.pkl")

metadata = {
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "cv_rmse_mean": float(np.mean(cv_rmse)),
    "cv_rmse_std": float(np.std(cv_rmse))
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logging.info("💾 Model, scaler, and metadata saved locally.")

# ─────────────────────────────────────────────
# 🚀 Upload to Hopsworks Model Registry
# ─────────────────────────────────────────────
mr = project.get_model_registry()
model = mr.python.create_model(
    name="rf_aqi_model",
    metrics=metadata,
    description="Random Forest model for AQI prediction trained to forecast next AQI value."
)
model.save(model_dir)
logging.info("🚀 Model successfully uploaded to Hopsworks Model Registry.")
