import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 🔐 Load environment variables
# -----------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

# -----------------------------
# 📦 Configuration
# -----------------------------
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "rf_aqi_model"
MODEL_DIR = "rf_aqi_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# ⚙️ Utility functions
# -----------------------------
def preprocess_features(df):
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)

    # Lags and rolling mean (match backfill.py)
    if "lag_1" not in df.columns:
        df["lag_1"] = df["aqi_aqicn"].shift(1).bfill().fillna(0)
    if "lag_2" not in df.columns:
        df["lag_2"] = df["aqi_aqicn"].shift(2).bfill().fillna(0)
    if "rolling_mean_3" not in df.columns:
        df["rolling_mean_3"] = df["aqi_aqicn"].rolling(window=3, min_periods=1).mean()

    df.dropna(inplace=True)
    return df

def train_test_split_time(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

# -----------------------------
# 🚀 Connect to Hopsworks
# -----------------------------
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------------
# 📥 Load Feature Data
# -----------------------------
fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
df = fg.read()
print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

df = preprocess_features(df)

# -----------------------------
# 🧩 Prepare features and labels
# -----------------------------
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed",
    "ow_wind_deg", "ow_clouds", "ow_co", "ow_no2",
    "ow_pm2_5", "ow_pm10", "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

target_col = "aqi_aqicn"
df = df.dropna(subset=[target_col]).reset_index(drop=True)

X = df[feature_cols]
y = df[target_col]

print(f"✅ Data ready: {len(X)} samples, {len(feature_cols)} features.")

# -----------------------------
# ⏳ Time-based Split
# -----------------------------
train_df, test_df = train_test_split_time(df, test_size=0.2)
X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]
print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------------
# ⚖️ Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 🌲 Model Training (Random Forest)
# -----------------------------
print("🚀 Training Random Forest with 5-fold TimeSeries CV...")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

tscv = TimeSeriesSplit(n_splits=5)
cv_rmse = []

for train_idx, val_idx in tscv.split(X_train_scaled):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    cv_rmse.append(rmse)

print(f"📊 CV RMSE (5-fold): {np.round(cv_rmse, 3)} | Mean: {np.mean(cv_rmse):.3f}")

# -----------------------------
# 🧠 Evaluate on Train & Test
# -----------------------------
rf.fit(X_train_scaled, y_train)
y_train_pred = rf.predict(X_train_scaled)
y_test_pred = rf.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"✅ Train R²: {train_r2:.3f}")
print(f"✅ Test  R²: {test_r2:.3f}")
print(f"✅ Test RMSE: {test_rmse:.3f}")
print(f"✅ Test MAE: {test_mae:.3f}")

# -----------------------------
# 🔁 Retrain on Full Data
# -----------------------------
print("🔁 Retraining model on full dataset for deployment...")
rf.fit(scaler.fit_transform(X), y)

# -----------------------------
# 💾 Save Model and Metadata
# -----------------------------
joblib.dump(rf, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

metadata = {
    "model_name": MODEL_NAME,
    "version": None,
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "rmse": float(test_rmse),
    "mae": float(test_mae),
    "features": feature_cols,
    "target": target_col
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("💾 Model, scaler, and metadata saved locally.")

# -----------------------------
# ⬆️ Upload to Hopsworks
# -----------------------------
mr = project.get_model_registry()
model = mr.python.create_model(
    name=MODEL_NAME,
    metrics={"train_r2": train_r2, "test_r2": test_r2, "rmse": test_rmse, "mae": test_mae},
    description="Random Forest model for AQI forecasting with temporal features"
)

model.save(MODEL_DIR)
print(f"🚀 Model successfully uploaded to Hopsworks Model Registry.")
print("✅ Done.")
