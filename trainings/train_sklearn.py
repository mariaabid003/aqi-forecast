import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "rf_aqi_model"
MODEL_DIR = "rf_aqi_model"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_features(df):
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
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
    return df.iloc[:split_idx], df.iloc[split_idx:]

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
df = fg.read()
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

df = preprocess_features(df)

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

print(f"Data ready for training: {len(X)} samples with {len(feature_cols)} features.")

train_df, test_df = train_test_split_time(df)
X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest model...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nModel performance on test data:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

rf.fit(scaler.fit_transform(X), y)

joblib.dump(rf, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

metadata = {
    "model_name": MODEL_NAME,
    "version": None,
    "r2": float(r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "features": feature_cols,
    "target": target_col
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\nModel and metadata saved locally.")

mr = project.get_model_registry()
model = mr.python.create_model(
    name=MODEL_NAME,
    metrics={"r2": r2, "rmse": rmse, "mae": mae},
    description="Random Forest model for AQI forecasting with weather and temporal features."
)

model.save(MODEL_DIR)
print("Model successfully uploaded to Hopsworks.")
