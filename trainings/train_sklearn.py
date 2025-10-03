# trainings/train_sklearn.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- Load dataset ---
DATA_PATH = "data/features/training_dataset.parquet"
df = pd.read_parquet(DATA_PATH)
print("Dataset shape:", df.shape)

# --- Fill missing values ---
df.fillna(method="ffill", inplace=True)  # forward-fill missing values

# --- Features & Target ---
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg", "ow_clouds",
    "ow_co", "ow_no", "ow_no2", "ow_o3", "ow_so2", "ow_pm2_5", "ow_pm10", "ow_nh3",
    "aqicn_co", "aqicn_no2", "aqicn_pm25", "aqicn_pm10", "aqicn_o3", "aqicn_so2",
    "hour", "day", "month", "weekday"
]
target_col = "aqi_aqicn"

X = df[feature_cols]
y = df[target_col]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = rf_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# --- Save model and scaler ---
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/rf_model.joblib"
SCALER_PATH = "models/rf_scaler.joblib"

joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
