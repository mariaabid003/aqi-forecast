# trainings/train_tf.py

import os
import time
import pandas as pd
import numpy as np
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 🔐 Connect to Hopsworks
# -----------------------------
print("🔐 Connecting to Hopsworks...")

load_dotenv()  # ✅ Load .env so your existing aqi_forecast_api_key works
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key! Please set it in .env or environment variables.")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------------
# 📥 Load data directly from Feature Group (safe)
# -----------------------------
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()

print("✅ Data loaded from Hopsworks Feature Store!")
print("Dataset shape:", df.shape)

# -----------------------------
# 🧹 Data Cleaning
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
if TARGET not in df.columns:
    raise ValueError(f"🚨 Target column '{TARGET}' not found!")

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[TARGET])

if df.empty:
    raise ValueError("🚨 Dataset is empty after cleaning!")

# -----------------------------
# 🎯 Feature and Target Setup
# -----------------------------
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df[TARGET]

print("✅ Features shape:", X.shape)
print("✅ Target shape:", y.shape)

if len(df) < 20:
    raise ValueError("🚨 Not enough data for training! Need at least 20 samples.")

# -----------------------------
# ✂️ Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# ⚖️ Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
SCALER_PATH = "models/tf_scaler.joblib"
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved to {SCALER_PATH}")

# -----------------------------
# 🧱 Build TensorFlow Model
# -----------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# 🚀 Train Model with Early Stopping
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("🚀 Training TensorFlow model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1,
    shuffle=True
)

# -----------------------------
# 📊 Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled).flatten()
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print("\n📈 TensorFlow Model Evaluation:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# -----------------------------
# 💾 Save Model Locally
# -----------------------------
MODEL_PATH = "models/tf_model.keras"
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# -----------------------------
# ☁️ Upload Model to Hopsworks Model Registry
# -----------------------------
mr = project.get_model_registry()
unique_id = int(time.time())

model_meta = mr.python.create_model(
    name=f"tf_aqi_model_{unique_id}",
    metrics={"MSE": mse, "MAE": mae, "R2": r2},
    description=f"TensorFlow AQI forecasting model ({unique_id})"
)
model_meta.save(MODEL_PATH)
print(f"🚀 Model uploaded to Hopsworks Model Registry!")

print("\n🏁 TensorFlow training completed successfully!")

