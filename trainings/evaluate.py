import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# -----------------------------
# CONFIGURATION
# -----------------------------
TARGET_COL = "aqi_aqicn"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_DIR = "models"

# Choose which model to evaluate: "sklearn" or "tensorflow"
MODEL_TYPE = "sklearn"  # change to "tensorflow" for TF model

# -----------------------------
# CONNECT TO HOPSWORKS
# -----------------------------
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# -----------------------------
# LOAD DATA FROM FEATURE GROUP
# -----------------------------
try:
    feature_group = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
except Exception as e:
    raise ValueError(f"🚨 Feature group '{FEATURE_GROUP_NAME}' not found ({e})")

df = feature_group.read()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df = df.dropna(subset=[TARGET_COL])

if TARGET_COL not in df.columns:
    raise ValueError(f"🚨 Target column '{TARGET_COL}' not found in dataset.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("✅ Data loaded for evaluation!")
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# -----------------------------
# LOAD SCALER AND TRANSFORM DATA
# -----------------------------
if MODEL_TYPE == "sklearn":
    SCALER_FILE = os.path.join(MODEL_DIR, "rf_scaler.joblib")
elif MODEL_TYPE == "tensorflow":
    SCALER_FILE = os.path.join(MODEL_DIR, "tf_scaler.joblib")
else:
    raise ValueError("MODEL_TYPE must be 'sklearn' or 'tensorflow'")

if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")

scaler = joblib.load(SCALER_FILE)
X_scaled = scaler.transform(X)

# -----------------------------
# LOAD MODEL AND PREDICT
# -----------------------------
if MODEL_TYPE == "sklearn":
    MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Sklearn model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    y_pred = model.predict(X_scaled)
elif MODEL_TYPE == "tensorflow":
    MODEL_FILE = os.path.join(MODEL_DIR, "tf_model.keras")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"TensorFlow model file not found: {MODEL_FILE}")
    model = tf.keras.models.load_model(MODEL_FILE)
    y_pred = model.predict(X_scaled).flatten()

# -----------------------------
# EVALUATE MODEL
# -----------------------------
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n📊 Evaluation Results for {MODEL_TYPE} model:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

print("\n🏁 Evaluation completed successfully!")
