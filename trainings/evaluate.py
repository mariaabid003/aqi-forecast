import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/features/training_dataset.csv"  # or .parquet
TF_MODEL_PATH = "models/tf_model.keras"
TF_SCALER_PATH = "models/tf_scaler.joblib"
RF_MODEL_PATH = "models/rf_model.joblib"
RF_SCALER_PATH = "models/rf_scaler.joblib"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {len(df)} rows")

# Forward-fill missing values
df.ffill(inplace=True)

# ----------------------------
# Features and Target
# ----------------------------
TARGET = "aqi_aqicn"
non_numeric_cols = ["timestamp_utc"]

df_aqi = df.drop(columns=non_numeric_cols)
FEATURES = df_aqi.drop(columns=[TARGET]).columns.tolist()
X = df_aqi[FEATURES].values
y = df_aqi[TARGET].values

print("Evaluation features:", FEATURES)

# ----------------------------
# TensorFlow Model Evaluation
# ----------------------------
# Load scaler and scale features
tf_scaler = joblib.load(TF_SCALER_PATH)
X_scaled_tf = tf_scaler.transform(X)
print(f"✅ TensorFlow scaler loaded from {TF_SCALER_PATH}")

# Load model
tf_model = load_model(TF_MODEL_PATH)
print(f"✅ TensorFlow model loaded from {TF_MODEL_PATH}")

# Predict & evaluate
y_pred_tf = tf_model.predict(X_scaled_tf, verbose=0).flatten()
rmse_tf = np.sqrt(mean_squared_error(y, y_pred_tf))
mae_tf = mean_absolute_error(y, y_pred_tf)

print("\n📊 TensorFlow Model Evaluation:")
print(f"RMSE: {rmse_tf:.2f}")
print(f"MAE: {mae_tf:.2f}")

# ----------------------------
# Sklearn Random Forest Evaluation
# ----------------------------
# Load scaler and model
rf_scaler = joblib.load(RF_SCALER_PATH)
X_scaled_rf = rf_scaler.transform(X)
rf_model = joblib.load(RF_MODEL_PATH)
print(f"\n✅ Sklearn scaler loaded from {RF_SCALER_PATH}")
print(f"✅ Sklearn model loaded from {RF_MODEL_PATH}")

# Predict & evaluate
y_pred_rf = rf_model.predict(X_scaled_rf)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
mae_rf = mean_absolute_error(y, y_pred_rf)
r2_rf = r2_score(y, y_pred_rf)

print("\n📊 Random Forest Model Evaluation:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"MAE: {mae_rf:.2f}")
print(f"R²: {r2_rf:.2f}")
