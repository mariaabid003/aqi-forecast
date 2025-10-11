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

# Drop timestamp or any non-numeric columns if they exist
X = X.select_dtypes(include=[np.number])

print("✅ Data loaded for evaluation!")
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# -----------------------------
# DEFINE A FUNCTION TO EVALUATE MODELS
# -----------------------------
def evaluate_model(model_type, model_file, scaler_file):
    """Evaluate a given model and return metrics."""
    if not os.path.exists(model_file):
        print(f"🚨 {model_type} model not found at {model_file}")
        return None

    if not os.path.exists(scaler_file):
        print(f"🚨 {model_type} scaler not found at {scaler_file}")
        return None

    print(f"\n🔍 Evaluating {model_type} model...")

    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X)

    if model_type == "sklearn":
        model = joblib.load(model_file)
        y_pred = model.predict(X_scaled)
    elif model_type == "tensorflow":
        model = tf.keras.models.load_model(model_file)
        y_pred = model.predict(X_scaled).flatten()
    else:
        raise ValueError("Invalid model_type")

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {"Model": model_type, "RMSE": rmse, "MAE": mae, "R²": r2}

# -----------------------------
# EVALUATE BOTH MODELS
# -----------------------------
results = []

# Sklearn (Random Forest)
sklearn_model = os.path.join(MODEL_DIR, "rf_model.joblib")
sklearn_scaler = os.path.join(MODEL_DIR, "rf_scaler.joblib")
sklearn_metrics = evaluate_model("sklearn", sklearn_model, sklearn_scaler)
if sklearn_metrics:
    results.append(sklearn_metrics)

# TensorFlow
tf_model = os.path.join(MODEL_DIR, "tf_model.keras")
tf_scaler = os.path.join(MODEL_DIR, "tf_scaler.joblib")
tf_metrics = evaluate_model("tensorflow", tf_model, tf_scaler)
if tf_metrics:
    results.append(tf_metrics)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
if results:
    results_df = pd.DataFrame(results)
    print("\n📊 Model Evaluation Comparison:")
    print(results_df.to_string(index=False))
else:
    print("🚨 No models were successfully evaluated!")

print("\n🏁 Evaluation completed successfully!")
