import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────
# Logging Setup
# ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─────────────────────────────
# Streamlit Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="🌤️ AQI Forecast Dashboard",
    page_icon="💨",
    layout="wide"
)

st.title("🌤️ Air Quality Index (AQI) Dashboard")
st.write("Get the latest predicted AQI and explore interactive insights through EDA and model analysis.")

# ─────────────────────────────
# Environment Variables
# ─────────────────────────────
load_dotenv()
API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    st.error("❌ Missing Hopsworks API key! Please set it in your environment.")
    st.stop()

# ─────────────────────────────
# Hopsworks Connection
# ─────────────────────────────
try:
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    st.success("✅ Connected to Hopsworks!")
except Exception as e:
    st.error(f"Connection failed: {e}")
    st.stop()

# ─────────────────────────────
# Load Feature Data
# ─────────────────────────────
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
except Exception as e:
    st.error(f"Error loading feature data: {e}")
    st.stop()

# ─────────────────────────────
# Feature Columns (match training)
# ─────────────────────────────
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

time_col = "timestamp_utc"
aqi_col = "aqi_aqicn"

# Ensure required columns exist
for col in feature_cols + [time_col, aqi_col]:
    if col not in df.columns:
        df[col] = 0 if col in feature_cols else pd.NaT

# Sort and clean
df = df.sort_values(time_col).reset_index(drop=True)

# Fill missing feature values
df[feature_cols] = df[feature_cols].ffill().bfill()

# ─────────────────────────────
# Load Model and Scaler
# ─────────────────────────────
all_models = mr.get_models("rf_aqi_model")
latest_model = max(all_models, key=lambda m: m.version)
model_dir = latest_model.download()

model_path = glob.glob(os.path.join(model_dir, "**/model.pkl"), recursive=True)[0]
scaler_path = glob.glob(os.path.join(model_dir, "**/scaler.pkl"), recursive=True)[0]

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
st.success(f"📦 Model version {latest_model.version} loaded successfully.")

# ─────────────────────────────
# Predict Latest AQI
# ─────────────────────────────
latest = df.tail(1)
X = latest[feature_cols].astype("float64")
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)[0]
today = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.subheader("🌤️ Today's AQI Prediction")
st.metric(label="Predicted AQI", value=f"{pred:.2f}")
st.write(f"📅 Timestamp: {today}")

# AQI category logic
if pred <= 50:
    status, color = "Good", "green"
elif pred <= 100:
    status, color = "Moderate", "gold"
elif pred <= 150:
    status, color = "Unhealthy (Sensitive)", "orange"
elif pred <= 200:
    status, color = "Unhealthy", "red"
else:
    status, color = "Very Unhealthy", "purple"

st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)

with st.expander("🌡️ Latest Input Features"):
    st.dataframe(latest[feature_cols].T)

# ─────────────────────────────
# Exploratory Data Analysis (EDA)
# ─────────────────────────────
st.markdown("---")
st.header("📊 Exploratory Data Analysis (EDA)")

eda_df = df.tail(150)

col1, col2 = st.columns(2)

# AQI Trend Over Time
with col1:
    st.subheader("📆 AQI Trend Over Time")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eda_df[time_col], eda_df[aqi_col], color="steelblue", marker='o')
    ax.set_title("AQI Over Time")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("AQI")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation Heatmap
with col2:
    st.subheader("🔥 Correlation Heatmap")
    corr = eda_df[feature_cols + [aqi_col]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

# AQI vs Weather Parameters
st.subheader("🌡️ AQI vs Weather Parameters")
cols = st.columns(3)
for i, feature in enumerate(["ow_temp", "ow_humidity", "ow_wind_speed"]):
    with cols[i]:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=eda_df, x=feature, y=aqi_col, color="dodgerblue", alpha=0.7, ax=ax)
        ax.set_title(f"AQI vs {feature}")
        st.pyplot(fig)

# Feature Importance
st.subheader("🌲 Feature Importance (Model Insight)")
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance in AQI Prediction")
    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")

# Actual vs Predicted AQI Trend
st.subheader("📈 Actual vs Predicted AQI Trend")
try:
    compare_df = df.tail(60)
    X_compare = compare_df[feature_cols].astype("float64")
    compare_df["Predicted_AQI"] = model.predict(scaler.transform(X_compare))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(compare_df[time_col], compare_df[aqi_col], label="Actual AQI", color="skyblue", linewidth=2, marker="o")
    ax.plot(compare_df[time_col], compare_df["Predicted_AQI"], label="Predicted AQI", color="orange", linewidth=2, marker="x")
    ax.set_title("Actual vs Predicted AQI Over Time")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    with st.expander("📋 Recent Actual vs Predicted Data"):
        st.dataframe(compare_df[[time_col, aqi_col, "Predicted_AQI"]].tail(10))
except Exception as e:
    st.error(f"Error generating Actual vs Predicted plot: {e}")
