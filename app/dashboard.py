"""
🌸 AQI Forecast Dashboard — Random Forest + Hopsworks Model Registry
-------------------------------------------------------------------
✅ Today AQI Prediction + 3-Day Recursive Forecast
✅ Latest Feature Data from Hopsworks Feature Store
✅ Actual vs Predicted Performance Chart
✅ Trend, Heatmap, Summary Stats
✅ No Local File Dependency
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 🌸 Streamlit Styling
# -----------------------------
st.set_page_config(page_title="AQI Forecast Dashboard 🌤️", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e3f2fd, #fce4ec, #ede7f6);
    color: #333;
    font-family: "Poppins", sans-serif;
}
.main-title {
    text-align: center;
    font-size: 2.2rem;
    color: #6a1b9a;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.3rem;
    color: #1565c0;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}
.stButton>button {
    background-color: #ba68c8;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
}
.stButton>button:hover {
    background-color: #ab47bc;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🌤️ AQI Forecast Dashboard")
st.caption("Air Quality Prediction with Explainable Insights")

# -----------------------------
# 🔐 Load Hopsworks Environment
# -----------------------------
load_dotenv()
API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    st.error("API key missing. Set AQI_FORECAST_API_KEY or HOPSWORKS_API_KEY")
    st.stop()

# -----------------------------
# 🔌 Connect to Hopsworks
# -----------------------------
try:
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    st.success("✅ Connected to Hopsworks")
except Exception as e:
    st.error(f"Hopsworks Login Failed: {e}")
    st.stop()

# -----------------------------
# 📌 Load Features from Feature Store
# -----------------------------
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    st.success("✅ Feature data loaded")
except Exception as e:
    st.error(f"Feature group error: {e}")
    st.stop()

# Data cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df.drop(columns=["timestamp_utc"], errors="ignore", inplace=True)
df.dropna(subset=[TARGET] + FEATURE_COLS, inplace=True)

# -----------------------------
# 🤖 Load RF Model and Scaler (recursive + latest version)
# -----------------------------
mr = project.get_model_registry()

try:
    # Get all versions and select latest dynamically
    all_models = mr.get_models("rf_aqi_model")
    latest_model = max(all_models, key=lambda m: m.version)
    model_dir = latest_model.download()

    # Recursively locate .pkl files
    model_path, scaler_path = None, None
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f == "model.pkl":
                model_path = os.path.join(root, f)
            elif f == "scaler.pkl":
                scaler_path = os.path.join(root, f)

    if not model_path or not scaler_path:
        raise FileNotFoundError(f"Could not locate model.pkl or scaler.pkl inside: {model_dir}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.success(f"✅ Random Forest model (version {latest_model.version}) loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -----------------------------
# 🎯 AQI Category Helper
# -----------------------------
def aqi_category(aqi):
    if aqi <= 50: return "🟢 Good"
    if aqi <= 100: return "🟡 Moderate"
    if aqi <= 150: return "🟠 USG"
    if aqi <= 200: return "🔴 Unhealthy"
    if aqi <= 300: return "🟣 Very Unhealthy"
    return "⚫ Hazardous"

# -----------------------------
# 🔮 Today + Recursive 3-Day Forecast
# -----------------------------
last_row = df.iloc[-1].copy()
current_features = last_row[FEATURE_COLS].copy()
base_date = datetime.now()
future_preds = []

today_pred = model.predict(scaler.transform([current_features]))[0]

for i in range(1, 4):
    future_date = base_date + timedelta(days=i)
    current_features["hour"] = future_date.hour
    current_features["day"] = future_date.day
    current_features["month"] = future_date.month
    current_features["weekday"] = future_date.weekday()

    pred = model.predict(scaler.transform([current_features]))[0]
    future_preds.append({
        "Date": future_date.strftime("%Y-%m-%d"),
        "Predicted AQI": round(pred, 2),
        "Category": aqi_category(pred)
    })

    # Update PM values heuristically for next recursive step
    current_features["ow_pm2_5"] = pred * 0.4
    current_features["ow_pm10"] = pred * 0.6

forecast_df = pd.DataFrame(future_preds)

# -----------------------------
# 🌞 Display Forecast
# -----------------------------
st.subheader("🌞 Today's AQI Prediction")
st.metric(f"AQI — {aqi_category(today_pred)}", f"{today_pred:.2f}")

st.subheader("📅 Forecast for Next 3 Days")
st.dataframe(forecast_df, use_container_width=True)

# -----------------------------
# 📈 Actual vs Predicted Performance
# -----------------------------
df["predicted"] = model.predict(scaler.transform(df[FEATURE_COLS]))
mse = mean_squared_error(df[TARGET], df["predicted"])
r2 = r2_score(df[TARGET], df["predicted"])
st.write(f"📌 Model Performance — R²: `{r2:.2f}` | MSE: `{mse:.2f}`")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 class='sub-header'>📊 Actual vs Predicted AQI</h3>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df[TARGET].tail(100).values, label="Actual", linewidth=2)
    ax1.plot(df["predicted"].tail(100).values, label="Predicted", linestyle="--")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.markdown("<h3 class='sub-header'>🔗 AQI Feature Correlation</h3>", unsafe_allow_html=True)
    corr = df[FEATURE_COLS + [TARGET]].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# 📋 Summary Statistics
# -----------------------------
st.markdown("<h3 class='sub-header'>📋 Summary Statistics</h3>", unsafe_allow_html=True)
st.dataframe(df[FEATURE_COLS + [TARGET]].describe().T.round(2), use_container_width=True)

# -----------------------------
# 🔁 Refresh Button
# -----------------------------
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.experimental_rerun()
