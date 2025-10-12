# dashboard.py
import os
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import hopsworks
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --------------------------
# INITIAL CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
)

# Dark appearance for matplotlib/seaborn
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

# Small CSS tweaks for a modern card look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0c0f;
        color: #f5f7fb;
    }
    .neon-card {
        background: linear-gradient(90deg, rgba(22,160,133,0.14), rgba(72,61,139,0.09));
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    .big-title {
        font-size: 28px;
        font-weight:700;
        color: #ffffff;
    }
    .muted {
        color: #bfc7d6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()

# --------------------------
# FILE & HOPSWORKS CONFIG
# --------------------------
FORECAST_CSV = os.path.join("data", "forecast_next_3_days.csv")
FEATURE_GROUP_NAME = "aqi_features"

HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

# --------------------------
# LOAD FORECAST CSV (authoritative)
# --------------------------
if not os.path.exists(FORECAST_CSV):
    st.error("❌ Forecast file not found. Run predict.py first to generate data/forecast_next_3_days.csv")
    st.stop()

forecast_df = pd.read_csv(FORECAST_CSV)
# robust parsing / coercion
forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"], errors="coerce")
forecast_df["predicted_aqi"] = pd.to_numeric(forecast_df["predicted_aqi"], errors="coerce")
if forecast_df["predicted_aqi"].isna().all():
    st.error("❌ Forecast file contains invalid numeric AQI values.")
    st.stop()

# --------------------------
# CONNECT TO HOPSWORKS (feature data + optional model info)
# --------------------------
fs = None
fg_df = None
model_path = None
model_type = None
model_basename = None
try:
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("Missing Hopsworks API key (env).")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    # try to read feature group for extra insights (not required)
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
        fg_df = fg.read()
        fg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        fg_df.ffill(inplace=True)
        fg_df.bfill(inplace=True)
        fg_df = fg_df.sort_values("timestamp_utc").reset_index(drop=True)
    except Exception:
        fg_df = None
except Exception as e:
    st.sidebar.warning(f"⚠️ Hopsworks not available: {e}")

# Try to find a model metadata in registry (optional)
model_registry_info = None
if fs is not None:
    try:
        mr = project.get_model_registry()
        # prefer rf_aqi_model then tf_lstm_aqi_model etc
        candidates = ["rf_aqi_model", "rf_model", "tf_lstm_aqi_model", "tf_aqi_model"]
        for name in candidates:
            try:
                models = mr.get_models(name=name)
                if models:
                    best = max(models, key=lambda m: getattr(m, "version", 0))
                    # download to local tmp folder and inspect
                    if hasattr(best, "download"):
                        model_dir = best.download()
                        # find an artifact
                        for p in ["rf_model.joblib", "model.pkl", "tf_lstm_model.keras", "tf_model.keras", "model.h5"]:
                            candidate = os.path.join(model_dir, p)
                            if os.path.exists(candidate):
                                model_path = candidate
                                model_basename = os.path.basename(candidate)
                                model_type = "sklearn" if candidate.endswith((".joblib", ".pkl")) else "tensorflow"
                                break
                        if model_path:
                            model_registry_info = (best.name, getattr(best, "version", None))
                            break
            except Exception:
                continue
    except Exception:
        pass

# If no registry model found, try local folder
if model_path is None:
    # try typical local model names
    local_candidates = [
        "models/rf_model.joblib", "models/model.pkl", "models/tf_lstm_model.keras", "models/tf_model.keras", "models/tf_model.h5"
    ]
    for c in local_candidates:
        if os.path.exists(c):
            model_path = c
            model_basename = os.path.basename(c)
            model_type = "sklearn" if c.endswith((".joblib", ".pkl")) else "tensorflow"
            break

# --------------------------
# UI - Header
# --------------------------
st.sidebar.header("Controls")
st.sidebar.write(f"Model: {model_basename or '– (not found)'}")
st.sidebar.write(f"Source: forecast_next_3_days.csv")
st.sidebar.write(f"Forecast days: {len(forecast_df)}")
if st.sidebar.button("🔄 Refresh"):
    st.experimental_rerun()

st.markdown("<div class='big-title'>AQI Forecast Dashboard — Karachi</div>", unsafe_allow_html=True)
st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.markdown("---")

# --------------------------
# Small helper: AQI category
# --------------------------
def get_aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "#2ecc71", "Air quality is satisfactory and poses little or no risk."
    if aqi <= 100:
        return "Moderate", "#f1c40f", "Air quality is acceptable; sensitive groups may be affected."
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "People with respiratory conditions should limit outdoor activity."
    if aqi <= 200:
        return "Unhealthy", "#e74c3c", "Everyone may begin to experience health effects."
    if aqi <= 300:
        return "Very Unhealthy", "#9b59b6", "Health alert: more serious effects likely."
    return "Hazardous", "#7f0000", "Emergency conditions: avoid outdoor exposure."

# --------------------------
# TOP CARD: tomorrow's forecast
# --------------------------
# choose the first row in forecast_df as "tomorrow" (predict.py writes days in order)
tomorrow_row = forecast_df.iloc[0]
tomorrow_aqi = float(tomorrow_row["predicted_aqi"])
category, color, advisory = get_aqi_category(tomorrow_aqi)

st.markdown(
    f"<div class='neon-card' style='border-left:6px solid {color};'>"
    f"<h2 style='margin:0; color: #ffffff;'>Forecast (tomorrow): {tomorrow_aqi:.1f} — {category}</h2>"
    f"<p class='muted' style='margin:6px 0 0 0'>{advisory}</p>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# --------------------------
# Forecast plot figure
# --------------------------
st.markdown("### 📈 3-Day AQI Forecast")
fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0b0c0f")
ax.set_facecolor("#0b0c0f")
dates = forecast_df["forecast_date"]
vals = forecast_df["predicted_aqi"].astype(float)

ax.plot(dates, vals, marker="o", linewidth=2.5, color="#00e5ff")  # cyan line
ax.fill_between(dates, vals - 3, vals + 3, alpha=0.08, color="#00e5ff")
ax.set_xlabel("Date", color="#bfc7d6")
ax.set_ylabel("Predicted AQI", color="#bfc7d6")
ax.set_title("3-Day AQI Forecast Trend", color="#ffffff", fontsize=16, pad=12)
ax.tick_params(colors="#bfc7d6")
ax.grid(True, linestyle="--", linewidth=0.5, color="#2f3640", alpha=0.8)
for spine in ax.spines.values():
    spine.set_color("#2f3640")

st.pyplot(fig)

# --------------------------
# Forecast table
# --------------------------
st.markdown("### 📋 Forecast Table")
forecast_df_disp = forecast_df.copy()
forecast_df_disp["forecast_date"] = pd.to_datetime(forecast_df_disp["forecast_date"]).dt.strftime("%Y-%m-%d")
st.dataframe(forecast_df_disp.style.format({"predicted_aqi": "{:.2f}"}), height=200)

# --------------------------
# Additional Insights from feature group (if available)
# --------------------------
if fg_df is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Summary Statistics (features)")
        # show descriptive stats
        st.dataframe(fg_df.describe().transpose().style.background_gradient(cmap="viridis"), height=260)

    with col2:
        st.markdown("### 🔥 Feature Correlation Heatmap")
        numeric = fg_df.select_dtypes(include=np.number)
        corr = numeric.corr()

        fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor="#0b0c0f")
        ax2.set_facecolor("#0b0c0f")
        sns.heatmap(corr, annot=True, cmap="RdYlBu", linewidths=0.3, ax=ax2)
        ax2.set_title("Feature correlation", color="#ffffff")
        st.pyplot(fig2)

# --------------------------
# SHAP explainability (best-effort)
# --------------------------
st.markdown("---")
st.markdown("### 🧠 SHAP Feature Importance (Explainability) — best-effort")
shap_warning = None
try:
    # Try to load the model if not already loaded from registry
    if model_path is None:
        # look locally for sklearn model first
        if os.path.exists("models/rf_model.joblib"):
            model_path = "models/rf_model.joblib"
            model_type = "sklearn"
        elif os.path.exists("models/rf_model.pkl"):
            model_path = "models/rf_model.pkl"
            model_type = "sklearn"
        elif os.path.exists("models/tf_lstm_model.keras"):
            model_path = "models/tf_lstm_model.keras"
            model_type = "tensorflow"

    loaded_model = None
    if model_path:
        if model_type == "sklearn":
            loaded_model = joblib.load(model_path)
        else:
            import tensorflow as tf  # local import
            loaded_model = tf.keras.models.load_model(model_path)

    # Build a sample X for explanation: take last row from feature group if available,
    # otherwise try to reconstruct from forecast file (best-effort)
    if fg_df is not None:
        expl_X = fg_df.sort_values("timestamp_utc").iloc[-1:][[c for c in fg_df.columns if pd.api.types.is_numeric_dtype(fg_df[c])]]
        # choose only columns that the model expects if we can detect them via scaler or train columns.
        # We'll pass the whole numeric set and let SHAP deal with it (best-effort).
    else:
        expl_X = None

    if loaded_model is None:
        st.info("Model not available locally or in registry — cannot produce SHAP plot.")
    else:
        # Use shap.Explainer which handles many model types automatically
        if expl_X is None:
            st.info("No feature-group data available to compute SHAP (need feature inputs).")
        else:
            # convert to DataFrame
            X_for_shap = expl_X.select_dtypes(include=np.number)
            # try fast TreeExplainer for tree models:
            try:
                if hasattr(shap, "TreeExplainer") and model_type == "sklearn":
                    expl = shap.TreeExplainer(loaded_model)
                    shap_values = expl.shap_values(X_for_shap)
                else:
                    expl = shap.Explainer(loaded_model, X_for_shap)
                    shap_values = expl(X_for_shap)
                # create summary bar plot and capture figure
                plt.figure(facecolor="#0b0c0f")
                shap.summary_plot(shap_values, X_for_shap, plot_type="bar", show=False)
                fig_shap = plt.gcf()
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"⚠️ Could not compute SHAP (fast path): {e}")
                # try fallback: approximate permutation importance plot using shap.KernelExplainer would be very slow -> skip
                st.info("SHAP failed — this can happen for TF LSTM models or when feature names mismatch. Try re-training/exporting a simpler model (random forest) for SHAP.")
except Exception as e:
    st.warning(f"⚠️ SHAP section failed: {e}")

st.markdown("---")
st.caption("Built with ❤️ — Dark modern theme. Predictions are read from `data/forecast_next_3_days.csv`.")

