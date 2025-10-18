AQI Forecast Dashboard — Air Quality Forecasting using Machine Learning

An end-to-end Air Quality Index (AQI) Forecasting System that predicts air quality for the next 3 days based on real-time environmental and pollutant data. The project integrates Feature Store (Hopsworks) for data management, TensorFlow LSTM for time-series forecasting, Streamlit for visualization, and GitHub Actions for CI/CD automation.

🚀 Project Overview

This project was developed as part of a Data Science Internship to design and deploy a professional-grade AQI forecasting dashboard.
It combines data pipelines, ML model training, explainability, and real-time predictions in one cohesive system.

The goal is to predict and visualize the air quality in major cities (e.g., Karachi) and provide insights into pollution patterns using machine learning explainability tools like SHAP.

✨ Key Features
🧩 Feature Pipeline Development

Fetches raw weather and pollutant data from APIs such as:

OpenWeather API

AQICN Air Quality API

Computes derived features including:

Time-based features: hour, day, month, weekday

Environmental parameters: temperature, humidity, pressure, etc.

Derived metrics like AQI rate of change

Stores cleaned and processed features into Hopsworks Feature Store for consistent training and inference data.

🕰️ Historical Data Backfill

Backfills data to create historical training datasets.

Generates a comprehensive feature-target dataset for model development and evaluation.

Ensures consistency between real-time and historical data pipelines.

🤖 Training Pipeline Implementation

Fetches historical data from Hopsworks Feature Store.

Trains multiple models:

Random Forest

Ridge Regression

LSTM (TensorFlow)

Evaluates performance using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (Coefficient of Determination)

Logs trained models and their metrics into the Model Registry for version control.

⚙️ Automated CI/CD Pipeline

Uses GitHub Actions to automate:

Feature pipeline (runs hourly)

Model retraining (runs daily)

Supports easy deployment and model updates without manual intervention.

Ensures robust MLOps-style workflow integration.

🖥️ Web Application Dashboard

Built using Streamlit for an interactive user experience.

Fetches latest data and models from Hopsworks.

Computes real-time AQI forecasts for the next 3 days.

Displays:

Forecast values

Recent AQI trends

Correlation heatmaps

Summary statistics

📊 Advanced Analytics & Explainability

Performs Exploratory Data Analysis (EDA) to identify temporal and pollutant-based trends.

Implements SHAP Explainability (with safe fallback to permutation-based feature importance for LSTM models).

Displays top influencing features impacting AQI predictions.

Includes color-coded alerts and interpretations for hazardous AQI levels.

🧠 Tech Stack
Category	Technologies Used
Language	Python 3.10
Frameworks	TensorFlow, Scikit-learn
Visualization	Streamlit, Matplotlib, Seaborn, SHAP
Data Management	Hopsworks Feature Store
Automation (CI/CD)	GitHub Actions
APIs	OpenWeather, AQICN
Model Explainability	SHAP (with fallback)
Deployment	Streamlit Cloud