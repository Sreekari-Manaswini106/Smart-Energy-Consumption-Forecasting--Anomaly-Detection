import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import timedelta
import matplotlib.pyplot as plt
import os

# --- NLP & LLM Imports ---
from nlp_module import NLPEnergyAssistant
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI

# ------------------------------------------
# ⚡ UI HEADER
# ------------------------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")
st.title("⚡ Smart Energy Consumption Forecasting & Anomaly Detection")
st.markdown("### Developed by: **Sreekari Manaswini**")
st.markdown("---")

# ------------------------------------------
# NLP Assistant setup
# ------------------------------------------
nlp_assistant = NLPEnergyAssistant()

def ask_llm(query):
    prompt = PromptTemplate.from_template("Answer this energy-related question: {question}")
    llm = OpenAI(temperature=0.2)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query)

# ------------------------------------------
# ✅ Load Local Data (Optimized)
# ------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    file_path = "dataset/Household_power_consumption.txt"
    df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                     infer_datetime_format=True, na_values=['?'], low_memory=False)
    df.dropna(inplace=True)
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df.set_index('datetime', inplace=True)
    # Reduce load: only first month for initial load
    df = df["2007-01-01":"2007-01-31"]
    return df

# ------------------------------------------
# Feature Engineering
# ------------------------------------------
def add_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    return data

# ------------------------------------------
# Evaluation Metrics
# ------------------------------------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def display_metrics(mae, mse, rmse, r2):
    metrics_data = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
        'Value': [f'{mae:.2f}', f'{mse:.2f}', f'{rmse:.2f}', f'{r2:.2f}']
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)

# ------------------------------------------
# Forecast: ARIMA
# ------------------------------------------
def arima_forecast(data):
    if st.button("Run ARIMA Forecast"):
        st.subheader("📉 ARIMA Forecast (Next 30 Days)")
        daily = data['Global_active_power'].resample('D').mean().dropna()
        model = ARIMA(daily, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(30)
        combined = pd.concat([daily[-60:], forecast])
        st.line_chart(combined)
        mae, mse, rmse, r2 = evaluate_model(daily[-30:], forecast)
        display_metrics(mae, mse, rmse, r2)

# ------------------------------------------
# Forecast: LSTM
# ------------------------------------------
def lstm_forecast(data):
    if st.button("Run LSTM Forecast"):
        st.subheader("🔮 LSTM Forecast (Next 30 Days)")
        df = data['Global_active_power'].resample('D').mean().dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.reshape(-1, 1))

        def create_seq(data, n_steps):
            X, y = [], []
            for i in range(n_steps, len(data)):
                X.append(data[i - n_steps:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X, y = create_seq(scaled, 10)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, verbose=0)

        input_seq = scaled[-10:]
        predictions = []
        for _ in range(30):
            pred = model.predict(input_seq.reshape(1, 10, 1), verbose=0)
            predictions.append(pred[0][0])
            input_seq = np.append(input_seq[1:], pred).reshape(10, 1)

        future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
        future_series = pd.Series(future.flatten(), index=future_dates)

        st.line_chart(pd.concat([df[-60:], future_series]))
        mae, mse, rmse, r2 = evaluate_model(df[-30:], future.flatten())
        display_metrics(mae, mse, rmse, r2)

# ------------------------------------------
# Anomaly Detection
# ------------------------------------------
def detect_anomalies(data):
    st.subheader("⚠️ Anomaly Detection")
    daily = data['Global_active_power'].resample('D').mean().dropna()
    z_scores = (daily - daily.mean()) / daily.std()
    anomalies = daily[z_scores.abs() > 3]
    st.write(f"Anomalies Detected: {len(anomalies)}")
    st.line_chart(daily)
    st.dataframe(anomalies)

# ------------------------------------------
# Optimization Tips
# ------------------------------------------
def power_usage_optimization(data):
    st.subheader("🔋 Power Usage Optimization Tips")
    daily = data['Global_active_power'].resample('D').mean().dropna()
    peak_day = daily.idxmax().strftime('%Y-%m-%d')
    low_day = daily.idxmin().strftime('%Y-%m-%d')
    st.success(f"📈 Peak Usage Day: {peak_day}")
    st.success(f"📉 Off-Peak Usage Day: {low_day}")

# ------------------------------------------
# Energy Consumption Dashboard
# ------------------------------------------
def energy_consumption_dashboard(data):
    st.subheader("📊 Energy Consumption Dashboard")

    min_date = data.index.min()
    max_date = data.index.max()

    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    filtered_data = data[(data.index >= str(start_date)) & (data.index <= str(end_date))]
    metric = st.selectbox("Select Metric", ['Global_active_power', 'Global_reactive_power', 'Voltage'])

    st.line_chart(filtered_data[metric])

    daily = filtered_data[metric].resample('D').mean().dropna()
    monthly = filtered_data[metric].resample('M').mean().dropna()
    yearly = filtered_data[metric].resample('Y').mean().dropna()

    view = st.radio("Aggregation", ['Daily', 'Monthly', 'Yearly'])
    if view == 'Daily':
        st.line_chart(daily)
    elif view == 'Monthly':
        st.line_chart(monthly)
    else:
        st.line_chart(yearly)

# ------------------------------------------
# Main App
# ------------------------------------------
def main():
    df = load_data()
    df = add_features(df)

    st.sidebar.title("📌 Navigation")
    option = st.sidebar.radio("Select View", [
        "View Raw Data",
        "Energy Usage Prediction (ARIMA)",
        "Energy Usage Prediction (LSTM)",
        "Anomaly Detection",
        "Power Usage Optimization",
        "Energy Consumption Dashboard",
        "Ask Energy Assistant"
    ])

    if option == "View Raw Data":
        st.write(df.head(100))
    elif option == "Energy Usage Prediction (ARIMA)":
        arima_forecast(df)
    elif option == "Energy Usage Prediction (LSTM)":
        lstm_forecast(df)
    elif option == "Anomaly Detection":
        detect_anomalies(df)
    elif option == "Power Usage Optimization":
        power_usage_optimization(df)
    elif option == "Energy Consumption Dashboard":
        energy_consumption_dashboard(df)
    elif option == "Ask Energy Assistant":
        st.subheader("📋 Energy Assistant")
        mode = st.radio("Choose Assistant Mode", ["NLP FAQ Assistant", "LLM-based Assistant (Optional)"])
        query = st.text_input("Ask your question about energy usage or forecasting:")
        if query:
            if mode == "NLP FAQ Assistant":
                answer = nlp_assistant.get_response(query)
            else:
                answer = ask_llm(query)
            st.success(f"Answer: {answer}")

if __name__ == "__main__":
    main()
