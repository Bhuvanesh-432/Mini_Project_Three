import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_power_data.csv", parse_dates=["Datetime"])
    df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
    df.set_index('Datetime', inplace=True)
    df = df.dropna()
    return df

df = load_data()

# --- Title ---
st.title("⚡ Power Consumption Dashboard with Prediction")

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")
date_range = st.sidebar.date_input("Select Date Range", [df.index.min().date(), df.index.max().date()])
if len(date_range) == 2:
    df = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])]

# --- Tabs ---
tabs = st.tabs(["📊 EDA", "🔮 Prediction"])

# --- 📊 EDA ---
with tabs[0]:
    st.subheader("Exploratory Data Analysis")

    # Histogram
    st.markdown("### Distribution of Global Active Power")
    fig, ax = plt.subplots()
    df['Global_active_power'].hist(bins=50, ax=ax)
    ax.set_xlabel('Kilowatts')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Boxplot
    st.markdown("### Boxplot of Sub Metering")
    fig, ax = plt.subplots()
    sns.boxplot(data=df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']], ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- 🔮 Prediction ---
with tabs[1]:
    st.subheader("Predict Global Active Power")

    # Feature Engineering
    df_model = df.copy()
    df_model['lag_1'] = df_model['Global_active_power'].shift(1)
    df_model['lag_24h'] = df_model['Global_active_power'].shift(1440)
    df_model['other_consumption'] = df_model['Global_active_power'] - (
        df_model['Sub_metering_1'] + df_model['Sub_metering_2'] + df_model['Sub_metering_3']) / 1000.0
    df_model.dropna(inplace=True)

    # Features & Target
    X = df_model[['lag_1', 'lag_24h', 'other_consumption']]
    y = df_model['Global_active_power']

    # Train-Test Split (80/20)
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model Training (same as main notebook)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    # Evaluation Metrics
    st.markdown("### Model Performance")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Fixed for older scikit-learn
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**RMSE:** {rmse:.3f}")
    st.write(f"**MAE :** {mae:.3f}")
    st.write(f"**R²   :** {r2:.3f}")


    # Plot Actual vs Predicted (full or toggleable)
    st.markdown("### Actual vs Predicted Global Active Power")
    plot_limit = st.slider("How many samples to plot?", min_value=100, max_value=len(y_test), value=500)
    plot_df = pd.DataFrame({"Actual": y_test.values[:plot_limit], "Predicted": y_pred[:plot_limit]})
    fig = px.line(plot_df, labels={'index': 'Time Step', 'value': 'Power (kW)'})
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.caption("Built with ❤️ using Streamlit")

