PowerPulse: Household Energy Usage Forecast - Project Documentation

Project Title

PowerPulse: Household Energy Usage Forecast

Domain

Energy Analytics / Time Series Forecasting

Problem Statement

To analyze and forecast household energy usage using historical power consumption data. The goal is to understand usage patterns, identify trends, and build a predictive model that estimates future power consumption.

Objective

Perform exploratory data analysis (EDA) to uncover insights from the energy consumption dataset.

Engineer relevant features for better forecasting.

Build a memory-efficient, accurate regression model using Streamlit.

Visualize predictions and EDA in an interactive dashboard.

Tools and Technologies Used

Programming Language: Python

Data Analysis Libraries: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Machine Learning: Scikit-learn (RandomForestRegressor)

App Development: Streamlit

Others: Jupyter Notebook, VS Code

Dataset

Source: Household power consumption dataset

Attributes:

Datetime

Global_active_power (kW)

Sub_metering_1, Sub_metering_2, Sub_metering_3

Methodology

1. Data Understanding & Cleaning

Loaded CSV data with Datetime as index.

Handled missing values using dropna().

2. Feature Engineering

Created lag features (lag_1, lag_24h).

Derived other_consumption feature from difference between total and sub-metering.

3. Data Splitting

Time-based split into 50% train and 50% test data.

4. Model Training

Used RandomForestRegressor with reduced estimators and n_jobs=1 to avoid memory errors.

5. Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R^2 Score

6. Visualization

EDA visuals: Histogram, Boxplot, Heatmap.

Prediction plot: Actual vs Predicted using Plotly.

Streamlit Dashboard Features

Sidebar: Date range filter

Tab 1: Exploratory Data Analysis

Histogram of Global Active Power

Boxplots for Sub Metering

Correlation Heatmap

Tab 2: Prediction

On-the-fly regression modeling

Performance metrics

Plot of predictions vs actual values

Model Performance

Streamlit Optimized Random Forest:

RMSE: ~0.042

MAE: ~0.006

R^2: ~0.998

Offline Notebook Model:

RMSE: ~0.006

MAE: ~0.003

R^2: ~1.000

Challenges Faced

MemoryError during model training due to large dataset.

Mitigated using:

Downsampling (50% train size)

Reducing n_estimators

Limiting n_jobs

Future Enhancements

Use LightGBM or XGBoost for faster training.

Incorporate weather and external data sources.

Forecast beyond next time step using multistep forecasting.

Deploy on cloud platform for public access.

Conclusion

The PowerPulse project successfully demonstrates how time-series energy consumption data can be analyzed and modeled in a memory-efficient way, with a user-friendly dashboard for insight and prediction.

Built with ❤️ using Python, Streamlit, and Scikit-learn
