
---

# SOLigence: AI-Driven Cryptocurrency Price Prediction App

## Overview

**SOLigence** is an AI-powered cryptocurrency price prediction application designed to empower traders and investors with actionable insights in the volatile crypto market. The app leverages advanced time-series forecasting models—including ARIMA, LSTM, XGBoost, and Prophet—to capture both short-term fluctuations and long-term trends for major cryptocurrencies. The platform features an intuitive Streamlit-based interface, real-time predictions, interactive visualizations, and automated buy/sell signals, making it accessible for both beginners and experienced users.

---

## Objectives

- **Provide AI-driven price predictions** to help users make informed trading decisions.
- **Combine multiple forecasting models** (ARIMA, LSTM, XGBoost, Prophet) for robust predictions.
- **Simplify crypto trading** for beginners with clear buy/sell signals and profit calculations.
- **Deliver an intuitive, user-friendly interface** for real-time interaction and visualization.

---

## Methodology

### Data Collection & Preparation

- **Source:** Daily closing prices for 30 cryptocurrencies (2024–2025) via the `yfinance` API.
- **Cleaning:** Removal of unnecessary columns (`open`, `volume`, `high`, `low`), standardization, and handling of missing values.
- **Dimensionality Reduction:** Principal Component Analysis (PCA) to capture key patterns while reducing complexity.

### Clustering

- **Algorithms Tested:** Mean-Shift, DBSCAN, K-Means.
- **Selected Approach:** K-Means (n=4) for optimal clustering.
- **Target Coins:** BTC-USD, BCH-USD, STETH-USD, LEO-USD (one from each cluster).

### Model Selection

- **LSTM:** Captures long-term dependencies in sequential data.
- **ARIMA:** Effective for short-term, stationary time series.
- **XGBoost:** Efficient and accurate for structured data.
- **Prophet:** Robust for seasonality and trend detection.

### Model Evaluation

- **Metrics:** MAE, RMSE, MAPE, R² Score.
- **Visualization:** Interactive forecast plots, correlation matrices, and cluster visualizations.

---

## Application Features

- **Streamlit Web Interface:** Modern, interactive dashboard with branding and sidebar controls.
- **Model Selection:** Users can choose the coin and forecasting model, and adjust model-specific parameters.
- **Performance Metrics:** Real-time display of MAE, RMSE, MAPE, and R² for each model.
- **Buy/Sell Signals:** Automated trading recommendations based on model outputs.
- **Profit Calculator:** Estimate potential returns from hypothetical trades.
- **Help & Documentation:** Built-in help tab with usage instructions and model descriptions.

---

## Usage

1. **Clone the repository and navigate to the project directory.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure `crypto_data.csv` (or your data source) is in the project directory.**
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
5. **Interact with the dashboard:** Select coins, models, and parameters; view predictions and trading signals.

---

## Technologies Used

- **Python 3**
- **Libraries:** pandas, numpy, scikit-learn, yfinance, plotly, matplotlib, statsmodels, tensorflow/keras, xgboost, streamlit

---

## Project Structure

- `crypto.py` — Data collection and preprocessing
- `models.py` — Model definitions and training
- `forecast_plots.py` — Visualization utilities
- `profit_calculator.py` — Profit estimation tools
- `buy_sell_signals.py` — Automated trading signals
- `app.py` — Streamlit web application

---

## Limitations & Future Work

- **Model Limitations:** Each model has strengths and weaknesses for different market conditions.
- **Data Constraints:** Predictions are only as good as the input data.
- **Future Enhancements:**
  - Integrate live data APIs for real-time prediction.
  - Add sentiment analysis from social media (Twitter, Reddit).
  - Expand to more cryptocurrencies and advanced trading strategies.

---

## References

- Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- Tripathy et al. (2025), "Deep Learning for Cryptocurrency Prediction"
- Alshara (2022), "Prophet vs. LSTM for Financial Forecasting"
- Sepp Hochreiter & Jürgen Schmidhuber (1997), "Long Short-Term Memory"

---



