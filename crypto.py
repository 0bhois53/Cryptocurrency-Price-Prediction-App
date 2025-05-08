import streamlit as st
st.set_page_config(layout="wide")

from datetime import date, datetime, timedelta
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from plotly import graph_objs as go
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import itertools
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from models import (
    train_arima,
    train_xgboost,
    train_prophet,
    calculate_metrics,
    train_neural_forecast_lstm
)
from profit_calculator import calculate_profit_table
from forecast_plots import (
    plot_prophet_forecast,
    plot_arima_forecast,
    plot_xgboost_forecast,
    plot_lstm_forecast
)
from buy_sell_signals import (
    buy_sell_signals_prophet,
    buy_sell_signals_arima,
    buy_sell_signals_xgboost,
    buy_sell_signals_lstm
)

# Custom CSS
st.markdown("""
<style>
    .news-container {
        background-color: rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .news-title {
        color: var(--text-color);
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .news-source {
        color: var(--secondary-text-color);
        font-size: 12px;
    }
    .news-date {
        color: var(--secondary-text-color);
        font-size: 12px;
    }
    .news-link {
        color: var(--link-color);
        text-decoration: none;
    }
    .news-link:hover {
        text-decoration: underline;
    }
    :root {
        --text-color: #262730;
        --secondary-text-color: #666666;
        --link-color: #0066cc;
    }
    [data-theme="dark"] {
        --text-color: #ffffff;
        --secondary-text-color: #9e9e9e;
        --link-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Create two columns for the main layout
main_col, news_col = st.columns([2, 1])

with main_col:
    st.image("vezzra.png", width=220)
    st.title('Cryptocurrency Forecast App')

    # Documentation and Help Section
    with st.expander("ℹ️ How to use this app", expanded=False):
        st.markdown("""
        **Welcome to the Cryptocurrency Price Forecast Dashboard!**

        - **Select a cryptocurrency** and a **forecasting model** from the dropdowns.
        - Adjust the **model parameters** in the sidebar to customize the forecast.
        - The app will display:
            - Historical price chart
            - Model forecast and performance metrics
            - Buy/Sell/Hold signals for the next week, 2 weeks, and 1 month
            - Latest crypto news

        **Model Descriptions:**
        - **Prophet:** Good for capturing seasonality and trends.
        - **ARIMA:** Classic statistical model for time series.
        - **XGBoost:** Machine learning model for regression.
        - **LSTM:** Deep learning model for sequential data.

        ⚠️ *Predictions are for educational purposes only. Cryptocurrency markets are highly volatile!*
        """)

    START = "2020-01-01"
    END = date.today()

    # Define crypto list
    crypto_list = ['BTC-USD', 'STETH-USD', 'LEO-USD', 'BCH-USD']
    selected_stock = st.selectbox('Select cryptocurrency for prediction', crypto_list, help="Choose the cryptocurrency you want to analyze.")

    # Add model selection
    model_list = ['Prophet', 'ARIMA', 'XGBoost', 'LSTM']
    selected_model = st.selectbox('Select forecasting model', model_list, help="Choose the forecasting model. See the help section above for descriptions.")

    n_years = st.slider('Years of prediction:', 1, 3, help="Number of years to predict into the future.")
    period = n_years * 365

    # Show the most recent price for the selected coin
    @st.cache_data
    def get_latest_price(symbol):
        data = yf.download(symbol, period='2d', interval='1d')
        if not data.empty:
            close = data['Close']
            if isinstance(close, pd.Series):
                return float(close.iloc[-1])
            elif isinstance(close, pd.DataFrame):
                return float(close.iloc[-1, 0])
        return None

    latest_price = get_latest_price(selected_stock)
    if latest_price is not None:
        st.metric(label=f"Most Recent {selected_stock} Price", value=f"${latest_price:,.2f}")
    else:
        st.info("Unable to fetch the latest price for this coin.")

    # --- Default values for all models ---
    DEFAULTS = {
        'arima_p': 2,
        'arima_d': 1,
        'arima_q': 2,
        'arima_conf': 95,
        'xgb_max_depth': 6,
        'xgb_n_estimators': 100,
        'xgb_learning_rate': 0.1,
        'lstm_forecast_horizon': 30,
        'lstm_max_steps': 750,
        'prophet_changepoint_prior_scale': 0.05,
        'prophet_seasonality_prior_scale': 10.0,
        'prophet_holidays_prior_scale': 10.0,
        'prophet_seasonality_mode': 'additive',
    }

    # --- Reset Parameters Button ---
    if st.sidebar.button('Reset Parameters'):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    # --- ARIMA Parameters ---
    if selected_model == 'ARIMA':
        st.sidebar.subheader('ARIMA Parameters')
        use_auto_arima = st.sidebar.checkbox('Use Auto-ARIMA', value=False, help="Automatically select the best ARIMA parameters (p, d, q) for your data.")
        if not use_auto_arima:
            p = st.sidebar.slider('p (AR order)', 0, 5, DEFAULTS['arima_p'], key='arima_p')
            d = st.sidebar.slider('d (Difference order)', 0, 2, DEFAULTS['arima_d'], key='arima_d')
            q = st.sidebar.slider('q (MA order)', 0, 5, DEFAULTS['arima_q'], key='arima_q')
            confidence_interval = st.sidebar.slider('Confidence Interval (%)', 80, 99, DEFAULTS['arima_conf'], key='arima_conf')
    elif selected_model in ['XGBoost', 'Stacked (XGBoost + LSTM)']:
        st.sidebar.subheader('XGBoost Parameters')
        max_depth = st.sidebar.slider('max_depth', 3, 10, DEFAULTS['xgb_max_depth'], key='xgb_max_depth')
        n_estimators = st.sidebar.slider('n_estimators', 50, 300, DEFAULTS['xgb_n_estimators'], key='xgb_n_estimators')
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.3, DEFAULTS['xgb_learning_rate'], key='xgb_learning_rate')
    elif selected_model == 'LSTM':
        st.sidebar.subheader('LSTM Parameters')
        forecast_horizon = st.sidebar.slider('Forecast Horizon (days)', 30, 365, DEFAULTS['lstm_forecast_horizon'], key='lstm_forecast_horizon', help="How many days into the future to predict. Longer horizons are less accurate.")
        max_steps = st.sidebar.slider('Max Training Steps', 100, 750, DEFAULTS['lstm_max_steps'], key='lstm_max_steps', help="Number of training steps for the LSTM model. More steps may improve accuracy but take longer.")
    elif selected_model == 'Prophet':
        st.sidebar.subheader('Prophet Parameters')
        changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale', 0.001, 0.5, DEFAULTS['prophet_changepoint_prior_scale'], key='prophet_changepoint_prior_scale')
        seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', 0.01, 10.0, DEFAULTS['prophet_seasonality_prior_scale'], key='prophet_seasonality_prior_scale')
        holidays_prior_scale = st.sidebar.slider('Holidays Prior Scale', 0.01, 10.0, DEFAULTS['prophet_holidays_prior_scale'], key='prophet_holidays_prior_scale')
        seasonality_mode = st.sidebar.selectbox('Seasonality Mode', ['additive', 'multiplicative'], index=0 if DEFAULTS['prophet_seasonality_mode']=='additive' else 1, key='prophet_seasonality_mode')

    # Function to fetch crypto news
    def fetch_crypto_news():
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = requests.get(url)
            news_data = response.json()
            return news_data.get('Data', [])
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []

    # Function to display news
    def display_news(news_items):
        for item in news_items[:10]:
            with st.container():
                st.markdown("""
                <div class="news-container">
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="news-title">{item['title']}</div>
                <div class="news-source">Source: {item['source']}</div>
                <div class="news-date">Published: {datetime.fromtimestamp(item['published_on']).strftime('%Y-%m-%d %H:%M')}</div>
                <a href="{item['url']}" target="_blank" class="news-link">Read More →</a>
                """, unsafe_allow_html=True)
                
                st.markdown("""</div>""", unsafe_allow_html=True)

    # Display news in the right column
    with news_col:
        st.subheader("📰 Cryptocurrency News")
        news_items = fetch_crypto_news()
        if news_items:
            display_news(news_items)
        else:
            st.write("Unable to fetch news at the moment.")

    # Main app functionality
    with main_col:
        @st.cache_data
        def download_data():
            try:
                st.write("Fetching data for all cryptocurrencies...")
                data = yf.download(
                    crypto_list,
                    start=START,
                    end=END,
                    progress=False
                )
                
                if data.empty:
                    st.error("No data found for cryptocurrencies")
                    return None
                    
                data.to_csv('crypto_data.csv', index=True, header=True)
                lstm_data = data['Close'].copy()
                lstm_data.to_csv('crypto_data.csv', index=True, header=True)
                
                return data
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None

        @st.cache_data
        def load_data():
            try:
                if not os.path.exists('crypto_data.csv'):
                    data = download_data()
                
                data = pd.read_csv('crypto_data.csv', 
                                  sep=',',
                                  encoding='utf-8',
                                  index_col=0,
                                  parse_dates=True)
                
                st.write(f"Data range: {data.index.min()} to {data.index.max()}")
                st.write(f"Number of records: {len(data)}")
                
                return data
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None

        # Load data
        data_load_state = st.text('Loading data...')
        data = load_data()

        if data is not None and not data.empty:
            data_load_state.text('Loading data... done!')
            
            st.subheader('Raw data')
            st.write(data.head())

            # Plot raw data
            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[selected_stock],
                    name=selected_stock,
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title_text=f'{selected_stock} Price History',
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    xaxis_rangeslider_visible=True,
                    template='plotly_dark',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                stats = {
                    'Current Price': f"${data[selected_stock].iloc[-1]:.2f}",
                    'Highest Price': f"${data[selected_stock].max():.2f}",
                    'Lowest Price': f"${data[selected_stock].min():.2f}",
                    'Average Price': f"${data[selected_stock].mean():.2f}",
                    'Date Range': f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                    'Number of Trading Days': len(data)
                }
                st.write(stats)
            
            plot_raw_data()

            # Prepare data for models
            df_train = pd.DataFrame({
                'ds': data.index,
                'y': data[selected_stock]
            }).reset_index(drop=True)

            # Split data
            train_size = len(df_train) - 30
            train_data = df_train[:train_size]
            test_data = df_train[train_size:]

            if selected_model == 'Prophet':
                params = {
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'seasonality_prior_scale': seasonality_prior_scale,
                    'holidays_prior_scale': holidays_prior_scale,
                    'seasonality_mode': seasonality_mode
                }
                
                m = train_prophet(train_data, params)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                
                st.subheader('Forecast data')
                st.write(forecast.tail())
                
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_prophet_forecast(m, forecast, n_years, data, selected_stock)
                st.plotly_chart(fig1)

                # Calculate and display Prophet model performance metrics
                st.subheader('Model Performance Metrics')
                # Get predictions for the test period
                test_forecast = forecast[forecast['ds'].isin(test_data['ds'])]
                test_actual = test_data['y'].values
                test_pred = test_forecast['yhat'].values

                # Calculate metrics
                metrics = calculate_metrics(test_actual, test_pred)
                
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
                    st.metric("Root Mean Square Error (RMSE)", f"${metrics['RMSE']:.2f}")
                with col2:
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%")
                    st.metric("R² Score", f"{metrics['R2 Score']:.4f}")

                
                current_price = data[selected_stock].iloc[-1]
                st.subheader('Buy/Sell Signals Based on Prophet Forecast')
                st.table(buy_sell_signals_prophet(data, selected_stock, forecast, current_price))

            elif selected_model == 'ARIMA':
                if use_auto_arima:
                    import pmdarima as pm
                    model = pm.auto_arima(data[selected_stock], seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
                    order = model.order
                    st.info(f"Auto-ARIMA selected order: (p={order[0]}, d={order[1]}, q={order[2]})")
                    # Forecast
                    future_predictions = model.predict(n_periods=period)
                else:
                    model = train_arima(data[selected_stock], p, d, q)
                    future_predictions = model.forecast(steps=period)
                
                st.subheader(f'Future Predictions for {n_years} years')
                future_dates = pd.date_range(start=data.index[-1], periods=period+1)[1:]
                fig4 = plot_arima_forecast(data, selected_stock, future_dates, future_predictions, n_years)
                st.plotly_chart(fig4)

                # Calculate and display ARIMA model performance metrics
                st.subheader('Model Performance Metrics')
                
                if use_auto_arima:
                    # Use last 30 in-sample predictions for metrics
                    test_predictions = model.predict_in_sample()[-30:]
                else:
                    test_predictions = model.predict(start=len(data)-30, end=len(data)-1)
                test_actual = data[selected_stock].iloc[-30:].values
                min_length = min(len(test_actual), len(test_predictions))
                test_actual = test_actual[:min_length]
                test_predictions = test_predictions[:min_length]
                # Calculate metrics
                metrics = calculate_metrics(test_actual, test_predictions)
                
                # Display metrics in a nice format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
                    st.metric("Root Mean Square Error (RMSE)", f"${metrics['RMSE']:.2f}")
                with col2:
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%")
                    st.metric("R² Score", f"{metrics['R2 Score']:.4f}")

                
                current_price = data[selected_stock].iloc[-1]
                st.subheader('Buy/Sell Signals Based on ARIMA Forecast')
                st.table(buy_sell_signals_arima(current_price, future_predictions))

            elif selected_model == 'XGBoost':
                df = pd.DataFrame(index=data.index)
                df[selected_stock] = data[selected_stock]
                
                
                for lag in [1, 3, 7]:
                    df[f'{selected_stock}_lag_{lag}'] = df[selected_stock].shift(lag)
                
                
                df[f'{selected_stock}_rolling_7'] = df[selected_stock].rolling(7).mean()
                
                df[f'{selected_stock}_pct_change'] = df[selected_stock].pct_change() * 100
                
                
                df = df.dropna()
                
                feature_cols = [col for col in df.columns if col != selected_stock]
                X = df[feature_cols]
                y = df[selected_stock]
                
                # Split data 
                train_size = len(df) - 30
                X_train = X[:train_size]
                X_test = X[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]
                
                model = train_xgboost(X_train, y_train, max_depth, n_estimators, learning_rate)
                
                # Prepare future data 
                last_data = df.iloc[-1:].copy()
                future_predictions = []
                
                for _ in range(period):
                    
                    pred = model.predict(last_data[feature_cols])[0]
                    future_predictions.append(pred)
                    
                    # Update features for next prediction
                    new_row = last_data.copy()
                    new_row[selected_stock] = pred
                    new_row[f'{selected_stock}_lag_1'] = last_data[selected_stock].iloc[0]
                    new_row[f'{selected_stock}_lag_3'] = df[selected_stock].iloc[-3]
                    new_row[f'{selected_stock}_lag_7'] = df[selected_stock].iloc[-7]
                    new_row[f'{selected_stock}_rolling_7'] = df[selected_stock].iloc[-7:].mean()
                    new_row[f'{selected_stock}_pct_change'] = (pred - last_data[selected_stock].iloc[0]) / last_data[selected_stock].iloc[0] * 100
                    
                    last_data = new_row
                
                st.subheader(f'Future Predictions for {n_years} years')
                future_dates = pd.date_range(start=data.index[-1], periods=period+1)[1:]
                # Anchor forecast 
                anchored_forecast = [data[selected_stock].iloc[-1]] + list(future_predictions)
                anchored_dates = [data.index[-1]] + list(future_dates)
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=data.index,
                    y=data[selected_stock],
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                fig4.add_trace(go.Scatter(
                    x=anchored_dates,
                    y=anchored_forecast,
                    name='Forecast',
                    line=dict(color='red', width=2)
                ))
                fig4.update_layout(
                    title=f'{selected_stock} Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=600
                )
                st.plotly_chart(fig4)

                # Display XGBoost model metrics
                st.subheader('Model Performance Metrics')
                test_predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, test_predictions)
                
                # Display metrics 
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
                    st.metric("Root Mean Square Error (RMSE)", f"${metrics['RMSE']:.2f}")
                with col2:
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%")
                    st.metric("R² Score", f"{metrics['R2 Score']:.4f}")

                # --- Buy/Sell/Hold Signal Table ---
                current_price = data[selected_stock].iloc[-1]
                st.subheader('Buy/Sell Signals Based on XGBoost Forecast')
                st.table(buy_sell_signals_xgboost(current_price, future_predictions))

            elif selected_model == 'LSTM':
                # Train NeuralForecast LSTM model
                nf_model, forecast = train_neural_forecast_lstm(
                    data[selected_stock],
                    h=forecast_horizon,
                    max_steps=max_steps
                )
                
                # Create forecast dates
                future_dates = pd.date_range(start=data.index[-1], periods=forecast_horizon+1)[1:]
                
                # Anchor forecast to last actual value 
                last_actual = data[selected_stock].iloc[-1]
                anchored_forecast = [last_actual] + list(forecast['LSTM'].values)
                anchored_dates = [data.index[-1]] + list(future_dates)
                
                # Plot the results
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[selected_stock],
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Add anchored forecast
                fig.add_trace(go.Scatter(
                    x=anchored_dates,
                    y=anchored_forecast,
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title=f'{selected_stock} Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig)

                # Calculate and display LSTM model performance metrics
                st.subheader('Model Performance Metrics')
                # Get the last forecast_horizon days of actual data for comparison
                test_actual = data[selected_stock].iloc[-forecast_horizon:].values
                test_predictions = forecast['LSTM'].values[:forecast_horizon]
                
                # Ensure both arrays have the same length
                min_length = min(len(test_actual), len(test_predictions))
                test_actual = test_actual[:min_length]
                test_predictions = test_predictions[:min_length]
                
                # Calculate metrics
                metrics = calculate_metrics(test_actual, test_predictions)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
                    st.metric("Root Mean Square Error (RMSE)", f"${metrics['RMSE']:.2f}")
                with col2:
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%")
                    st.metric("R² Score", f"{metrics['R2 Score']:.4f}")

                
                current_price = data[selected_stock].iloc[-1]
                st.subheader('Buy/Sell Signals Based on LSTM Forecast')
                st.table(buy_sell_signals_lstm(current_price, forecast))

            # Profit Calculator
            st.subheader('Investment Profit Calculator')
            current_price = data[selected_stock].iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                investment_amount = st.number_input('Investment Amount (USD)', 
                                                  min_value=100.0, 
                                                  value=1000.0, 
                                                  step=100.0)
            with col2:
                coins_bought = investment_amount / current_price
                st.write(f'Coins you can buy: {coins_bought:.8f}')
            
            st.write('Potential Profit Calculations')
            if selected_model == 'Prophet':
                profit_df = calculate_profit_table(coins_bought, investment_amount, forecast, forecast_type='prophet')
            elif selected_model == 'ARIMA':
                profit_df = calculate_profit_table(coins_bought, investment_amount, future_predictions, forecast_type='arima')
            elif selected_model == 'XGBoost':
                profit_df = calculate_profit_table(coins_bought, investment_amount, future_predictions, forecast_type='xgboost')
            elif selected_model == 'LSTM':
                lstm_forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast['LSTM'].values
                })
                profit_df = calculate_profit_table(coins_bought, investment_amount, lstm_forecast, forecast_type='lstm')
            
            st.table(profit_df)
            
            st.warning("""
            ⚠️ **Investment Risk Warning:**
            - These predictions are based on historical data and mathematical models
            - Cryptocurrency markets are highly volatile
            - Past performance does not guarantee future results
            - Never invest more than you can afford to lose
            """)

        else:
            st.error("Failed to load data. Please try again.")

        if st.button('Stop App'):
            st.write('Stopping the app...')
            st.stop()