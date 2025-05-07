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
    train_lstm,
    train_arima,
    train_xgboost,
    train_prophet,
    calculate_metrics,
    create_sequences
)
from profit_calculator import calculate_profit_table
from forecast_plots import (
    plot_prophet_forecast,
    plot_arima_forecast,
    plot_xgboost_forecast,
    plot_lstm_forecast
)

# Helper functions
def optimize_arima_parameters(data, max_p=5, max_d=2, max_q=5):
    best_aic = float('inf')
    best_params = None
    
    # Create progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_iterations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_iteration = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                current_iteration += 1
                progress = current_iteration / total_iterations
                progress_bar.progress(progress)
                progress_text.text(f'Testing ARIMA({p},{d},{q})')
                
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                except:
                    continue
    
    progress_bar.empty()
    progress_text.empty()
    return best_params

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
    st.title('Cryptocurrency Forecast App')

    START = "2021-01-01"
    END = date.today()

    # Define crypto list
    crypto_list = ['BTC-USD', 'STETH-USD', 'LEO-USD', 'BCH-USD']
    selected_stock = st.selectbox('Select cryptocurrency for prediction', crypto_list)

    # Add model selection
    model_list = ['Prophet', 'ARIMA', 'XGBoost', 'LSTM']
    selected_model = st.selectbox('Select forecasting model', model_list)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Model-specific hyperparameters
    if selected_model == 'ARIMA':
        st.sidebar.subheader('ARIMA Parameters')
        p = st.sidebar.slider('p (AR order)', 0, 5, 2)
        d = st.sidebar.slider('d (Difference order)', 0, 2, 1)
        q = st.sidebar.slider('q (MA order)', 0, 5, 2)

        # Add confidence interval selection
        confidence_interval = st.sidebar.slider('Confidence Interval (%)', 80, 99, 95)
    elif selected_model == 'XGBoost':
        st.sidebar.subheader('XGBoost Parameters')
        max_depth = st.sidebar.slider('max_depth', 3, 10, 6)
        n_estimators = st.sidebar.slider('n_estimators', 50, 300, 100)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.3, 0.1)

    elif selected_model == 'LSTM':
        st.sidebar.subheader('LSTM Parameters')
        seq_length = st.sidebar.slider('Sequence Length', 10, 60, 30)
        units = st.sidebar.slider('units', 32, 128, 50)
        dropout = st.sidebar.slider('dropout', 0.0, 0.5, 0.2)
        epochs = st.sidebar.slider('epochs', 10, 100, 50)

    elif selected_model == 'Prophet':
        st.sidebar.subheader('Prophet Parameters')
        changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale (trend flexibility)', 0.001, 0.5, 0.05)
        seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', 0.01, 10.0, 10.0)
        holidays_prior_scale = st.sidebar.slider('Holidays Prior Scale', 0.01, 10.0, 10.0)
        seasonality_mode = st.sidebar.selectbox('Seasonality Mode', ['additive', 'multiplicative'])

    # Function to fetch crypto news
    def fetch_crypto_news():
        try:
            # CryptoCompare News API 
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = requests.get(url)
            news_data = response.json()
            return news_data.get('Data', [])
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []

    # Function to display news in the sidebar
    def display_news(news_items):
        for item in news_items[:10]:  # Display top 10 news 
            with st.container():
                st.markdown("""
                <div class="news-container">
                """, unsafe_allow_html=True)
                
                # Create columns for image and text
                img_col, text_col = st.columns([1, 3])
                
                with img_col:
                    try:
                        # Load and display the news image
                        response = requests.get(item['imageurl'])
                        img = Image.open(BytesIO(response.content))
                        st.image(img, use_container_width=True)
                    except:
                        # Display a placeholder if image loading fails
                        st.write("📰")
                
                with text_col:
                    # Display news title and details
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

    # Continue with the main app in the main column
    with main_col:
        @st.cache_data
        def download_data():
            try:
                # Debug information
                st.write("Fetching data for all cryptocurrencies...")
                
                # Download data for all cryptos at once
                data = yf.download(
                    crypto_list,
                    start=START,
                    end=END,
                    progress=False
                )
                
                if data.empty:
                    st.error("No data found for cryptocurrencies")
                    return None
                    
                # Save the complete dataset
                data.to_csv('crypto_data.csv', index=True, header=True)
                
                # Process the data 
                lstm_data = data['Close'].copy()
                lstm_data.to_csv('LSTM_crypto_data.csv', index=True, header=True)
                
                return data
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None

        @st.cache_data
        def load_data():
            try:
                if not os.path.exists('LSTM_crypto_data.csv'):
                    data = download_data()
                
                # Load the processed data
                data = pd.read_csv('LSTM_crypto_data.csv', 
                                  sep=',',
                                  encoding='utf-8',
                                  index_col=0,
                                  parse_dates=True)
                
                # Debug information
                st.write(f"Data range: {data.index.min()} to {data.index.max()}")
                st.write(f"Number of records: {len(data)}")
                
                return data
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None

        def optimize_hyperparameters(df_train):
            """Grid search for hyperparameter optimization."""
            # Smaller parameter grid to reduce computation time
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
                'seasonality_prior_scale': [0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }
            
            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []  # Store the RMSEs for each params
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate appropriate cross-validation parameters based on data size
            n_days = len(df_train)
            # Use 60% of data for initial training
            initial = f"{int(n_days * 0.6)} days"
            # Use 20% of data for each validation step
            period = f"{int(n_days * 0.2)} days"
            # Use 10% of data for horizon
            horizon = f"{int(n_days * 0.1)} days"
            
            st.write(f"Cross-validation parameters:")
            st.write(f"Initial training period: {initial}")
            st.write(f"Validation period: {period}")
            st.write(f"Forecast horizon: {horizon}")
            
            # Use cross validation to evaluate all parameters
            for i, params in enumerate(all_params):
                status_text.text(f'Trying parameters {i+1}/{len(all_params)}...')
                progress_bar.progress((i + 1)/len(all_params))
                
                try:
                    m = Prophet(**params).fit(df_train)  # Fit model with given params
                    
                    # Cross-validation
                    df_cv = cross_validation(
                        m, 
                        initial=initial,
                        period=period,
                        horizon=horizon,
                        parallel="threads"
                    )
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    
                    rmses.append(df_p['rmse'].mean())
                except Exception as e:
                    st.error(f"Error during cross-validation with parameters {params}: {str(e)}")
                    rmses.append(float('inf'))  # Assign worst possible score
            
            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmses
            
            # Filter out any failed attempts
            valid_results = tuning_results[tuning_results['rmse'] != float('inf')]
            
            if len(valid_results) > 0:
                best_params = all_params[valid_results['rmse'].idxmin()]
                status_text.text('Hyperparameter optimization completed!')
            else:
                st.error("No valid parameter combinations found. Using default parameters.")
                best_params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'holidays_prior_scale': 10.0,
                    'seasonality_mode': 'additive'
                }
            
            progress_bar.empty()
            
            return best_params, tuning_results

        # Load data once
        data_load_state = st.text('Loading data...')
        data = load_data()

        if data is not None and not data.empty:
            data_load_state.text('Loading data... done!')
            
            st.subheader('Raw data')
            st.write(data.head())
            

            # Plot raw data
            def plot_raw_data():
                try:
                    fig = go.Figure()
                    
                    # Add trace for selected cryptocurrency
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[selected_stock],
                        name=selected_stock,
                        line=dict(color='#1f77b4', width=2),
                        mode='lines'
                    ))
                    
                    # Update layout with better styling
                    fig.update_layout(
                        title_text=f'{selected_stock} Price History ({data.index.min().strftime("%Y-%m-%d")} to {data.index.max().strftime("%Y-%m-%d")})',
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        xaxis_rangeslider_visible=True,
                        template='plotly_dark',
                        showlegend=True,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display some basic statistics
                    st.subheader('Basic Statistics')
                    stats = {
                        'Current Price': f"${data[selected_stock].iloc[-1]:.2f}",
                        'Highest Price': f"${data[selected_stock].max():.2f}",
                        'Lowest Price': f"${data[selected_stock].min():.2f}",
                        'Average Price': f"${data[selected_stock].mean():.2f}",
                        'Date Range': f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                        'Number of Trading Days': len(data)
                    }
                    st.write(stats)
                    
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
                    st.write("Data shape:", data.shape)
                    st.write("Data columns:", data.columns.tolist())
            
            plot_raw_data()

            # Prepare data for Prophet
            df_train = pd.DataFrame({
                'ds': data.index,
                'y': data[selected_stock]
            }).reset_index(drop=True)

            # Split the data into training and testing sets (last 30 days for testing)
            train_size = len(df_train) - 30
            train_data = df_train[:train_size]
            test_data = df_train[train_size:]

            if selected_model == 'Prophet':
                # Use default or user-selected parameters
                m = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    seasonality_mode=seasonality_mode
                )

                m.fit(train_data)

                # Make predictions on test set
                future_test = m.make_future_dataframe(periods=30)
                forecast_test = m.predict(future_test)
                
                # Calculate metrics on test set
                y_true = test_data['y'].values
                y_pred = forecast_test.tail(30)['yhat'].values
                metrics = calculate_metrics(y_true, y_pred)

                # Display model performance metrics
                st.subheader('Model Performance Metrics (Last 30 Days)')
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" for v in metrics.values()]
                })
                st.table(metrics_df)

                # Make future predictions
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                # Show and plot forecast
                st.subheader('Forecast data')
                st.write(forecast.tail())
                
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_prophet_forecast(m, forecast, n_years, data, selected_stock)
                st.plotly_chart(fig1)

                st.write("Forecast components")
                fig2 = m.plot_components(forecast)
                st.write(fig2)

                # Plot actual vs predicted for test period
                st.subheader('Model Validation (Last 30 Days)')
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=test_data['ds'],
                    y=test_data['y'],
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=test_data['ds'],
                    y=forecast_test.tail(30)['yhat'],
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                fig3.update_layout(
                    title='Actual vs Predicted Values (Last 30 Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig3)

                # After the forecast section, add the profit calculator
                st.subheader('Investment Profit Calculator')
                # Get current price
                current_price = data[selected_stock].iloc[-1]
                # Investment input
                col1, col2 = st.columns(2)
                with col1:
                    investment_amount = st.number_input('Investment Amount (USD)', 
                                                      min_value=100.0, 
                                                      value=1000.0, 
                                                      step=100.0)
                with col2:
                    coins_bought = investment_amount / current_price
                    st.write(f'Coins you can buy: {coins_bought:.8f}')
                # Create profit calculation table
                st.write('Potential Profit Calculations')
                profit_df = calculate_profit_table(coins_bought, investment_amount, forecast, forecast_type='prophet')
                st.table(profit_df)
                
                # Add risk warning
                st.warning("""
                ⚠️ **Investment Risk Warning:**
                - These predictions are based on historical data and mathematical models
                - Cryptocurrency markets are highly volatile
                - Past performance does not guarantee future results
                - Never invest more than you can afford to lose
                """)
                
                # Visualization of potential returns
                forecast_dates = {
                    '1 Week': 7,
                    '1 Month': 30,
                    '3 Months': 90,
                    '6 Months': 180,
                    '1 Year': 365
                }
                fig_profit = go.Figure()
                
                # Add investment amount line
                fig_profit.add_trace(go.Scatter(
                    x=[period for period in forecast_dates.keys()],
                    y=[investment_amount] * len(forecast_dates),
                    name='Initial Investment',
                    line=dict(color='yellow', dash='dash'),
                    mode='lines'
                ))
                
                # Add predicted value line
                fig_profit.add_trace(go.Scatter(
                    x=[period for period in forecast_dates.keys()],
                    y=[float(data['Investment Value'].replace('$', '')) for data in profit_df.to_dict('records')],
                    name='Predicted Value',
                    line=dict(color='green'),
                    mode='lines+markers'
                ))
                
                fig_profit.update_layout(
                    title='Potential Investment Growth Over Time',
                    xaxis_title='Time Period',
                    yaxis_title='Value (USD)',
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig_profit, use_container_width=True)

            elif selected_model == 'ARIMA':
                # Train the model
                model = train_arima(data[selected_stock], p, d, q)
                
                # Make predictions for the test period (last 30 days)
                test_predictions = model.forecast(steps=30)
                
                # Make future predictions for the selected period
                future_predictions = model.forecast(steps=period)
                
                # Calculate metrics using only the test period
                y_true = data[selected_stock].values[-30:]
                y_pred = test_predictions
                metrics = calculate_metrics(y_true, y_pred)
                
                # Display model performance metrics
                st.subheader('Model Performance Metrics (Last 30 Days)')
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" for v in metrics.values()]
                })
                st.table(metrics_df)
                
                # Plot actual vs predicted with confidence intervals
                st.subheader('Model Validation (Last 30 Days)')
                fig3 = go.Figure()
                
                # Plot actual values
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_true,
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                
                # Plot predicted values
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_pred,
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                
                # Add confidence intervals
                conf_int = model.get_forecast(steps=30).conf_int(alpha=(100-confidence_interval)/100)
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=conf_int.iloc[:, 0],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    showlegend=False
                ))
                
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=conf_int.iloc[:, 1],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name=f'{confidence_interval}% Confidence Interval'
                ))
                
                fig3.update_layout(
                    title='Actual vs Predicted Values with Confidence Intervals (Last 30 Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig3)

                # Plot future predictions
                future_dates = pd.date_range(start=data.index[-1], periods=period+1)[1:]
                st.subheader(f'Future Predictions for {n_years} years')
                fig4 = plot_arima_forecast(data, selected_stock, future_dates, future_predictions, n_years)
                st.plotly_chart(fig4)

                # Profit Calculator for ARIMA
                st.subheader('Investment Profit Calculator')
                current_price = data[selected_stock].iloc[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    investment_amount = st.number_input('Investment Amount (USD)', 
                                                      min_value=100.0, 
                                                      value=1000.0, 
                                                      step=100.0,
                                                      key='arima_investment')
                
                with col2:
                    coins_bought = investment_amount / current_price
                    st.write(f'Coins you can buy: {coins_bought:.8f}')
                
                # Create profit calculation table
                st.write('Potential Profit Calculations')
                profit_df = calculate_profit_table(coins_bought, investment_amount, future_predictions, forecast_type='arima')
                st.table(profit_df)
                
                st.warning("""
                ⚠️ **Investment Risk Warning:**
                - These predictions are based on historical data and mathematical models
                - Cryptocurrency markets are highly volatile
                - Past performance does not guarantee future results
                - Never invest more than you can afford to lose
                """)

            elif selected_model == 'XGBoost':
                # Train the model
                model = train_xgboost(
                    data[selected_stock].values.reshape(-1, 1),
                    data[selected_stock].values,
                    max_depth,
                    n_estimators,
                    learning_rate
                )

                # Make predictions
                predictions = model.predict(data[selected_stock].values.reshape(-1, 1))

                # Calculate metrics
                y_true = data[selected_stock].values
                y_pred = predictions
                metrics = calculate_metrics(y_true, y_pred)

                # Display model performance metrics
                st.subheader('Model Performance Metrics (Last 30 Days)')
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" for v in metrics.values()]
                })
                st.table(metrics_df)

                # Plot actual vs predicted for test period
                st.subheader('Model Validation (Last 30 Days)')
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_true[-30:],
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_pred[-30:],
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                fig3.update_layout(
                    title='Actual vs Predicted Values (Last 30 Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig3)

                # Make future predictions
                future_data = np.arange(len(data), len(data) + period).reshape(-1, 1)
                future_predictions = model.predict(future_data)
                
                # Plot future predictions
                future_dates = pd.date_range(start=data.index[-1], periods=period+1)[1:]
                st.subheader(f'Future Predictions for {n_years} years')
                fig4 = plot_xgboost_forecast(data, selected_stock, future_dates, future_predictions, n_years)
                st.plotly_chart(fig4)

                # Profit Calculator for XGBoost
                st.subheader('Investment Profit Calculator')
                current_price = data[selected_stock].iloc[-1]
                col1, col2 = st.columns(2)
                with col1:
                    investment_amount = st.number_input('Investment Amount (USD)', 
                                                      min_value=100.0, 
                                                      value=1000.0, 
                                                      step=100.0,
                                                      key='xgb_investment')
                with col2:
                    coins_bought = investment_amount / current_price
                    st.write(f'Coins you can buy: {coins_bought:.8f}')
                # Create profit calculation table
                st.write('Potential Profit Calculations')
                profit_df = calculate_profit_table(coins_bought, investment_amount, future_predictions, forecast_type='xgboost')
                st.table(profit_df)
                
                st.warning("""
                ⚠️ **Investment Risk Warning:**
                - These predictions are based on historical data and mathematical models
                - Cryptocurrency markets are highly volatile
                - Past performance does not guarantee future results
                - Never invest more than you can afford to lose
                """)

            elif selected_model == 'LSTM':
                # Train the model
                model, scaler, (X_train, X_test, y_train, y_test) = train_lstm(
                    data[selected_stock].values,
                    seq_length,
                    units,
                    dropout,
                    epochs
                )

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate metrics
                y_true = y_test
                y_pred = predictions
                metrics = calculate_metrics(y_true, y_pred)

                # Display model performance metrics
                st.subheader('Model Performance Metrics (Last 30 Days)')
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': [f"{v:.4f}" for v in metrics.values()]
                })
                st.table(metrics_df)

                # Plot actual vs predicted for test period
                st.subheader('Model Validation (Last 30 Days)')
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_true[-30:],
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=y_pred[-30:],
                    name='Predicted',
                    line=dict(color='red', width=2)
                ))
                fig3.update_layout(
                    title='Actual vs Predicted Values (Last 30 Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig3)

                # Make future predictions
                last_sequence = X_test[-1:]
                future_predictions = []
                
                for _ in range(period):
                    next_pred = model.predict(last_sequence)
                    future_predictions.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = next_pred[0, 0]
                
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                
                # Plot future predictions
                future_dates = pd.date_range(start=data.index[-1], periods=period+1)[1:]
                st.subheader(f'Future Predictions for {n_years} years')
                fig4 = plot_lstm_forecast(data, selected_stock, future_dates, future_predictions, n_years)
                st.plotly_chart(fig4)

                # Profit Calculator for LSTM
                st.subheader('Investment Profit Calculator')
                current_price = data[selected_stock].iloc[-1]
                col1, col2 = st.columns(2)
                with col1:
                    investment_amount = st.number_input('Investment Amount (USD)', 
                                                      min_value=100.0, 
                                                      value=1000.0, 
                                                      step=100.0,
                                                      key='lstm_investment')
                with col2:
                    coins_bought = investment_amount / current_price
                    st.write(f'Coins you can buy: {coins_bought:.8f}')
                # Create profit calculation table
                st.write('Potential Profit Calculations')
                profit_df = calculate_profit_table(coins_bought, investment_amount, future_predictions, forecast_type='lstm')
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