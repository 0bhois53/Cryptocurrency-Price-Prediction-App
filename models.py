import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM as NF_LSTM
from neuralforecast.losses.pytorch import MSE, MAE

def train_arima(data, p, d, q):
    # Train ARIMA model
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

def train_xgboost(X_train, y_train, max_depth, n_estimators, learning_rate):
    # Create and train XGBoost model with early stopping
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    # Create validation set
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Fit model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def train_prophet(train_data, params):
    # Create and train Prophet model
    m = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode']
    )
    
    # Add custom seasonalities
    m.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    # Fit the model
    m.fit(train_data)
    
    return m

def train_neural_forecast_lstm(data, h=30, max_steps=750):
    # Train LSTM model using NeuralForecast library
    df = pd.DataFrame({
        'ds': pd.to_datetime(data.index),
        'y': data.values,
        'unique_id': 'crypto'
    })
    lstm_model = NF_LSTM(
        h=h,  # forecast horizon
        input_size=h,  # input window size should match forecast horizon
        max_steps=max_steps,
        loss=MAE(),
        scaler_type='standard'
    )
    nf = NeuralForecast(models=[lstm_model], freq='D')
    nf.fit(df=df)
    forecast = nf.predict()
    return nf, forecast

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R2 Score': r2_score(y_true, y_pred)
    }
    return metrics 