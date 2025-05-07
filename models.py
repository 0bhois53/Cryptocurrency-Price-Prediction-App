import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import streamlit as st

def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
        target.append(data[i + seq_length])
    return np.array(sequences), np.array(target)

class LSTMWrapper:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.model = None
        
    def build_model(self, units, dropout):
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(self.seq_length, 1)),
            Dropout(dropout),
            LSTM(units=units),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mse')
        return model
    
    def fit(self, X, y, **kwargs):
        self.model = self.build_model(kwargs['units'], kwargs['dropout'])
        self.model.fit(X, y, epochs=kwargs['epochs'], batch_size=kwargs['batch_size'], verbose=0)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

def optimize_lstm_parameters(data, seq_length):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Define parameter grid
    param_grid = {
        'units': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100]
    }
    
    # Create base model
    model = LSTMWrapper(seq_length)
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    progress_bar.empty()
    progress_text.empty()
    
    return grid_search.best_params_

def train_lstm(data, seq_length, units=None, dropout=None, epochs=None, auto_optimize=False):
    if auto_optimize:
        best_params = optimize_lstm_parameters(data, seq_length)
        units = best_params['units']
        dropout = best_params['dropout']
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']
        st.write(f"Optimal LSTM parameters found: {best_params}")
    else:
        batch_size = 32
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and compile the model
    model = LSTMWrapper(seq_length)
    
    # Fit the model
    model.fit(X_train, y_train, units=units, dropout=dropout, epochs=epochs, batch_size=batch_size)
    
    return model, scaler, (X_train, X_test, y_train, y_test)

def optimize_arima_parameters(data, max_p=5, max_d=3, max_q=5):
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

def train_arima(data, p, d, q, P=0, D=0, Q=0, s=0, auto_optimize=False):
    # Convert data to time series
    index = pd.date_range(start=data.index[0], end=data.index[-1], freq='D')
    ts = pd.Series(data.values, index=index)
    
    if auto_optimize:
        # Find optimal parameters
        p, d, q = optimize_arima_parameters(ts)
        st.write(f"Optimal ARIMA parameters found: p={p}, d={d}, q={q}")
    
    ts = ts.interpolate()  # or ts = ts.fillna(method='ffill')
    
    # Fit SARIMAX model (ARIMA if seasonal_order is all zeros)
    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s) if s > 0 else (0, 0, 0, 0))
    results = model.fit(disp=False)
    
    # Display model summary in expandable section
    with st.expander("View SARIMAX Model Summary"):
        st.text(str(results.summary()))
    
    return results

def optimize_xgboost_parameters(X_train, y_train):
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create base model
    model = xgb.XGBRegressor()
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    progress_bar.empty()
    progress_text.empty()
    
    return grid_search.best_params_

def train_xgboost(X_train, y_train, max_depth=None, n_estimators=None, learning_rate=None, auto_optimize=False):
    if auto_optimize:
        best_params = optimize_xgboost_parameters(X_train, y_train)
        max_depth = best_params['max_depth']
        n_estimators = best_params['n_estimators']
        learning_rate = best_params['learning_rate']
        subsample = best_params['subsample']
        colsample_bytree = best_params['colsample_bytree']
        st.write(f"Optimal XGBoost parameters found: {best_params}")
    else:
        subsample = 1.0
        colsample_bytree = 1.0
    
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )
    model.fit(X_train, y_train)
    return model

def train_prophet(train_data, params):
    # Create and train Prophet model with optimized parameters
    m = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
       
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