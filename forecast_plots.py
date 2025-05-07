import plotly.graph_objs as go
import pandas as pd

def plot_prophet_forecast(m, forecast, n_years, data, selected_stock):
    from prophet.plot import plot_plotly
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(title=f'Forecast plot for {n_years} years', xaxis_title='Date', yaxis_title='Price (USD)')
    return fig1

def plot_arima_forecast(data, selected_stock, future_dates, future_predictions, n_years):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[selected_stock],
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title=f'{selected_stock} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=600
    )
    return fig

def plot_xgboost_forecast(data, selected_stock, future_dates, future_predictions, n_years):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[selected_stock],
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title=f'{selected_stock} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=600
    )
    return fig

def plot_lstm_forecast(data, selected_stock, future_dates, future_predictions, n_years):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[selected_stock],
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions.flatten(),
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title=f'{selected_stock} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=600
    )
    return fig 