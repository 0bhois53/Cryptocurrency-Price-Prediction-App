import pandas as pd

def buy_sell_signals_prophet(data, selected_stock, forecast, current_price):
    signal_periods = {'1 Week': 7, '2 Weeks': 14, '1 Month': 30}
    signal_data = []
    for label, days in signal_periods.items():
        if days < len(forecast):
            future_row = forecast.iloc[days]
            forecasted_price = future_row['yhat']
            pct_change = (forecasted_price - current_price) / current_price * 100
            if pct_change > 2:
                signal = 'Buy'
            elif pct_change < -2:
                signal = 'Sell'
            else:
                signal = 'Hold'
            signal_data.append({
                'Period': label,
                'Forecasted Price': f"${forecasted_price:.2f}",
                'Change (%)': f"{pct_change:.2f}%",
                'Signal': signal
            })
    return pd.DataFrame(signal_data)

def buy_sell_signals_arima(current_price, future_predictions):
    signal_periods = {'1 Week': 7, '2 Weeks': 14, '1 Month': 30}
    signal_data = []
    for label, days in signal_periods.items():
        if days <= len(future_predictions):
            forecasted_price = future_predictions[days-1]
            pct_change = (forecasted_price - current_price) / current_price * 100
            if pct_change > 2:
                signal = 'Buy'
            elif pct_change < -2:
                signal = 'Sell'
            else:
                signal = 'Hold'
            signal_data.append({
                'Period': label,
                'Forecasted Price': f"${forecasted_price:.2f}",
                'Change (%)': f"{pct_change:.2f}%",
                'Signal': signal
            })
    return pd.DataFrame(signal_data)

def buy_sell_signals_xgboost(current_price, future_predictions):
    signal_periods = {'1 Week': 7, '2 Weeks': 14, '1 Month': 30}
    signal_data = []
    for label, days in signal_periods.items():
        if days <= len(future_predictions):
            forecasted_price = future_predictions[days-1]
            pct_change = (forecasted_price - current_price) / current_price * 100
            if pct_change > 2:
                signal = 'Buy'
            elif pct_change < -2:
                signal = 'Sell'
            else:
                signal = 'Hold'
            signal_data.append({
                'Period': label,
                'Forecasted Price': f"${forecasted_price:.2f}",
                'Change (%)': f"{pct_change:.2f}%",
                'Signal': signal
            })
    return pd.DataFrame(signal_data)

def buy_sell_signals_lstm(current_price, forecast):
    signal_periods = {'1 Week': 7, '2 Weeks': 14, '1 Month': 30}
    signal_data = []
    for label, days in signal_periods.items():
        if days < len(forecast['LSTM']):
            forecasted_price = forecast['LSTM'].values[days-1]
            pct_change = (forecasted_price - current_price) / current_price * 100
            if pct_change > 2:
                signal = 'Buy'
            elif pct_change < -2:
                signal = 'Sell'
            else:
                signal = 'Hold'
            signal_data.append({
                'Period': label,
                'Forecasted Price': f"${forecasted_price:.2f}",
                'Change (%)': f"{pct_change:.2f}%",
                'Signal': signal
            })
    return pd.DataFrame(signal_data) 