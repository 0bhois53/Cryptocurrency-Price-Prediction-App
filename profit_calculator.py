import pandas as pd
import streamlit as st

def calculate_profit_table(coins_bought, investment_amount, forecast, forecast_type='prophet'):
   
    if forecast_type in ['prophet', 'lstm']:
        today = pd.Timestamp.now()
        forecast_dates = {
            '1 Week': today + pd.Timedelta(weeks=1),
            '1 Month': today + pd.Timedelta(days=30),
            '3 Months': today + pd.Timedelta(days=90),
            '6 Months': today + pd.Timedelta(days=180),
            '1 Year': today + pd.Timedelta(days=365)
        }
        profit_data = []
        for period, future_date in forecast_dates.items():
            closest_date = forecast['ds'].iloc[(forecast['ds'] - future_date).abs().argsort()[0]]
            predicted_price = forecast.loc[forecast['ds'] == closest_date, 'yhat'].iloc[0]
            future_value = coins_bought * predicted_price
            profit = future_value - investment_amount
            profit_percentage = (profit / investment_amount) * 100
            profit_data.append({
                'Period': period,
                'Predicted Price': f'${predicted_price:.2f}',
                'Investment Value': f'${future_value:.2f}',
                'Profit/Loss': f'${profit:.2f}',
                'Return': f'{profit_percentage:.2f}%'
            })
        return pd.DataFrame(profit_data)
    else:
        forecast_dates = {
            '1 Week': 7,
            '1 Month': 30,
            '3 Months': 90,
            '6 Months': 180,
            '1 Year': 365
        }
        profit_data = []
        for period, days in forecast_dates.items():
            if days <= len(forecast):
                predicted_price = forecast[days-1]
                future_value = coins_bought * predicted_price
                profit = future_value - investment_amount
                profit_percentage = (profit / investment_amount) * 100
                profit_data.append({
                    'Period': period,
                    'Predicted Price': f'${predicted_price:.2f}',
                    'Investment Value': f'${future_value:.2f}',
                    'Profit/Loss': f'${profit:.2f}',
                    'Return': f'{profit_percentage:.2f}%'
                })
        return pd.DataFrame(profit_data)

 