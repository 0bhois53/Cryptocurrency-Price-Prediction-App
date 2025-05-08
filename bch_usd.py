# -*- coding: utf-8 -*-


import pandas as pd

data = pd.read_csv('LSTM_crypto_data.csv', sep=',',encoding='utf-8',index_col=0,parse_dates=True)

data['ds'] = pd.to_datetime(data.index)
data['y'] = data['BCH-USD']

data['unique_id'] = 'BCH-USD'



from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MSE,MAE
import matplotlib.pyplot as plt
import torch

bch_lstm_model = LSTM(
    h=30,
    input_size=60,
    max_steps=750,
    #hidden_size=128,
    loss= MAE(),
    scaler_type='standard'

    )

nf = NeuralForecast(models=[bch_lstm_model],freq='D')
nf.fit(df=data)
forecast = nf.predict()

import plotly.express as px

fig = px.line(data, x='ds', y='y', labels={'ds': 'Date', 'y': 'Value'}, title='BCH-USD Forecast')
fig.add_scatter(x=forecast['ds'], y=forecast['LSTM'], mode='lines', name='Forecast', line=dict(color='red'))
fig.show()

