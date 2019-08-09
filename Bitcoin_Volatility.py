import pandas as pd
import requests
import os.path
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
import arch
import numpy as np
from datetime import date, timedelta

matplotlib.rcParams['figure.figsize'] = [10.0, 4.0]

def get_crypto_data_daily(crypto, use_cache=True):
    filename = '{}_daily.csv'.format(crypto)
    if use_cache and os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=[0, 6]).set_index('Date')
    dates = [date(2013, 3, 15), date(2013, 9, 15),
            date(2014, 3, 15), date(2014, 9, 15),
            date(2015, 3, 15), date(2015, 9, 15),
            date(2016, 3, 15), date(2016, 9, 15),
            date(2017, 3, 15), date(2017, 9, 15),
            date(2018, 3, 15)]
    url = 'https://api.gdax.com/products/{}-USD/candles?granularity=86400&start={}&end={}'
    df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Open',
                               'Close', 'Volume'])
    for i in range(len(dates) - 1):
        sleep(1)
        data = requests.get(url.format(crypto, dates[i], dates[i+1]))
        # don't do eval
        data = pd.DataFrame(columns=['Date', 'Low', 'High', 'Open',
                                     'Close', 'Volume'], data=eval(data.content))
        df = df.append(data)
        
    df = df.sort_values('Date')
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df.to_csv(filename, index=False)
    return df.set_index('Date')


btc_d = get_crypto_data_daily('BTC', use_cache=False)

gold = (pd.read_csv('gold.csv', parse_dates=[0])
        .sort_values('Date').set_index('Date').query("Date > '1980-01-01'"))

sp = pd.read_csv('sp500.csv').query("Date > '2001-03-15'")
sp['Date'] = pd.to_datetime(sp['Date'])
sp = sp.set_index('Date')
sp['Week'] = sp.index.week

fed_rate_probs = pd.read_csv('FedMeeting_20171213.csv')
fed_rate_probs['Date'] = pd.to_datetime(fed_rate_probs['Date'])
fed_rate_probs = fed_rate_probs.set_index('Date')

