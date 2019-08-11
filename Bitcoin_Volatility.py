import pandas as pd
import requests
import os.path
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
import arch
import numpy as np
from datetime import date, timedelta
import statsmodels.tsa.api as smt

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
        # don't do eval. ever... okay maybe just this once
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



# Smoothness metric by approximating the integral of the square of the second
# derivative inspired by the penalty used in smoothing splines
def smoothness(s):
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    s = s.apply(np.log)
    return ((s.diff()[1:].diff()[1:])**2).mean()

# standard deviation of log returns, the most common metric for volatility
def return_std(s):
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    d = s.apply(np.log).diff()[1:]
    return d.std()

# calculates the volatilities pre and post shock for a given metric and time window
def pre_post_shock(stock, date, metric='Close', vol_fn=smoothness, window=10):
    pre = stock[stock.index < date][metric].tail(window)
    post = stock[stock.index > date][metric].head(window)
    return vol_fn(pre), vol_fn(post)

def centered_window(df, date, window=7):
    pre = df[df.index < date].tail(window)
    post = df[df.index >= date].head(window+1)
    return pre.append(post)



# Fit ARIMA(p, d, q) model
# pick best order and final model based on aic

def get_best_model(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None
    pq_rng = range(1, 7, 2)
    d_rng = range(2)
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl


btc_arima_dat = btc_d.query("Date > '2015-03-15'")['Close'].apply(np.log).diff()[1:]
get_best_model(btc_arima_dat) # (3,0,3)



# now define an egarch model using the optimal arima parameters
def egarch_model(data, dist='Normal'):
    return (arch.arch_model(data.apply(np.log).diff()[1:],
                            vol='EGARCH', p=3, o=0, q=3, dist=dist, mean='AR')
            .fit(disp='off'))



# Bitcoin Volatility
plt.plot((egarch_model(btc_d.query("Date > '2015-03-15'")['Close']).conditional_volatility))
plt.title("BTC Volatility Over Time")
plt.ylabel('Conditional Volatility')

# Stock Market Volatility
plt.plot((egarch_model(sp.query("Date > '2015-03-15'")['Close']).conditional_volatility))
plt.title("S&P 500 Volatility Over Time")
plt.ylabel('Conditional Volatility')

# Gold Volatility
plt.plot((egarch_model(gold.query("Date > '2015-03-15'")['Close']).conditional_volatility))
plt.title("Gold Volatility Over Time")
plt.ylabel('Conditional Volatility')




# fed meetings, rate hikes, and hike probabilities
interest_rate_hikes = ['2017-06-14', '2017-12-13']
fomc_meetings = ['2017-05-03', '2017-06-14',
'2017-07-26', '2017-09-20', '2017-11-01', '2017-12-13']
fomc_meetings.sort()

# rate changes will only be a shock if they are unexpected
# look at market's expectations for interest rates the day before FOMC announcements
q = fed_rate_probs.loc[pd.to_datetime(fomc_meetings) - timedelta(days=1)]
q[['(75-100)', '(100-125)', '(125-150)',
'(150-175)', '(175-200)']].fillna(0)

