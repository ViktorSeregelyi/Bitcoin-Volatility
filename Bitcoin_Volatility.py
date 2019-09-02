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

# https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/
# https://www.quandl.com/collections/markets/gold
# http://www.cmegroup.com/trading/interest-rates/countdown-to-fomc.html/
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


fomc_shock_df = pd.DataFrame(columns=['Date', 'Smoothness Shock Ratio',
                                      'Return SD Shock Ratio',
                                      'Interest Rate Hike',
                                      'Predicted Prob of Increase'])

prob_increase = {'2017-05-03': 89.9, '2017-06-14': 99.6,
                 '2017-07-26': 52.7, '2017-09-20': 57.7,
                 '2017-11-01': 92.8, '2017-12-13': 100.00}

# s&p 500 volatility comparison for the weeks immediately prior to and following FOMC meetings, two measures
for d in fomc_meetings:
    pre, post = pre_post_shock(sp, d, window=7, vol_fn=smoothness)
    pre2, post2 = pre_post_shock(sp, d, window=7, vol_fn=return_std)
    fomc_shock_df.loc[len(fomc_shock_df)] = [d, post/pre, post2/pre2,
                                             ('Yes' if d in interest_rate_hikes else 'No'),
                                             prob_increase[d]]
fomc_shock_df


# https://99bitcoins.com/price-chart-history/
# Pre and post weekly volatility for different bitcoin-specific shocks
bitcoin_events = ['2017-08-01', '2017-09-15', '2017-10-25','2017-11-08', '2017-12-11', '2017-12-28']
event_type = ['Negative', 'Negative', 'Negative',
              'Negative', 'Positive', 'Negative']

bitcoin_events.sort()
bitcoin_events = pd.Series([pd.to_datetime(b) for b in bitcoin_events])
btc_volatility = pd.DataFrame(columns=['Date', 'Smoothness Shock Ratio',
                                       'Return SD Shock Ratio', 'Shock Type'])
for i, d in enumerate(bitcoin_events):
    pre, post = pre_post_shock(btc_d, d, 'Close', window=7, vol_fn=smoothness)
    pre2, post2 = pre_post_shock(btc_d, d, 'Close', window=7, vol_fn=return_std)
    btc_volatility.loc[len(btc_volatility)] = [d, post/pre,
                                               post2/pre2, event_type[i]]
btc_volatility


egarch_model(sp.query("'2001-01-01' < Date < '2018-03-15'")['Close'])

egarch_model(gold.query("'2006-01-01' < Date < '2018-03-15'")['Close'])

egarch_model(btc_d.query("'2015-03-15' < Date < '2018-03-15'")['Close'])

# We can also use a univariate EGARCH model to capture the asymmetric effect of shocks to BTC on volatility
def asymm_egarch(data, dist='StudentsT'):
    return (arch.arch_model(data.apply(np.log).diff()[1:],
                            vol='EGARCH', p=1, o=1, q=1, dist=dist, mean='AR')
            .fit(disp='off'))


asymm_egarch(btc_d.query("'2015-03-15' < Date < '2017-06-15'")['Close'])


# we see a structural break between these two periods

asymm_egarch(btc_d.query("'2017-06-15' < Date < '2018-03-15'")['Close'])


# From 2015-03-15 to 2017-06-15, Bitcoin may have acted like gold, providing a safe haven from 
# uncertainty in other markets. During this time, positive shocks were associated with more volatility
# in the Bitcoin market than negative shocks of the same magnitude. This relationship reversed in the 
# following period of 2017-06-15 to 2018-03-15, and Bitcoin began acting more like an equity,
# demonstrating a greater susceptibility to volatility caused by negative news shocks.

