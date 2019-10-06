from datetime import date, timedelta
import sys
sys.path.append("/Users/ignacio/workspace/Projects/stockBuyer/src")
# from stock import Stock
# from utils import getPreviousWeekDay, prPurple, prYellow

# tick_name = "SCHW"
# ticker = Stock(tick_name, "")
# day = getPreviousWeekDay(date.today())
# tomorrow = day + timedelta(days=1)
# temp_hist = ticker.getData(day, tomorrow).reset_index()
# index = 50
# start_index = index - 30
# temp_len = len(temp_hist[start_index:index]["Close"])
# temp_average = sum(temp_hist[start_index:index]["Close"])/temp_len
# angle = ticker.getAngle(temp_hist[start_index:index]["Close"].iloc(0)[0], temp_average)
# print(angle)
"""
16 - 2: Check this... 
how to prevent when distance between real stock and mavg is too big?
BUYING 1 21.62 10:31:00 60
SELLING 1 21.48 11:18:00 107
NBL -0.6475485661424614 MAX: 0.6012950971322795 63
"""

import pandas as pd
import yfinance as yf
import numpy as np
from rdp import rdp
from _plotly_future_ import v4_subplots
import plotly.offline as py
from plotly.figure_factory import create_candlestick
from plotly.graph_objs import *
from plotly.subplots import make_subplots


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def getSignals(stock, short_window=30, long_window=60, debug=False):
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # Create short simple moving average over the short window
    signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    # return pd.concat([stock, signals])
    return signals

def relative_strength_index2(dataframe, n):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    df = dataframe.copy()
    df = df.reset_index()
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    rsi = PosDI / (PosDI + NegDI)
    dataframe['RSI'] = rsi.values
    return dataframe

def relative_strength_index(stock, period):
    series = stock.Close
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / \
         d.ewm(com=period-1, adjust=False).mean()
    stock['RSI'] = 100 - 100 / (1 + rs)
    return stock

def stochastic_oscillator_k(df):
    """Calculate stochastic oscillator %K for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    df = df.join(SOk)
    return df

def stochastic_oscillator_d(df, n):
    """Calculate stochastic oscillator %D for given data.
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    # SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SOK')
    # SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SOD')
    # SOd = pd.Series(SOk.rolling(window=n).mean(), name='SOD')
    # df = df.join(SOd)
    low_min  = df['Low'].rolling( window = n ).min()
    high_max = df['High'].rolling( window = n ).max()

    # Fast Stochastic
    df['k_fast'] = 100 * (df['Close'] - low_min)/(high_max - low_min)
    df['d_fast'] = df['k_fast'].rolling(window = 3).mean()

    # Slow Stochastic
    df['k_slow'] = df["d_fast"]
    df['SOD'] = df['k_slow'].rolling(window = 3).mean()
    return df

def stochastics( dataframe, k, d ):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal 
    When the %K crosses below %D, sell signal
    """

    df = dataframe.copy()

    # Set minimum low and maximum high of the k stoch
    low_min  = df['Low'].rolling( window = k ).min()
    high_max = df['High'].rolling( window = k ).max()

    # Fast Stochastic
    df['k_fast'] = 100 * (df['Close'] - low_min)/(high_max - low_min)
    df['d_fast'] = df['k_fast'].rolling(window = d).mean()

    # Slow Stochastic
    df['k_slow'] = df["d_fast"]
    df['d_slow'] = df['k_slow'].rolling(window = d).mean()

    return df

def getPercent(stock_buy, stock_sell):
    return (stock_sell * 100 /  stock_buy) - 100

import utils

def action(stock_name, day=None, short_w=30, long_w=60, debug=False):
    # Initialize the `signals` DataFrame with the `signal` column
    if day is None:
        today = date.today() - timedelta(days=2)
        day = utils.getPreviousWeekDay(today)
    tomorrow = day + timedelta(days=1)
    appl = yf.Ticker(stock_name).history(period='1d', start=day,end=tomorrow, interval='1m')
    buy = 0.0
    sel = 0.0
    total_percent = 0.0
    prev_s = 0.0
    END_TIME = "15:00"
    possible_buy = None
    possible_buy_wait = -1
    for i in range(30, len(appl['Close'])):
        stock = appl[:i].copy()
        signals = getSignals(stock, short_w, long_w)
        stock = stochastic_oscillator_d(stock, 14)
        index = i-1
        s = signals['positions'][index]
        current_close = stock['Close'][index]
        current_time = stock.index[index].strftime("%H:%M")
        # stock_trend = 1.0 if stock['SOD'][index-1] < stock['SOD'][index] else -1.0
        # sod_value = stock['SOD'][index]
        # sod_trend = (sod_value > 80 and s == -1.0) or (sod_value < 20 and s == 1.0)
        # stock = relative_strength_index(stock, 14)
        rsi_value = stock['RSI'][index]
        rsi_trend = (rsi_value > 70 and s == -1.0) or (rsi_value < 30 and s == 1.0) 
        # if stock_trend == s:

        if current_time > END_TIME:
            if buy != 0.0:
                sel = current_close
                percent = getPercent(buy, sel) if prev_s == 1.0 else getPercent(sel, buy)
                total_percent += percent
                if debug: print("Selling LAST", sel, buy, percent, index, current_time)
            break

        if possible_buy_wait >= 0:
            possible_buy_wait += 1

        if buy != 0:
            sel = current_close
            percent = getPercent(buy, sel) if prev_s == 1.0 else getPercent(sel, buy)
            if percent > 0.5:
                total_percent += percent
                if debug: print("Selling" if prev_s == 1.0  else "Buying", sel, "BUY", buy, percent, index, current_time)
                buy = 0.0
                sel = 0.0
                possible_buy_wait = 0

        if s != 0.0:
            if buy != 0.0:
                sel = current_close
                percent = getPercent(buy, sel) if prev_s == 1.0 else getPercent(sel, buy)
                total_percent += percent
                if debug: print("Selling" if prev_s == 1.0  else "Buying", sel, "BUY", buy, percent, index, current_time)
                buy = 0.0
                sel = 0.0
            if debug: print("HIT FOUND", current_time)
            possible_buy = current_close
            possible_buy_wait = 0
            prev_s = s
        
        if possible_buy_wait == 4:
            in_trend = True if ((possible_buy < current_close and prev_s == 1.0) or (possible_buy > current_close and prev_s == -1.0)) else False
            if in_trend:
                if debug: print("BUYING" if prev_s == 1.0 else "SELLING", prev_s, current_close, index, current_time)
                buy = current_close
            possible_buy_wait = -1

    return total_percent, 0.0


def action2(stock_name, day=None, short_w=30, long_w=60, debug=False):
    # Initialize the `signals` DataFrame with the `signal` column
    if day is None:
        today = date.today() - timedelta(days=2)
        day = utils.getPreviousWeekDay(today)
    tomorrow = day + timedelta(days=1)
    appl = yf.Ticker(stock_name).history(period='1d', start=day,end=tomorrow, interval='1m')
    buy = 0.0
    sel = 0.0
    total_percent = 0.0
    prev_s = 0.0
    END_TIME = "15:00"
    for i in range(30, len(appl['Close'])):
        stock = appl[:i].copy()
        signals = getSignals(stock, short_w, long_w)
        stock = stochastic_oscillator_d(stock, 14)
        index = i-1
        s = signals['positions'][index]
        current_close = stock['Close'][index]
        current_time = stock.index[index].strftime("%H:%M")
        # stock_trend = 1.0 if stock['SOD'][index-1] < stock['SOD'][index] else -1.0
        sod_value = stock['SOD'][index]
        sod_trend = (sod_value > 80 and s == -1.0) or (sod_value < 20 and s == 1.0)
        stock = relative_strength_index(stock, 14)
        rsi_value = stock['RSI'][index]
        rsi_trend = (rsi_value > 70 and s == -1.0) or (rsi_value < 30 and s == 1.0) 
        # if stock_trend == s:
        if s != 0.0:
            if buy != 0.0:
                sel = current_close
                percent = getPercent(buy, sel) if prev_s == -1.0 else getPercent(sel, buy)
                total_percent += percent
                if debug: print("Selling" if prev_s == -1.0  else "Buying", sel, "BUY", buy, percent, index, current_time)
            if rsi_trend and sod_trend:
                if debug: print("BUYING" if s == -1.0 else "SELLING", s, current_close, index, current_time)
                if debug: print("STOCKA", stock['SOD'][index-1], stock['SOD'][index])
                if debug: print("RSI", rsi_value, s)
                buy = current_close
                prev_s = s
        if current_time > END_TIME and buy != 0.0:
            sel = current_close
            percent = getPercent(buy, sel) if prev_s == -1.0 else getPercent(sel, buy)
            total_percent += percent
            if debug: print("Selling LAST", sel, buy, percent, index, current_time)
            break
    return total_percent, 0.0

import json

all_n_stocks = []
all_p_stocks = []

def finishedRelaxed(stock_name, day):
    yesterday = day - timedelta(days=1)
    day = utils.getPreviousWeekDay(yesterday)
    tomorrow = day + timedelta(days=1)
    data = yf.Ticker(stock_name).history(period='1d', start=day,end=tomorrow, interval='1m')
    last_2_hours_index = int(len(data) - (60/1.0))
    temp_hist = data[last_2_hours_index:len(data)]
    deviation = temp_hist['Close'].std()
    if deviation < 0.015:
        print("RELAXED:", stock_name)
        return True
    return False

def getYesterdaySignalsLength(stock_name, day):
    yesterday = day - timedelta(days=1)
    day = utils.getPreviousWeekDay(yesterday)
    tomorrow = day + timedelta(days=1)
    data = yf.Ticker(stock_name).history(period='1d', start=day,end=tomorrow, interval='1m')
    signals = getSignals(data)
    return len(signals.loc[signals.positions == 1.0] + signals.loc[signals.positions == -1.0])

def main():
    TICKERS_PATH = "/Users/ignacio/workspace/Projects/stockBuyer/data/tickers.json"
    with open(TICKERS_PATH) as json_file:  
        data = json.load(json_file)
    stocks = [v for s in data for v in data[s]]
    #stocks = ['ETR', 'LKQ', 'KIM', 'ATO', 'MAC', 'EXC', 'DLTR', 'CNP', 'WDC', 'ADI', 'PEG', 'CBOE', 'CME', 'FE', 'PBCT', 'ZBH', 'PWR', 'ABBV', 'HD', 'ED', 'WCG', 'EXPD', 'DAL', 'KHC', 'ETFC', 'BAC', 'STI', 'BXP', 'ZION', 'NUE', 'DG', 'EVRG', 'NTRS', 'PPL', 'EMR', 'AAPL', 'BBT', 'MXIM', 'FL', 'SEE', 'SO']
    #BEST short 30!
    # stocks = [ 'HES', 'CFG', 'HII', 'MTB', 'SEE', 'TXN', 'PNC', 'CL', 'PBCT', 'HP', 'ALL', 'RCL', 'PPL', 'ABBV', 'INCY', 'PG', 'VLO', 'STI', 'MCHP', 'APH', 'SCHW', 'NUE', 'AAPL', 'AON', 'ADBE', 'DAL', 'STZ', 'DG', 'NTRS', 'KIM', 'PFE', 'AMD', 'JCI', 'TEL', 'CMI', 'EXPD', 'HD', 'CBOE', 'KLAC', 'WCG', 'EMR' ]
    # stocks = ['BBY', 'WMT', 'COF', 'APTV', 'RMD', 'LNC', 'DFS', 'SPGI', 'TAP', 'OMC', 'XEL', 'TRIP', 'BBT', 'OKE', 'CCI', 'RJF', 'ICE', 'CNP', 'GOOG', 'NWSA', 'IFF', 'AME', 'JEC', 'XRAY', 'XLNX', 'EL', 'SLG', 'CAH', 'SCHW', 'ALB', 'LW', 'PKI', 'CMCSA', 'FCX', 'RE', 'FE', 'GPS', 'ADP', 'FISV', 'DUK']
    short_range = [30]
    long_range = [60]
    results = {}
    days = [1]
    for day_num in days: 
        today = date.today() - timedelta(days=day_num)
        day = utils.getPreviousWeekDay(today)
        print("DAY:", day)
        for sr in short_range:
            for lr in long_range:
                percent = 0
                max_percent = 0
                wrong_stocks = 0
                negative_stocks = []
                positive_stocks = []
                signals_dict = {}
                # printProgressBar(0, len(stocks), prefix = 'Progress:', suffix = 'Complete', length = 50)
                for i, stock in enumerate(stocks):
                    try:
                        tmp_percent, tmp_max_percent = action(stock, day, sr, lr, debug=False)
                        if tmp_percent == -100.0:
                            wrong_stocks += 1
                            continue
                        if tmp_percent <= 0:
                            negative_stocks.append(stock)
                        else:
                            positive_stocks.append(stock)
                        if tmp_percent: print(stock, tmp_percent)
                        percent += tmp_percent
                        max_percent += tmp_max_percent
                    except Exception as e:
                        print("FAIL", e)
                        wrong_stocks += 1
                        continue
                    printProgressBar(i, len(stocks), prefix = 'Progress:', suffix = 'Complete', length = 50)
                print("RANGE:", sr, lr)
                print("TOTAL PERCENT", percent,"MAX:", max_percent, "for", len(stocks) - wrong_stocks)
                print("TOTAL POSITIVE", len(positive_stocks))
                print("TOTAL NEGATIVE", len(negative_stocks))
                print("")
                all_n_stocks.append(negative_stocks)
                all_p_stocks.append(positive_stocks)

# Import `pyplot` module as `plt`
import matplotlib.pyplot as plt
import os

def plot1(stock_name, day=None):
    if day is None:
        day = date.today() - timedelta(days=3)
    tomorrow = day + timedelta(days=1)
    stock = yf.Ticker(stock_name).history(period='1d', start=day, end=tomorrow, interval='1m')
    #stock = stock[:160]
    stock = stochastic_oscillator_d(stock, 14)
    signals = getSignals(stock, debug=True)
    stock = relative_strength_index(stock, )
    # Initialize the plot figure
    fig = plt.figure()

    # Add a subplot and label for y-axis
    ax1 = fig.add_subplot(111,  ylabel='Price in $')
    ax2 = fig.add_subplot(515,  ylabel='Price in $')

    # Plot the closing price
    stock['Close'].plot(ax=ax1, color='r', lw=2.)
    stock['SOD'].plot(ax=ax2, color='r', lw=2.)

    # Plot the short and long moving averages
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the buy signals
    ax1.plot(signals.loc[signals.positions == 1.0].index, 
            signals.short_mavg[signals.positions == 1.0],
            '^', markersize=10, color='m')
    # tmp_signals = stock.reset_index()
    # epsilon = 0.1
    # rdp_points = rdp(np.column_stack((tmp_signals.index, stock.Close)), epsilon=epsilon, return_mask=False)
    # print("RPD:", len(rdp_points))
    # rdp_mask = rdp(np.column_stack((tmp_signals.index, stock.Close)), epsilon=epsilon, return_mask=True)
    # signals['fit'] = np.where(rdp_mask, stock['Close'], np.nan)
    # Plot the sell signals
    # ax1.plot(signals['fit'], 'x')
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
            signals.short_mavg[signals.positions == -1.0],
            'v', markersize=10, color='k')
    # Show the plot
    plt.show()

def plot(stock_name, day=None):
    if day is None:
        day = date.today() - timedelta(days=3)
    tomorrow = day + timedelta(days=1)
    stock = yf.Ticker(stock_name).history(period='1d', start=day, end=tomorrow, interval='1m')
    # stock = stock[:160]
    stock = stochastic_oscillator_d(stock, 14)
    signals = getSignals(stock, debug=True)
    stock = relative_strength_index(stock, 14)
    shapes = []
    data = [ dict(
        type = 'candlestick',
        open = stock.Open,
        high = stock.High,
        low = stock.Low,
        close = stock.Close,
        x = stock.index.astype('str'),
        yaxis = 'y2',
        name = stock_name,
    ) ]
    s_mavg_scatter = dict(
        # x=stock.Datetime,
        x = stock.index.astype('str'),
        y=signals['short_mavg'],
        name= 'short_mavg',
        line=scatter.Line(color='olive'),
        type='scatter',
        yaxis='y2',
        # text=angles
        )
    data.append(s_mavg_scatter)
    l_mavg_scatter = dict(
        # x=stock.Datetime,
        x = stock.index.astype('str'),
        y=signals['long_mavg'],
        name= 'long_mavg',
        line=scatter.Line(color='violet'),
        type='scatter',
        yaxis='y2',
        # text=angles
        )
    data.append(l_mavg_scatter)
    buy_markers = dict(
        x=signals.loc[signals.positions == 1.0].index.astype('str'), 
        y=stock.Close[signals.positions == 1.0],
        type='scatter',
        mode='markers',
        name="Buy",
        yaxis = 'y2',
        marker=dict(
            size=15,
            color='green',
            symbol='triangle-up',
            line=dict(
                color='black',
                width=2
            )
            )
        )
    data.append(buy_markers)
    sell_markers = dict(
        x=signals.loc[signals.positions == -1.0].index.astype('str'), 
        y=stock.Close[signals.positions == -1.0],
        type='scatter',
        mode='markers',
        name="Sell",
        yaxis = 'y2',
        marker=dict(
            size=15,
            color='red',
            symbol='triangle-down',
            line=dict(
                color='black',
                width=2
            )
            )
        )
    data.append(sell_markers)
    #######################
    # SOD
    sod = dict(
        name="SOD",
        x=stock.index.astype('str'),
        y=stock['SOD'],
        type='scatter',
        mode='lines',
        yaxis = 'y1',
    )
    data.append(sod)
    shapes.extend([
        dict(
            type="line",
            xref="x1",
            yref="y1",
            x0=stock.index.astype('str')[0],
            y0=80,
            x1=stock.index.astype('str')[-1],
            y1=80,
            line=dict(
                color="LightSeaGreen",
                width=1,
                ),
        ),
        dict(
            type="line",
            xref="x1",
            yref="y1",
            x0=stock.index.astype('str')[0],
            y0=20,
            x1=stock.index.astype('str')[-1],
            y1=20,
            line=dict(
                color="LightSeaGreen",
                width=1,
                ),
        )
    ])
    #######################
    # RSI
    rsi = dict(
        name="RSI",
        x=stock.index.astype('str'),
        y=stock['RSI'],
        type='scatter',
        mode='lines',
        yaxis = 'y3',
    )
    data.append(rsi)
    shapes.extend([
        dict(
            type="line",
            xref="x1",
            yref="y3",
            x0=stock.index.astype('str')[0],
            y0=70,
            x1=stock.index.astype('str')[-1],
            y1=70,
            line=dict(
                color="LightSeaGreen",
                width=1,
                ),
        ),
        dict(
            type="line",
            xref="x1",
            yref="y3",
            x0=stock.index.astype('str')[0],
            y0=30,
            x1=stock.index.astype('str')[-1],
            y1=30,
            line=dict(
                color="LightSeaGreen",
                width=1,
                ),
        )
    ])

    plots_path = "/Users/ignacio/workspace/Projects/stockBuyer/data/plots/{day}/{stock}.html"
    path = plots_path.format(day="test", stock=stock_name)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    layout = dict(
        margin=dict(l=120, r=20, t=20, b=20),
        yaxis={"domain": [0, 0.2]},
        yaxis3={"domain": [0.2, 0.4]},
        yaxis2={"domain": [0.4, 1.0]},
        shapes=shapes
    )
    fig = dict( data=data, layout=layout )
    py.plot(fig, filename=path, validate=False)
stock = 'NBL'
#stock = 'PXD'
# stock = 'MPC'
#stock = 'FTI'
#stock = 'PSX'
# stock = 'MMM'
stock = 'ROL'

# main()

today = date.today() - timedelta(days=1)
day = utils.getPreviousWeekDay(today)
print("DAY:", day)
result, _ = action(stock, day=day, debug=True)
print("RESULT", result)
plot(stock, day=day)
"""
14 Aug 2019

RANGE: 20 60
TOTAL PERCENT -28.61646194563238 MAX: 176.46724478513283 for 493
TOTAL POSITIVE 137
TOTAL NEGATIVE 356

RANGE: 20 65
TOTAL PERCENT -33.97726175675935 MAX: 179.96781402741664 for 493
TOTAL POSITIVE 132
TOTAL NEGATIVE 361

RANGE: 20 70
TOTAL PERCENT -26.39882212832076 MAX: 189.18284843711677 for 493
TOTAL POSITIVE 134
TOTAL NEGATIVE 359

RANGE: 20 75
TOTAL PERCENT -31.916671353170585 MAX: 196.32183093376557 for 493
TOTAL POSITIVE 128
TOTAL NEGATIVE 365

RANGE: 20 80
TOTAL PERCENT -34.98582800394091 MAX: 201.21561530185636 for 493
TOTAL POSITIVE 125
TOTAL NEGATIVE 368

RANGE: 20 85
TOTAL PERCENT -39.93592000015839 MAX: 201.23794189043144 for 493
TOTAL POSITIVE 125
TOTAL NEGATIVE 368

RANGE: 20 90
TOTAL PERCENT -35.37062025296518 MAX: 211.6441077474201 for 493
TOTAL POSITIVE 124
TOTAL NEGATIVE 369

RANGE: 20 95
TOTAL PERCENT -43.8232827139706 MAX: 205.97821108906538 for 492
TOTAL POSITIVE 114
TOTAL NEGATIVE 378

RANGE: 20 100
TOTAL PERCENT -38.00279311833515 MAX: 220.82424900526644 for 492
TOTAL POSITIVE 118
TOTAL NEGATIVE 374

RANGE: 25 60
TOTAL PERCENT 0.08619156476629541 MAX: 208.114420791748 for 493
TOTAL POSITIVE 172
TOTAL NEGATIVE 321

RANGE: 25 65
TOTAL PERCENT -2.1987292673660193 MAX: 211.06481261539028 for 493
TOTAL POSITIVE 159
TOTAL NEGATIVE 334

RANGE: 25 70
TOTAL PERCENT -5.05347448211549 MAX: 222.54792618118512 for 493
TOTAL POSITIVE 168
TOTAL NEGATIVE 325

RANGE: 25 75
TOTAL PERCENT -12.509771219885891 MAX: 226.79115743653136 for 493
TOTAL POSITIVE 154
TOTAL NEGATIVE 339

RANGE: 25 80
TOTAL PERCENT -13.00627571547615 MAX: 233.4612063488567 for 493
TOTAL POSITIVE 150
TOTAL NEGATIVE 343

RANGE: 25 85
TOTAL PERCENT -10.15206267857333 MAX: 245.47683998263403 for 493
TOTAL POSITIVE 146
TOTAL NEGATIVE 347

RANGE: 25 90
TOTAL PERCENT -13.227510263380879 MAX: 249.37061887345004 for 493
TOTAL POSITIVE 148
TOTAL NEGATIVE 345

RANGE: 25 95
TOTAL PERCENT -19.100401650025248 MAX: 240.02707261108418 for 492
TOTAL POSITIVE 143
TOTAL NEGATIVE 349

RANGE: 25 100
TOTAL PERCENT -22.532203009331397 MAX: 242.40908990402664 for 492
TOTAL POSITIVE 149
TOTAL NEGATIVE 343

RANGE: 30 60
TOTAL PERCENT 19.013518084887266 MAX: 223.0008121282354 for 493
TOTAL POSITIVE 184
TOTAL NEGATIVE 309

RANGE: 30 65
TOTAL PERCENT 16.23386867442649 MAX: 230.4826218175899 for 493##################################################
TOTAL POSITIVE 181
TOTAL NEGATIVE 312

RANGE: 30 70
TOTAL PERCENT 17.05118472909301 MAX: 238.19489970215065 for 493
TOTAL POSITIVE 187
TOTAL NEGATIVE 306

RANGE: 30 75
TOTAL PERCENT 7.334505787805696 MAX: 239.4265093589559 for 493
TOTAL POSITIVE 179
TOTAL NEGATIVE 314

######################
#################################################################
# #
DAY: 2019-08-21
RANGE: 30 60################################################-| 99.8% Complete
TOTAL PERCENT 71.39179391420876 MAX: 0.0 for 493
TOTAL POSITIVE 295
TOTAL NEGATIVE 198

DAY: 2019-08-20
RANGE: 30 60################################################-| 99.8% Complete
TOTAL PERCENT 101.09409955938199 MAX: 0.0 for 493
TOTAL POSITIVE 323
TOTAL NEGATIVE 170

DAY: 2019-08-19
RANGE: 30 60################################################-| 99.8% Complete
TOTAL PERCENT -113.66409988046745 MAX: 0.0 for 493
TOTAL POSITIVE 159
TOTAL NEGATIVE 334

DAY: 2019-08-16
RANGE: 30 60################################################-| 99.8% Complete
TOTAL PERCENT -316.65379575133704 MAX: 0.0 for 493
TOTAL POSITIVE 82
TOTAL NEGATIVE 411

DAY: 2019-08-16
RANGE: 30 60################################################-| 99.8% Complete
TOTAL PERCENT -316.2314517502963 MAX: 0.0 for 493
TOTAL POSITIVE 85
TOTAL NEGATIVE 408

#######################################################################
"""