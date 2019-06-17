# -*- coding: utf-8 -*-
"""
TODO:
- Define better 3 bar rules
- How to detect which stocks to buy/sell?
- Define rules to stop
    - Stop should change over time.
    - If Stock goes up.. increase stop.
- What to do if the market is about to close?
    And we have stocks!

- Dynamic stop seems to fail more :/
- MAKE DYNAMIC WORK!
- MAYBE CHECK deviation on todays market too
PINOCHIO SEEMS TO WORK!
UPDATE SELL PROCESS BECAUSE IS NOT DOING THE RIGHT THING.

Mala lectura del trend en algunos casos!
 Pinochio F 2019-06-05 09:55 9.7 True 0.009999999999999787 145851 False
 USING LAST F 2019-06-05 15:59 9.77 False 0.0 635097 9.7 1
PERCENT: -0.7164790174002178
 LOOSER! -23.710248093869268 F 2019-06-05 09:55 9.7 True 0.009999999999999787 145851 9.77 25
Tal vez hay algun patron que detecte el cambio de direccion.
Ver si corriendo el offset en MA se arregla!
Mejorar Stop! 
Algunas veces no llega a bajar un 1% y despues sube..
tal vez hacer dynamico y que actualize cuando llega a 0.5%
"""
#%%
from urllib.request import urlopen, Request
import pytz

from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import yfinance as yf
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def prRed(message, bar, extra=[]): printf("\033[91m {}\033[00m", message, bar, extra) 
def prGreen(message, bar, extra=[]): printf("\033[92m {}\033[00m", message, bar, extra) 
def prYellow(message, bar, extra=[]): printf("\033[93m {}\033[00m", message, bar, extra ) 
def prLightPurple(message, bar, extra=[]): printf("\033[94m {}\033[00m", message, bar, extra) 
def prPurple(message, bar, extra=[]): printf("\033[95m {}\033[00m", message, bar, extra) 
def prCyan(message, bar, extra=[]): printf("\033[96m {}\033[00m", message, bar, extra) 
def prLightGray(message, bar, extra=[]): printf("\033[97m {}\033[00m", message, bar, extra) 
def prBlack(message, bar, extra=[]): printf("\033[98m {}\033[00m", message, bar, extra) 

def printf(color, message, bar, extra=[]):
    msg = message
    msg += " "
    msg += " ".join([str(s) for s in [bar.name, str(bar.date)[:-9], bar.close, bar.isUp, bar.height, bar.volume, bar.index]])
    msg += " "
    msg += " ".join([str(s) for s in extra])
    print(color.format(msg))

#%%
SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START = datetime(1900, 1, 1, 0, 0, 0, 0, pytz.utc)
END = datetime.today().utcnow()

#%%
def scrape_list(site):
    """Get dict of tickers organized by sector."""
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = Request(site, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page)

    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip()).lower().replace(' ', '_')
            ticker = str(col[0].find('a').string.strip())
            if sector not in sector_tickers:
                sector_tickers[sector] = list()
            sector_tickers[sector].append(ticker)
    return sector_tickers

#%%
tickers = scrape_list(SITE)

#%%
# Fee for converting from pound to dollar.
CONVERSION_FEE = 0.005
# Fee for spread (Learn what is this)
SPREAD_FEE = 0.09
DOLLAR_TO_POUND = 0.7522

investment = 2000 #Pounds
pound_to_dollar = 1.0/DOLLAR_TO_POUND
investment_to_dollars = investment * 1.0/DOLLAR_TO_POUND
investment_after_conversion = investment*(pound_to_dollar-CONVERSION_FEE)
investment = investment_after_conversion
stock_buy = 68.65
stock_sell = 72.64

def getGain(investment, stock_buy, stock_sell):
    inv_with_spread = investment - getPercent(investment, SPREAD_FEE)
    percent = (stock_sell * 100 /  stock_buy) - 100
    gain = inv_with_spread + getPercent(inv_with_spread, percent)
    print("PERCENT:", percent)
    gain_with_spread = gain + (-1.0*getPercent(gain, SPREAD_FEE))
    # print(gain_with_spread)
    return gain_with_spread - investment

def getPercent(value, percent):
    return value*percent/100.0

def getGainPercent(investment, percent):
    inv_with_spread = investment - getPercent(investment, SPREAD_FEE)
    gain = inv_with_spread + getPercent(inv_with_spread, percent)

    gain_with_spread = gain + (-1.0*getPercent(gain, SPREAD_FEE))
    # print(gain_with_spread)
    return gain_with_spread - investment
#%%
def getGlobalGain():
    inves = investment
    for i in range(25*12):
        inves += getGainPercent(inves, 1.0)
        inves += getGainPercent(inves, 1.0)

    current_percent = (inves * 100 /  investment) - 100
    print(investment, inves, current_percent)
getGlobalGain()

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
#%%
class Bar(object):
    height_difference = 2.0
    def __init__(self, name, stock_data, average_height=None, average_volume=None):
        #print(stock_data['Open'][0])
        self.name = name
        self._stock_data = stock_data
        self._average_height = average_height
        self._average_volume = average_volume
    
    def isTall(self, other=None):
        # Maybe this is not the best approach
        # Maybe compare to previous bar if it is 3 times taller.
        if other:
            return other.height * self.height_difference <= self.height
        return self.average_height * self.height_difference <= self.height

    def isShort(self, other=None):
        if other:
            return self.absoluteHeight <= other.height / 1.5
        # TODO: check with average better
        return self.average_height * self.height_difference / 1.5 >= self.height

    def valueInRange(self, value):
        if self.isUp:
            return self.open <= value <= self.close
        return self.close <= value <= self.open

    @property
    def average_height(self):
        return self._average_height

    @property
    def average_volume(self):
        return self._average_volume

    @property
    def height(self):
        return self._stock_data['Height'].iloc[0]

    @property
    def open(self):
        return self._stock_data['Open'].iloc[0]

    @property
    def absoluteHeight(self):
        return abs(self.high - self.low)

    @property
    def close(self):
        return self._stock_data['Close'].iloc[0]

    @property
    def high(self):
        return self._stock_data['High'].iloc[0]

    @property
    def low(self):
        return self._stock_data['Low'].iloc[0]

    @property
    def volume(self):
        return self._stock_data['Volume'].iloc[0]

    @property
    def date(self):
        date = self._stock_data['Datetime'].iloc[0]
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
        return date

    @property
    def MA(self):
        return self._stock_data['MA'].iloc[0]

    @property
    def index(self):
        return int(self._stock_data.index[0])

    @property
    def time(self):
        return self.date.strftime("%H:%M")

    @property
    def isUp(self):
        return self._stock_data['IsUp'].iloc[0]

    def inRange(self, high_value, low_value):
        return self.high < high_value and self.low > low_value

    def isSimilarHeight(self, other):
        epsilon = self.average_height / 3.0
        difference = abs(self.height - other.height)
        return difference <= epsilon

    def isNumberSimilar(self, value, other):
        epsilon = self.average_height / 3.0
        return abs(other - value) <= epsilon

    def isGap(self, first):
        if self.isUp:
            return self.close < first.close
        else:
            return self.close > first.close


    ################################ 
    # PATTERNS
    def isReversal(self, first, trend):
        """
            A bullish reversal bar pattern goes below the low 
                of the previous bar before closing higher.
            A bearish reversal bar pattern goes above the high 
                of the last bar before closing lower.

        When to buy/sell:
            - Buy above the bullish reversal bar in an uptrend
            - Sell below the bearish reversal bar in a downtrend

        Args:
            first (Bar): Previous bar to compare.
            trend (bool): True if its going UP, False if DOWN.
        """
        b_trend = trend > 0
        if self.isUp == first.isUp:
            return False
        if self.isUp and b_trend:
            if not self.isNumberSimilar(self.open, first.close):
                return False
            if self.low <= first.low:
                prGreen("Reversal", self, [trend])
                return True
        elif not self.isUp and not b_trend:
            if not self.isNumberSimilar(self.close, first.open):
                return False
            if self.high >= first.high:
                prGreen("Reversal", self, [trend])
                return True
        return False

    def isKeyReversal(self, first):
        pass

    def isExhaustion(self, first):
        """
        A bullish exhaustion bar opens with a gap down.
            Then, it works its way up to close near its top.
        A bearish exhaustion bar opens with a gap up before
            moving down to close as a bearish bar.
        
        In both cases, the gap remains unfilled.
        Also, high volume should occur with the exhaustion bar.

        When to buy/sell:
            - Buy above a bullish exhaustion bar
            - Sell below a bearish exhaustion bar

        Args:
            first (Bar): Previous bar to compare.
        """
        if not self.volume > self.average_volume * 3:
            return False
        if self.isUp == first.isUp:
            return False
        if not self.isGap(first):
            return False
        prGreen("Exhaustion", self)
        return True

    def isPinochio(self, first, trend):
        """
        For bullish pin bars, the lower tail takes up most of the bar.
        For bearish pin bars, it is the upper tail that dominates.
        """
        b_trend = trend
        if self.height == 0.0:
            return False
        if b_trend:
            # UP with resistance to go DOWN.
            if self.isUp and (self.low - self.open) > self.height * 2:
                if first.low < self.low and first.high >= self.high:
                    prGreen("Pinochio", self, [trend])
                    return True
            elif not self.isUp and (self.low - self.close) > self.height * 2:
                if first.low < self.low and first.high >= self.high:
                    prGreen("Pinochio", self, [trend])
                    return True
        else:
            # UP with resistance to continue UP
            if self.isUp and (self.high - self.close) > self.height * 2:
                if first.high < self.high and first.low <= self.low:
                    prGreen("Pinochio", self, [trend])
                    return True
            elif not self.isUp and (self.high - self.open) > self.height * 2:
                if first.high < self.high and first.low <= self.low:
                    prGreen("Pinochio", self, [trend])
                    return True
        return False

    def isTwoBarReversal(self, first):
        pass

    def isThreeBarReversal(self, first, second):
        pass

    def isThreeBarPullBack(self, first, second, third):
        pass

    def isInside(self, first):
        pass

    def isOutside(self, first):
        pass
    ################################ 

    def isThreeBar(self, first, second, third):
        # 1 means is threebar
        # 2 means is fourbar
        isUp = self.isUp
        if not self.isTall():
            return 0
        if not first.isShort(self):
            # print("FIRST NOT SHORT", self.name, self.date)
            return 0
        if self.volume < self.average_volume:
            # print("LOW VOLUME",self.name, self.average_volume, self.volume, self.date)
            return 0
        if second.isTall() and second.isUp == isUp:
            #print("FOUND 3 BAR", self.name, self.date)
            if second.volume < self.average_volume:
                # print("LOW VOLUME IN 2",self.name, self.average_volume, second.volume, self.date)
                return 0
            if isUp:
                # print("IS UP!")
                if first.inRange(self.close, second.open):
                    # print("3 BAR LOCATED", self.name, self.date)
                    return 1
            else:
                # print("IS DOWN!")
                if first.inRange(second.open, self.close):
                    # print("3 BAR LOCATED",self.name, self.date)
                    return 1
            # if self.compareBars(first, second):
            #     return 1
        elif third.isTall() and third.isUp == isUp:
            #print("FOUND 4 BAR", self.name, self.date)
            # if third.volume < self.average_volume:
            #     print("LOW VOLUME IN 3",self.name, self.average_volume, third.volume, self.date)
            #     return 0
            if second.isShort(self):
                if isUp:
                    if first.inRange(self.close, third.open) and second.inRange(self.close, third.open):
                        # print("4 BAR LOCATED", self.name, self.date)
                        return 2
                else:
                    if first.inRange(third.open, self.close) and second.inRange(third.open, self.close):
                        # print("4 BAR LOCATED", self.name, self.date)
                        return 2
                # if third.valueInRange(second.close) and third.valueInRange(first.close):
                #     print("4 BAR LOCATED")
                #     return 2
                # print("Four Bar:", bar.date, isUp, index)
                #print("Second Bar", second_bar.date, second_bar.isUp)
                #print("Third Bar", third_bar.date, third_bar.isUp)
                # TODO: Implement correct 4 bar
                # if all([self.compareBars(bar, third) for bar in [first, second]]):
                #     return 2
                # return 2
        return 0

    def compareBars(self, bar, prev_bar):
        epsilon = self.average_height / 3.0
        if bar.isUp:
            difference = abs(bar.close - prev_bar.close)
            if difference <= epsilon:
                # print("Three Bar 1:", bar.date, bar.volume, isUp, prev_bar.isUp, difference)
                return 1
        else:
            difference = abs(bar.open - prev_bar.close)
            if difference <= epsilon:
                # print("Three Bar 2:", bar.date, bar.volume, isUp, prev_bar.isUp, difference)
                return 1
        return 0


PERIOD = "1d"
INTERVALS = "1m"

class Stock(object):
    """Class to handle Stock.
    TODO: Check how to get stock data for current minute and add it to csv.
    """
    CSV_FOLDER = "/Users/ignacio/workspace/Projects/stockBuyer/data/{name}/{day}_{period}_{intervals}.csv"
    RELAX_DEVIATION = 0.015
    START_TIME = "10:20"
    END_TIME = "14:00"
    SLOPE_DEGREE = 4
    def __init__(self, name, sector=None):
        self._name = name
        self._sector = sector
        self._average_volume = None
        self._average_height = None
        self._inAction = None # If the stock was bought or sold.
        self._data = None

    @property
    def data(self):
        return self._data

    @property
    def inAction(self):
        return self._inAction

    @property
    def sector(self):
        return self._sector

    @property
    def name(self):
        return self._name

    @property
    def average_height(self):
        return self._average_height

    @property
    def average_volume(self):
        return self._average_volume

    def getDataPath(self, day, period=PERIOD, intervals=INTERVALS):
        return self.CSV_FOLDER.format(
            name=self.name, 
            day=day, 
            period=period, 
            intervals=intervals
        ) 

    def getData(self, day, tomorrow, period=PERIOD, intervals=INTERVALS, update=False):
        csv_path = self.getDataPath(day, period, intervals)
        if os.path.exists(csv_path) and not update:
            self._data = pd.read_csv(csv_path)
        else:
            ticker = yf.Ticker(self.name)
            self._data = ticker.history(period=period, interval=intervals, start=day, end=tomorrow)
            self._data['Height'] = (self._data['Close'] - self._data['Open']).abs()
            self._data['IsUp'] = (self._data['Close'] > self._data['Open'])
            if not os.path.exists(os.path.dirname(csv_path)):
                os.makedirs(os.path.dirname(csv_path))
            self._data.to_csv(csv_path)
        return self._data

    def getBars(self, day, period=PERIOD, intervals=INTERVALS):
        if self.average_height is None:
            return []
        tomorrow = day + timedelta(days=1)
        day = day.strftime("%Y-%m-%d")
        tomorrow = tomorrow.strftime("%Y-%m-%d")
        data = self.getData(day, tomorrow, period, intervals, update=True)
        self._bars = [Bar(self.name, 
                data.reset_index().iloc[[x]],
                self.average_height,
                self.average_volume) for x in range(len(data))]

    def finishedRelax(self, day, period=PERIOD, intervals=INTERVALS):
        # Get previous day that is not a weekend
        yesterday = day - timedelta(days=1)
        while True: 
            weekno = yesterday.weekday()
            if weekno<5:
                break
            yesterday = yesterday - timedelta(days=1)
        day = day.strftime("%Y-%m-%d")
        yesterday = yesterday.strftime("%Y-%m-%d")
        data = self.getData(yesterday, day, period, intervals)
        # Convert interval to int
        i_interval = int(intervals[:-1])
        last_2_hours_index = int(len(data) - (60/i_interval))
        temp_hist = data[last_2_hours_index:len(data)]
        deviation = temp_hist['Close'].std()
        self._average_height = data.loc[:,"Height"].mean()
        self._average_volume = data.loc[:,"Volume"].mean()
        if deviation < self.RELAX_DEVIATION:
            print("Deviation:",self.name, deviation)
            return True
        return False

    def getNow(self):
        """Get Stock data of current minute"""
        ticker = yf.Ticker(self.name)
        return ticker.info

    def findAction(self, day, start_time=None):
        self.getBars(day)
        if start_time is None:
            start_time = self.START_TIME
        for i, bar in enumerate(self._bars):
            if bar.time < start_time:
                continue
            if bar.time > self.END_TIME:
                continue
            trend = self.getTrend(i)
            # Test
            first = self._bars[i - 1]
            pinochio = bar.isPinochio(first, trend)
            reversal = bar.isReversal(first, trend)
            if pinochio or reversal:
                return bar
        return False

    def getTrend(self, index):
        temp_hist = self._data.reset_index()
        temp_hist = temp_hist.drop(["Datetime"], axis=1)
        temp_hist = temp_hist.drop([0])
        slopes = np.polyfit(temp_hist[0:index].index, temp_hist[0:index]["Close"], self.SLOPE_DEGREE)
        trend_values = np.polyval(slopes, temp_hist[0:index].index)
        return trend_values[-1] > trend_values[-2]

    def getStop(self, index):
        trend = self.getTrend(index)
        buy_bar = self._bars[index]
        for i, bar in enumerate(self._bars[index+1:]):
            ind = index + 1 + i
            # slopes = np.polyfit(temp_hist[0:ind].index, temp_hist[0:ind]["Close"], degree)
            # trend_values = np.polyval(slopes, temp_hist[0:ind].index)
            # Trend True is going up
            #print(len(trend_values), len(temp_hist[0:index]))
            # b_trend = trend_values[-1] > trend_values[-2]
            # if ind > 10:
            # b_trend = trend_values[-1] > trend_values[-2]
            # else:
            #     # if first periods
            # b_trend = bar.MA > bars[ind-1].MA
            b_trend = self.getTrend(ind)
            if b_trend != trend:
                if trend:
                    gain = getGain(investment, buy_bar.close, bar.close)
                else:
                    gain = getGain(investment, bar.close, buy_bar.close)
                prYellow("STOP!", bar, [b_trend, gain])
                return bar
        if trend:
            gain = getGain(investment, buy_bar.close, self._bars[-1].close)
        else:
            gain = getGain(investment, self._bars[-1].close, buy_bar.close)
        prPurple("USING LAST", self._bars[-1], [trend, gain])
        return self._bars[-1]


import json

class StockManager(object):
    TICKERS_PATH = "/Users/ignacio/workspace/Projects/stockBuyer/data/tickers.json"
    def __init__(self, day):
        self._data = self.load()
        self._relaxed = []
        self._day = day

    @property
    def tickers(self):
        return [v for s in self._data for v in self._data[s]]

    @property
    def sectors(self):
        return self._data.keys()

    def relaxedTickers(self):
        if self._relaxed:
            return self._relaxed
        for sector in self.sectors:
            for ticker in self._data[sector]:
                stock = Stock(ticker, sector)
                try:
                    if stock.finishedRelax(self._day):
                        self._relaxed.append(stock)
                except Exception as e:
                    print("Failed to load", ticker, e)
                    continue
        return self._relaxed

    def load(self, update=False):
        if os.path.exists(self.TICKERS_PATH) and not update:
            with open(self.TICKERS_PATH) as json_file:  
                return json.load(json_file)

        # Update tickers
        site = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # START = datetime(1900, 1, 1, 0, 0, 0, 0, pytz.utc)
        # END = datetime.today().utcnow()
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = Request(site, headers=hdr)
        page = urlopen(req)
        soup = BeautifulSoup(page)

        table = soup.find('table', {'class': 'wikitable sortable'})
        sector_tickers = dict()
        for row in table.findAll('tr'):
            col = row.findAll('td')
            if len(col) > 0:
                sector = str(col[3].string.strip()).lower().replace(' ', '_')
                ticker = str(col[0].find('a').string.strip())
                if sector not in sector_tickers:
                    sector_tickers[sector] = list()
                sector_tickers[sector].append(ticker)

        with open(self.TICKERS_PATH, 'w') as outfile:
            json.dump(sector_tickers, outfile)
        return sector_tickers

    def buy(self, start_time=None):
        for stock in self.relaxedTickers():
            # FIXME: This should search for current time.
            bar = stock.findAction(self._day, start_time)
            if bar:
                self.sell(stock, bar)
                break

    def sell(self, stock, start_bar):
        stopBar = stock.getStop(start_bar.index)
        # If market still open try to buy another time.
        if stopBar.date.hour <= 15:
            self.buy(stopBar.time)
        


        
#%%
def checkBuy(stock, start_date, end_date, interval):
    ticker = yf.Ticker(stock)
    s_interval = str(interval)+"m"
    tomorrow = start_date + timedelta(days=1)
    today = start_date.strftime("%Y-%m-%d")
    yesterday = end_date.strftime("%Y-%m-%d")
    tomorrow = tomorrow.strftime("%Y-%m-%d")
    try:
        hist = ticker.history(period="1d", interval=s_interval, start=today, end=tomorrow)
        yesterday_hist = ticker.history(period="1d", interval=s_interval, start=yesterday, end=today)
    except:
        print("FAILING TO LOAD FOR: "+stock)
        return
    # hist = hist.reset_index()
    yesterday_hist['Height'] = (yesterday_hist['Close'] - yesterday_hist['Open']).abs()
    hist['Height'] = (hist['Close'] - hist['Open']).abs()
    hist['IsUp'] = (hist['Close'] > hist['Open'])
    # TODO: Merge yesterday_hist and hist to calculate MA.
    yes_len = len(yesterday_hist)
    ma_hist = pd.concat([yesterday_hist[-50:], hist])
    hist['MA'] = ma_hist['Close'].rolling(window=50).mean()[50:]
    average_height = yesterday_hist.loc[:,"Height"].mean()
    average_volume = yesterday_hist.loc[:,"Volume"].mean()
    bars = [Bar(stock, hist.reset_index().iloc[[x]], average_height, average_volume) for x in range(len(hist))]
    threeFound = {}
    totalWin = 0
    win = ""
    temp_hist = hist.reset_index()
    temp_hist = temp_hist.drop(["Datetime"], axis=1)
    temp_hist = temp_hist.drop([0])
    degree = 10
    for i,b in enumerate(bars[4:]):
        index = i + 4
        first = bars[index - 1]
        second = bars[index - 2]
        third = bars[index - 3]
        # Buy only until 12 pm
        if b.date.hour < 10 and b.date.minute < 40:
            continue
        if b.date.hour > 14:
            continue
        # if index > 10:
        #     degree = 5
        # else:
        #     degree = 1
        slopes = np.polyfit(temp_hist[0:index].index, temp_hist[0:index]["Close"], degree)
        trend_values = np.polyval(slopes, temp_hist[0:index].index)
        # Trend True is going up
        #print(len(trend_values), len(temp_hist[0:index]))
        # if index > 10:
        #     trend = trend_values[-1] > trend_values[-2]
        # else:
        # if first periods
        # trend = b.MA > first.MA
        trend = trend_values[-1] > trend_values[-2]
        # trend = trend_values[-1] > trend_values[-2]
        #trend = trend[0]
        #trend = b.MA > first.MA
        #reversal = b.isReversal(first, trend)
        #exhaust = b.isExhaustion(first)
        pinochio = b.isPinochio(first, trend)
        # TODO: Maybe checking the past to see if.
        # Check AMT
        # isThree = b.isThreeBar(first, second, third)
        if pinochio:
            # Check if there is some previous bar in oposite direction
            # and with high volume
            # temp_hist = hist[0:index]
            # if b.isUp:
            #     match = temp_hist.loc[(~temp_hist['IsUp']) & (temp_hist['Volume'] > average_volume * 1.5)]
            # else:
            #     match = temp_hist.loc[(temp_hist['IsUp']) & (temp_hist['Volume'] > average_volume * 1.5)]
            # if len(match):
            #     print(match)
            #     print("SKIPPING BUY! "+stock+" , PREVIOUS STOCKS WITH HIGH NEGATIVE VOLUME", b.date, average_volume)
            #     return win 
            # Save index and Stop
            # stop = (abs(second.close - second.open) /2.0) + second.open
            # FIXME: if It is four bars stop should be third.open
            # Set Stop 5% higher or lower of current Close
            if trend:
                stop = b.close - getPercent(b.close, 0.75)
                print("STOP INICIAL", str(stop), trend)
            else:
                stop = b.close + getPercent(b.close, 0.75)
                print("STOP INICIAL", str(stop), trend)
            # if isThree == 1:
            #     stop = second.open
            # else:
            #     stop = third.open
            threeFound[b] = [index, stop, trend]
    # FIXME: Here win is being overriden by many ThreeBars.
    if threeFound:
        # print("Average Height:", average_height)
        # print("Average Volume:", average_volume)
        for t in threeFound:
            index = threeFound[t][0]
            stop = threeFound[t][1]
            trend = threeFound[t][2]
            future = getSellBar(trend, bars, index, temp_hist, degree)
            # future = getSellBar(t, stop, index, bars, 1.0, 1.0, False, trend)
            win = "LOOSER!"
            if trend:
                gain = getGain(investment, t.close, future.close)
                if t.close < future.close:
                    win = "WIN!"
                    totalWin += 1
            else:
                gain = getGain(investment, future.close, t.close)
                if t.close > future.close:
                    win = "WIN!"
                    totalWin += 1
            if "WIN!" == win:
                prGreen(win +" " +str(gain), t, [future.close, index])
            else:
                prRed(win +" " +str(gain), t, [future.close, index])
            break
    return win

def getSellBar(trend, bars, index, temp_hist, degree):
    for i, bar in enumerate(bars[index+1:]):
        ind = index + 1 + i
        slopes = np.polyfit(temp_hist[0:ind].index, temp_hist[0:ind]["Close"], degree)
        trend_values = np.polyval(slopes, temp_hist[0:ind].index)
        # Trend True is going up
        #print(len(trend_values), len(temp_hist[0:index]))
        # b_trend = trend_values[-1] > trend_values[-2]
        # if ind > 10:
        b_trend = trend_values[-1] > trend_values[-2]
        # else:
        #     # if first periods
        # b_trend = bar.MA > bars[ind-1].MA
        if b_trend != trend:
            prYellow("STOP!", bar, [b_trend])
            return bar
    prBlack("USING LAST", bars[-1])
    return bars[-1]
    
def getSellBar2(buy_bar, stop, index, bars, percent, max_percent, dynamic, trend):
    """Get the Bar that we should sell.

    Don't sell until percent reached.
    If dynamic:
        - if Percernt reached, check if there is bar between 
    Args:
        buy_bar (Bar): Bar that we bought.
        stop (float): The stop that we set for selling.
        index (int): Index of the bar that we bought in bars.
        percent (float): How much we want to gain.
        max_percent (float): If this is reached sell!.
        dynamic (bool): If its dynamic it will increase the stop and percent
                when the stock increase.
    """
    initial_stop = stop
    initial_percent = percent
    # TODO: In some cases we don't reach percent..
    # but we are close.. see if we can sell at that point.
    # What if the 2 large bars are in similar size?
    max_pc = 0
    for i, bar in enumerate(bars[index+1:]):
        # if bar.isUp != buy_bar.isUp and bar.volume > bar.average_volume * 1.5:
        #     print("STOP! High negative volume", bar.date, bar.close)
        #     return bar
        if trend:
            current_percent = (bar.close * 100 /  buy_bar.close) - 100
            if max_pc < current_percent:
                max_pc = current_percent
            if bar.close < stop:
                # We reach stop :(
                prYellow("STOP!", bar, [stop, max_percent])
                return bar
                # continue
            elif current_percent > max_percent:
                prCyan("MAXIMUM REACHED", bar)
                return bar
            elif current_percent > percent:
                if dynamic:
                    # stop += (bar.close - stop)*0.75
                    stop = bar.close - getPercent(bar.close, 0.25)
                    percent = current_percent + 0.5
                    prLightPurple(" ".join(["INCREASING STOP!",str(stop), "INCREASING PERCENT", str(percent)]), bar)
                else:
                    prPurple("SELL!", bar)
                    # We reach the percent that we want to win :)
                    return bar
        else:
            current_percent = (buy_bar.close * 100 / bar.close) - 100
            if max_pc < current_percent:
                max_pc = current_percent
            if bar.close > stop:
                # We reach stop :(
                prYellow("STOP!", bar, [stop, max_pc])
                return bar
                # continue
            elif current_percent > max_percent:
                prCyan("MAXIMUM REACHED", bar)
                return bar
            elif current_percent > percent:
                if dynamic:
                    # stop -= (stop - bar.close)*0.75
                    stop = bar.close + getPercent(bar.close, 0.25)
                    percent = current_percent + 0.5
                    prLightPurple(" ".join(["INCREASING STOP!",str(stop), "INCREASING PERCENT", str(percent)]), bar)
                else:
                    # We reach the percent that we want to win :)
                    prPurple("SELL!", bar)
                    return bar
    # if not percent or stop reached
    # return last bar
    prBlack("USING LAST", bars[-1], [stop, max_pc])
    return bars[-1]

def detectPosibleGoodStock(stock, start_date, end_date, interval):
    """Based on the bars of the previous day decide if is a good stock."""
    ticker = yf.Ticker(stock)
    s_interval = str(interval)+"m"
    today = start_date.strftime("%Y-%m-%d")
    yesterday = end_date.strftime("%Y-%m-%d")
    try:
        # hist = ticker.history(period="1d", interval=interval, start=start_date)
        yesterday_hist = ticker.history(period="1d", interval=s_interval, start=yesterday, end=today)
    except:
        print("FAILING TO LOAD FOR: "+stock)
        return False
    # Only use the last hour
    last_2_hours_index = int(len(yesterday_hist) - (60/interval))
    temp_hist = yesterday_hist[last_2_hours_index:len(yesterday_hist)]
    #print(temp_hist)
    deviation = temp_hist['Close'].std()
    #mean = temp_hist[:,'Close'].mean()
    if deviation < 0.015:
        print("Deviation:",stock, deviation)
        return True
    return False

def plot(stock):
    offset_days = 0
    #today = date.today()
    today = date.today() - timedelta(days=1)
    # today = yesterday
    yesterday = date.today() - timedelta(days=2)
    interval = 2
    ticker = yf.Ticker(stock)
    s_interval = str(interval)+"m"
    tomorrow = today + timedelta(days=1)
    hist = ticker.history(period="1d", interval=s_interval, start='2019-06-05', end='2019-06-06')
    hist['MA'] = hist['Close'].rolling(window=50).mean()
    result = pd.DataFrame({
        "Close":hist['Close'],
        "MA_50":hist['MA']
        })
    result.plot(title=stock)
    #print(result)
    plt.show()
    #matplotlib.pyplot.show()

#%%
def checkMarket(tickers, today, yesterday):
    totalWin = 0
    totalLoose = 0
    totalFound = 0
    stocks_len = len([s for sector in tickers for s in tickers[sector]]) 
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    printProgressBar(0, stocks_len, prefix = 'Progress:', suffix = 'Complete', length = 50)
    i = 0
    print("TODAY:", today)
    print("YESTERDAY:", yesterday)
    print("Total Tickers:", stocks_len)
    check_buy = True
    interval = 1
    posible = True
    for sector in tickers:
        print("Sector:", sector)
        for t in tickers[sector]:
            # prRed("Ticker", t)
            posible = detectPosibleGoodStock(t, today, yesterday, interval)
            if check_buy and posible:
                win = checkBuy(t, today, yesterday, interval)
                if win:
                    totalFound += 1
                    if "WIN!" == win:
                        totalWin += 1
                    else:
                        totalLoose += 1
            i += 1
            printProgressBar(i + 1, stocks_len, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print("Total FOUND:", totalFound)
    print("Total WIN:", totalWin)
    print("Total LOOSE:", totalLoose)

def checkMarketManyDays(tickers):
    today = date.today()
    yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    # today = yesterday
    # yesterday = today - timedelta(days=1)
    for day in range(2):
        checkMarket(tickers, today, yesterday)
        today = yesterday
        yesterday = today - timedelta(days=1)
# checkMarket({"test":['COTY']})
#plot("COTY")
#checkMarketManyDays(tickers)
#checkMarketManyDays({"test":['WY']})
#checkMarketManyDays({"test":['CTL']})
def checkMarketToday(tickers):
    today = date.today()
    yesterday = today - timedelta(days=1)
    checkMarket(tickers, today, yesterday)

#checkMarketToday(tickers)
#checkMarketToday({"test":['AMAT']})

def run():
    today = date.today()
    # today = today - timedelta(days=1)
    manager = StockManager(today)
    manager.buy()
run()
#%%
# from matplotlib.finance import candlestick_ohlc
def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def plotSlopes():
    ticker = yf.Ticker("ROL")
    today = "2019-06-13"
    tomorrow = "2019-06-14"
    hist = ticker.history(period="1d", interval="1m", start=today, end=tomorrow)
    hist = hist.reset_index()
    yesterday = "2019-06-12"
    yes_hist = ticker.history(period="1d", interval="1m", start=yesterday, end=today)
    len_yes = len(yes_hist)
    hist = hist.drop(["Datetime"], axis=1)
    hist = hist.drop([0])
    #print(len(hist))
    hist = hist[0:92]
    ma_hist = pd.concat([yes_hist[-50:], hist])
    # ma_hist = ma_hist[0:64]
    degree = 5
    #slopes = hist.apply(lambda x: np.polyfit(hist.index, x,degree)[0])
    #print(slopes)
    plt.scatter(hist.index, hist["Close"], label="Close")
    vals = np.polyval(np.polyfit(hist.index, hist["Close"], degree), hist.index)
    print(vals[-1], vals[-2])
    print(vals[-1] > vals[-2])
    plt.plot(np.polyval(np.polyfit(hist.index, hist["Close"], degree), hist.index),'r--')
    plt.plot(ma_hist['Close'].rolling(window=50).mean()[50:], 'b-')

    plt.legend()
    plt.show()

#plotSlopes()
# from matplotlib.dates import MONDAY, DateFormatter, DayLocator, WeekdayLocator
#%%
def plotBuys():
    from mpl_finance import candlestick_ohlc
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w')
    ax1 = plt.subplot2grid((1,1), (0,0))

    # plot_day_summary(ax, quotes, ticksize=3)
    candlestick_ohlc(ax1, zip(mdates.date2num(hist.index.to_pydatetime()),
                            hist['Open'], hist['High'],
                            hist['Low'], hist['Close']),
                    width=0.001, colorup='#77d879', colordown='#db3f3f')


    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(90)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(25))
    ax1.grid(True)
    ma_value = 30
    ma_type = "coco"
    ma = moving_average(hist['Close'], ma_value, type=ma_type)
    linema, = ax1.plot(hist.index, ma, color='blue', lw=2, label='MA ({0})'.format(ma_value))
    isthre = ax1.scatter(hist.index, hist['Close'], color="black", s=three_bar*20)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(stock)
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()


    
