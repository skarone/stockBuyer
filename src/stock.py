from datetime import datetime, timedelta
import os
import pandas as pd
import yfinance as yf
import numpy as np

from utils import prGreen, prRed, prYellow, prPurple, getGain
from bar import Bar
from settings import PERIOD, INTERVALS, INVESTMENT

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
                    gain = getGain(INVESTMENT, buy_bar.close, bar.close)
                else:
                    gain = getGain(INVESTMENT, bar.close, buy_bar.close)
                prYellow("STOP!", bar, [b_trend, gain])
                return bar
        if trend:
            gain = getGain(INVESTMENT, buy_bar.close, self._bars[-1].close)
        else:
            gain = getGain(INVESTMENT, self._bars[-1].close, buy_bar.close)
        prPurple("USING LAST", self._bars[-1], [trend, gain])
        return self._bars[-1]
