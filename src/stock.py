from datetime import datetime, timedelta, date
import os
import pandas as pd
import yfinance as yf
import numpy as np
import math

from utils import prGreen, prRed, prYellow, prPurple, prCyan, getGain, getPreviousWeekDay
from bar import Bar
from settings import PERIOD, INTERVALS, INVESTMENT, SLOPE_DEGREE

class Stock(object):
    """Class to handle Stock.
    TODO: Check how to get stock data for current minute and add it to csv.
    """
    CSV_FOLDER = "/Users/ignacio/workspace/Projects/stockBuyer/data/{name}/{day}_{period}_{intervals}.csv"
    RELAX_DEVIATION = 0.015
    START_TIME = "10:20"
    END_TIME = "14:00"
    def __init__(self, name, sector=None):
        self._name = name
        self._sector = sector
        self._average_volume = None
        self._average_height = None
        self._inAction = None # If the stock was bought or sold.
        self._data = None
        self._gain = 0

    @property
    def data(self):
        return self._data

    @property
    def inAction(self):
        if self._inAction == 1:
            return "buy"
        elif self._inAction == -1:
            return "sell"
        return None

    @inAction.setter
    def inAction(self, action):
        self._inAction = action

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

    @property
    def gain(self):
        return self._gain

    def getDataPath(self, day, period=PERIOD, intervals=INTERVALS):
        return self.CSV_FOLDER.format(
            name=self.name, 
            day=day, 
            period=period, 
            intervals=intervals
        ) 

    def getData(self, day, tomorrow, period=PERIOD, intervals=INTERVALS, update=False):
        csv_path = self.getDataPath(day, period, intervals)
        if os.path.exists(csv_path) and not date.today() == day:
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
        # day = day.strftime("%Y-%m-%d")
        # tomorrow = tomorrow.strftime("%Y-%m-%d")
        data = self.getData(day, tomorrow, period, intervals, update=True)
        self._bars = [Bar(self.name, 
                data.reset_index().iloc[[x]],
                self.average_height,
                self.average_volume) for x in range(len(data))]

    def finishedRelax(self, day, period=PERIOD, intervals=INTERVALS):
        # Get previous day that is not a weekend
        yesterday = day - timedelta(days=1)
        yesterday = getPreviousWeekDay(yesterday)
        # day = day.strftime("%Y-%m-%d")
        # yesterday = yesterday.strftime("%Y-%m-%d")
        data = self.getData(yesterday, day, period, intervals)
        # Convert interval to int
        i_interval = int(intervals[:-1])
        last_2_hours_index = int(len(data) - (60/i_interval))
        temp_hist = data[last_2_hours_index:len(data)]
        deviation = temp_hist['Close'].std()
        self._average_height = data.loc[:,"Height"].mean()
        self._average_volume = data.loc[:,"Volume"].mean()
        if deviation < self.RELAX_DEVIATION:
            return True
        # FIXME: HACK!
        return False

    def getNow(self):
        """Get Stock data of current minute"""
        ticker = yf.Ticker(self.name)
        return ticker.info

    def findAction(self, day, start_time=None):
        self.getBars(day)
        if start_time is None:
            start_time = self.START_TIME
        self._gain = 0
        self.inAction = None
        for i, bar in enumerate(self._bars):
            if bar.time <= start_time:
                continue
            if bar.time > self.END_TIME:
                continue
            if bar.height <= 0.01:
                continue
            trend = self.getTrend(i)
            # Test
            first = self._bars[i - 1]
            second = self._bars[i - 2]
            # if bar.isPinochio(first, trend):
            #     return bar
            # if bar.isExhaustion(first):
            #     return bar
            if bar.isTwoBarReversal(first):
                if bar.isUp:
                    self.inAction = 1
                else:
                    self.inAction = -1
                return bar
            # if bar.isReversal(first, trend):
            #     return bar
            # if bar.isThreeBarReversal(first, second):
            #     return bar
        return False

    def getTrend(self, index, start_index=0, debug=False):
        trend_values = self.getTrendCurve(index, start_index)
        epsilon = 0.15
        difference = trend_values[-1] - trend_values[-2]
        angle = math.degrees(math.atan(difference))
        if debug:
            print("ANGLE", angle, difference)
        if epsilon >= angle >= -epsilon:
            return None
        elif epsilon <= angle:
            return True
        elif angle <= -epsilon:
            return False
    
    def getTrendCurve(self, index, start_index=0, debug=False):
        temp_hist = self._data.reset_index()
        index = index+1
        degree = SLOPE_DEGREE
        diff_index = index - start_index
        if diff_index < 3:
            degree = 1
        elif 3 < diff_index < 30:
            degree = 2
        if debug:
            print("DEGREE", degree, len(temp_hist[start_index:index]))
        slopes = np.polyfit(
            temp_hist[start_index:index].index,
            temp_hist[start_index:index]["Close"],
            degree)
        trend_values = np.polyval(slopes, temp_hist[start_index:index].index)
        return trend_values

    def getStop(self, index, debug=False):
        buy_bar = self._bars[index]
        if "reversal" in buy_bar.actionType:
            trend = buy_bar.isUp
        else:
            trend = self.getTrend(index)

        for i, bar in enumerate(self._bars[index+1:]):
            ind = index + 1 + i
            b_trend = self.getTrend(ind, index)
            if b_trend is None:
                continue
            if b_trend != trend and i < 5:
                if buy_bar.isUp and bar.low <= buy_bar.close:
                    continue
                if not buy_bar.isUp and bar.high >= buy_bar.close:
                    continue
            if debug:
                prYellow("DEBUG_STOP", bar)
            if b_trend != trend:
                if trend:
                    gain = getGain(INVESTMENT, buy_bar.close, bar.close)
                else:
                    gain = getGain(INVESTMENT, bar.close, buy_bar.close)
                if gain > 0:
                    prCyan("STOP!", bar, [b_trend, gain])
                else:
                    prRed("STOP!", bar, [b_trend, gain])
                self._gain += gain
                return bar
        if trend:
            gain = getGain(INVESTMENT, buy_bar.close, self._bars[-1].close)
        else:
            gain = getGain(INVESTMENT, self._bars[-1].close, buy_bar.close)
        if gain > 0:
            prCyan("USING LAST", self._bars[-1], [trend, gain])
        else:
            prRed("USING LAST", self._bars[-1], [trend, gain])
        # prPurple("USING LAST", self._bars[-1], [trend, gain])
        self._gain += gain
        return self._bars[-1]
