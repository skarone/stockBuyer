from datetime import datetime
from utils import prGreen, prRed, prYellow

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