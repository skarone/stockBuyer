import json
import os
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

from stock import Stock
from plotter import Plotter
from settings import BLACK_LIST

class StockManager(object):
    TICKERS_PATH = "/Users/ignacio/workspace/Projects/stockBuyer/data/tickers.json"
    def __init__(self, day):
        self._data = self.load()
        self._relaxed = []
        self._day = day
        self._stocks = {}
        self._gain = 0

    @property
    def tickers(self):
        return [v for s in self._data for v in self._data[s]]

    @property
    def sectors(self):
        return self._data.keys()

    @property
    def gain(self):
        return self._gain

    def relaxedTickers(self):
        if self._relaxed:
            return self._relaxed
        for sector in self.sectors:
            for ticker in self._data[sector]:
                if ticker in BLACK_LIST:
                    continue
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

    @property
    def stocks(self):
        return self._stocks

    def buy(self, start_time=None):
        for stock in self.relaxedTickers():
            # FIXME: This should search for current time.
            try:
                bar = stock.findAction(self._day, start_time)
                if bar:
                    self.sell(stock, bar)
                    break
            except Exception as e:
                print("FAILED BUYING", stock.name)
                print(e)
                continue

    def sell(self, stock, start_bar):
        stopBar = stock.getStop(start_bar.index)
        if not stock in self._stocks:
            self._stocks[stock] = {"buy":[], "sell":[]}
        self._stocks[stock]["buy"].append(start_bar)
        self._stocks[stock]["sell"].append(stopBar)
        self._gain += stock.gain
        # If market still open try to buy another time.
        if stopBar.date.hour <= 15:
            self.buy(stopBar.time)
 