#!python
from stock_manager import StockManager
from datetime import date, timedelta
from utils import getPreviousWeekDay, prPurple, prYellow
from plotter import Plotter

"""
TOW BAR REVERSAL:
/2019/06/27/LKQ
How to prevent this from happening?
-13!!!!!! maldita sea.
"""

def main(days=10):
    today = date.today()
    today = today - timedelta(days=1)
    # today = today - timedelta(days=1)
    plot = Plotter()
    total_gain = 0
    for i in range(days):
        today = getPreviousWeekDay(today)
        print(today, "-"*40)
        manager = StockManager(today)
        print("FOUND {0} STOCKS TO PLAY".format(len(manager.relaxedTickers())))
        manager.buy()
        prPurple("DAY GAIN", None, [manager.gain])
        total_gain += manager.gain
        plot.create_plot(manager.stocks)
        today = today - timedelta(days=1)
    prYellow("TOTAL GAIN", None, [total_gain])





if __name__ == "__main__":
    main()