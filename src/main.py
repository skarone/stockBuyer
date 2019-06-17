#!python
from stock_manager import StockManager
from datetime import date, timedelta

def main():
    today = date.today()
    # today = today - timedelta(days=1)
    manager = StockManager(today)
    manager.buy()

if __name__ == "__main__":
    main()