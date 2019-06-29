from settings import SPREAD_FEE
from datetime import datetime, timedelta


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
    if bar:
        msg += " ".join([str(s) for s in [bar.name, str(bar.date)[:-9], bar.close, bar.isUp, bar.height, bar.volume, bar.index]])
        msg += " "
    msg += " ".join([str(s) for s in extra])
    print(color.format(msg))


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

def getPreviousWeekDay(day):
    while True: 
        weekno = day.weekday()
        if weekno<5:
            break
        day = day - timedelta(days=1)
    return day
