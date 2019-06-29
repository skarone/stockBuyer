import os
import plotly.offline as py
from plotly.figure_factory import create_candlestick
from plotly.graph_objs import *
# from plotly.tools import FigureFactory as FF
import numpy as np
import pandas as pd

COLORS = [
    "aliceblue", 
    "antiquewhite", "aqua", "aquamarine", "azure",
    "beige", "bisque", "black", "blanchedalmond", "blue",
    "blueviolet", "brown", "burlywood", "cadetblue",
    "chartreuse", "chocolate", "coral", "cornflowerblue",
    "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
    "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
    "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
    "darkorchid", "darkred", "darksalmon", "darkseagreen",
    "darkslateblue", "darkslategray", "darkslategrey",
    "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
    "dimgray", "dimgrey", "dodgerblue", "firebrick",
    "floralwhite", "forestgreen", "fuchsia", "gainsboro",
    "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
    "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
    "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
    "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
    "lightgoldenrodyellow", "lightgray", "lightgrey",
    "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
    "lightskyblue", "lightslategray", "lightslategrey",
    "lightsteelblue", "lightyellow", "lime", "limegreen",
    "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple",
    "mediumseagreen", "mediumslateblue", "mediumspringgreen",
    "mediumturquoise", "mediumvioletred", "midnightblue",
    "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
    "oldlace", "olive", "olivedrab", "orange", "orangered",
    "orchid", "palegoldenrod", "palegreen", "paleturquoise",
    "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
    "plum", "powderblue", "purple", "red", "rosybrown",
    "royalblue", "saddlebrown", "salmon", "sandybrown",
    "seagreen", "seashell", "sienna", "silver", "skyblue",
    "slateblue", "slategray", "slategrey", "snow", "springgreen",
    "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
    "violet", "wheat", "white", "whitesmoke", "yellow",
    "yellowgreen"
]

class Plotter(object):
    plots_path = "/Users/ignacio/workspace/Projects/stockBuyer/data/plots/{day}/{stock}.html"
    def __init__(self):
        self._stocks = []
        self._fig = None

    def addStock(self, new_stock):
        self._stocks.append(new_stock)

    def create_stock_plot(self, stock, buy_bars, sell_bars):

        df = stock.data
        day = buy_bars[0].date.strftime("%Y/%m/%d")
        name = stock.name
        data = [ dict(
            type = 'candlestick',
            open = df.Open,
            high = df.High,
            low = df.Low,
            close = df.Close,
            x = df.Datetime,
            yaxis = 'y2',
            name = name,
        ) ]

        annotations = self._createAnnotations(buy_bars, "Buy")
        annotations.extend(self._createAnnotations(sell_bars, "Sell"))
        layout = dict(
            title='{0} Stock - {1}'.format(name, str(day)),
            plot_bgcolor='rgb(230,230,230)',
            annotations= annotations,
            xaxis= dict( 
                rangeselector = dict( visible = False ),
                rangeslider = dict(visible = False)
            ),
            yaxis= dict( domain = [0, 0.2], showticklabels = True ),
            yaxis2= dict( domain = [0.2, 0.8] ),
            legend= dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' ),
            margin= dict( t=40, b=40, r=200, l=200 ),
        )

        fig = dict( data=data, layout=layout )

        bars = [item for sublist in zip(buy_bars,sell_bars) for item in sublist]
        trends = self._createTrends(stock, bars)
        data.extend(trends)
        # for trend in trends:
        #     self._fig.add_trace(trend)
        data.append(self._createVolume(df))

        path = self.plots_path.format(day=day, stock=name)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        py.plot(fig, filename=path, validate=False)

    def _createAnnotations(self, bars, title="Buy"):
        annotations=[]
        for bar in bars:
            bar_dict = dict(
                    x=bar.date.strftime('%Y-%m-%d %H:%M:%S%z'),
                    y=bar.close,
                    xref='x',
                    yref='y2',
                    text=title,
                    showarrow=True,
                    type='scatter',
                    arrowhead = 1,
                    arrowsize = 0.5
            )
            annotations.append(bar_dict)
        return annotations

    def _createVolume(self, df):
        bar = dict(
            x=df.Datetime,
            y=df['Volume'],
            name= 'Volume',
            yaxis='y',
            type='bar'
        )
        return bar

    def _createTrends(self, stock, bars):
        df = stock.data
        trends = []
        start_index = 0
        for i,bar in enumerate(bars):
            _scatter = self._createTrendCurve(stock, bar, df, i, start_index)
            if "reversal" in bar.actionType:
                start_index = bar.index
            else:
                start_index = 0
            trends.append(_scatter)
        return trends

    def _createTrendCurve(self, stock, bar, df, i, start_index=0):
        distance = start_index - bar.index
        trend_curve = stock.getTrendCurve(bar.index, start_index, debug=False)
        # stock.getTrend(bar.index, start_index, debug=False)
        # print(trend_curve[-2], trend_curve[-1])
        trend_curve_name = "{0}-{1}-{2}".format(distance, bar.actionType, bar.time)
        nones = np.array([np.nan for _ in range(len(df.Datetime)-len(trend_curve))])
        if start_index != 0:
            front_nones = np.array([np.nan for _ in range(start_index)])
            trend_curve = np.concatenate((front_nones, trend_curve))
        trend_curve = np.concatenate((trend_curve, nones))
        _scatter = dict(
            x=df.Datetime,
            y=trend_curve,
            name= trend_curve_name,
            line=scatter.Line(color=COLORS[i+30]),
            type='scatter',
            yaxis='y2'
            )
        return _scatter

    def create_plot(self, stocks_dict):
        for stock in stocks_dict:
            buy_bars = stocks_dict[stock]['buy']
            sell_bars = stocks_dict[stock]['sell']
            self.create_stock_plot(stock, buy_bars, sell_bars)

    
