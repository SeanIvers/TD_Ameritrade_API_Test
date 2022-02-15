import requests
import pandas as pd
import plotly.graph_objects as go
from config import apiKey
import datetime

class TestApp:
    def __init__(self):
        self.data = None
        self.df = None
        self.df2 = None
        self.y = None
        self.EMA_list = []

    def get_historical_price_data(self, symbol, periodType, period, frequencyType, frequency, endDate=None, startDate=None, needExtendedHoursData=False):
        endpoint = f"https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory"

        parameters = {
            'apikey': apiKey,
            'periodType': periodType, # day, month, year, ytd
            'period': period,   # day: 1, 2, 3, 4, 5, 10*
                                # month: 1*, 2, 3, 6
                                # year: 1*, 2, 3, 5, 10, 15, 20
                                # ytd: 1*
            'frequencyType': frequencyType, # day: minute*
                                            # month: daily, weekly*
                                            # year: daily, weekly, monthly*
                                            # ytd: daily, weekly*
            'frequency': frequency, # minute: 1*, 5, 10, 15, 30
                                    # daily: 1*
                                    # weekly: 1*
                                    # monthly: 1*
            'needExtendedHoursData': needExtendedHoursData
        }

        epoch = datetime.datetime.utcfromtimestamp(0)

        def unix_time_milliseconds(datetime):
            return (datetime - epoch).total_seconds() * 1000

        # If endDate and startDate are provided period will be removed from parameters as required by api
        if endDate and startDate:
            parameters.pop('period')
            parameters.update({'endDate': int(unix_time_milliseconds(datetime.datetime.strptime(endDate, '%Y%m%d %H:%M:%S'))), 'startDate': int(unix_time_milliseconds(datetime.datetime.strptime(startDate, '%Y%m%d %H:%M:%S')))})

        self.data = requests.get(endpoint, params=parameters).json()

        self.df = pd.DataFrame(self.data['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        self.df2 = self.df

        return self.data

    def show_candlestick_chart_linear_regression(self):
        self.df.iloc[:, 0] = pd.to_datetime(self.df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        fig = go.Figure(
            [
                go.Candlestick(x=self.df['datetime'], open=self.df['open'], high=self.df['high'], low=self.df['low'], close=self.df['close']),
                go.Scatter(x=self.df['datetime'], y = self.y)
            ]
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9.5], pattern='hour'),
                                        dict(bounds=['sat', 'mon'])])
        fig.show()

    def linear_regression(self, learning, iterations):
        # 1 & 2 Day Five Minute - learning_rate = 0.0001, num_iterations = 500000
        # 3 to 5 Day Five Minute - learning_rate = 0.00001, num_iterations = 1000000
        # 10 Day Five Minute - learning_rate = 0.000001, num_iterations = 10000000

        # list of x values - ms since epoch
        new_x = [x for x in range(self.df['datetime'].size)]
        # list of y values - closing price
        closing_prices_list = self.df['close'].to_list()

        def get_gradient_at_b(b, m):
            # m is the current gradient guess
            # b is the current intercept guess
            diff = 0
            # gradient of loss as intercept changes = -(2 / number of points) * sum of (y point - (current gradient guess * x point + intercept guess))
            for x, closing_price in zip(new_x, closing_prices_list):
                diff += (closing_price - (m * x + b))
            b_gradient = (-1 * (2 / len(new_x)) * diff)
            return b_gradient

        def get_gradient_at_m(b, m):
            # gradient descent for slope
            diff = 0
            for x, closing_price in zip(new_x, closing_prices_list):
                diff += (x * (closing_price - (m * x + b)))
            m_gradient = (-1 * (2 / len(new_x)) * diff)
            # print(m_gradient)
            return m_gradient

        def step_gradient(b_current, m_current, learning_rate):
            b_gradient = get_gradient_at_b(b_current, m_current)
            m_gradient = get_gradient_at_m(b_current, m_current)
            b = (b_current - (learning_rate * b_gradient))
            m = (m_current - (learning_rate * m_gradient))
            return [b, m]

        def gradient_descent(learning_rate, num_iterations):
            b = 0
            m = 0
            for iteration in range(num_iterations):
                b, m = step_gradient(b, m, learning_rate)
                # Timer countdown
                if (iteration / 100000).is_integer():
                    print(int(num_iterations / 100000 - iteration / 100000))
            return [b, m]

        b, m = gradient_descent(learning, iterations)

        self.y = [m * x + b for x in new_x]

    def show_candlestick_chart_EMA(self):
        self.df.iloc[:, 0] = pd.to_datetime(self.df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

        figures =   [
                        go.Candlestick(x=self.df['datetime'], open=self.df['open'], high=self.df['high'], low=self.df['low'], close=self.df['close']),
                    ]
        
        for num in self.EMA_list:
            figures.append(go.Scatter(x=self.df['datetime'], y=self.df[f'{num} EMA']))

        fig = go.Figure(figures)
        
        fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9.5], pattern='hour'),
                                        dict(bounds=['sat', 'mon'])])
        fig.show()

    def add_EMA(self, *args):
        self.EMA_list = list(args)
        for arg in self.EMA_list:
            self.df[f'{arg} EMA'] = self.df['close'].ewm(span=arg, adjust=False).mean()

test = TestApp()

test.get_historical_price_data('MSFT', 'day', 10, 'minute', 5)
# print(test.df)
# test.linear_regression(0.000001, 10000000)
# test.show_candlestick_chart_linear_regression()
# Indicate if y-values are too large

# test.df['9 EWM'] = test.df['close'].ewm(span=9, adjust=False).mean()

# test.show_candlestick_chart_test()

# print(test.y)

test.add_EMA(9, 15, 200)

print(test.df.head())
test.show_candlestick_chart_EMA()