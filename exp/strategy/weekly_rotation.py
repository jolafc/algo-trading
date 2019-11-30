from timeit import default_timer as timer

import numpy as np
import pandas as pd
import ta.momentum
from sklearn.base import BaseEstimator

from exp import SP500_PKL
from exp.backtesting import Backtesting
from exp.data_getter import load_pickled_dict
from exp.data_loader import get_feature, slice_backtesting_window
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN, VOLUME_COLUMN, DEFAULT_START_BALANCE
from exp.reporting import make_backtesting_report


### TODO: Need the joiner and leaver data of the SP500 stocks and filtering functions
### TODO: Need to (re)-implement the RSI/EMA indicators.
### TODO: Forward pad: limit the number of padded values
### TODO: Ticker changes, merger, acquisitions, and bankrupcies: currently selling at the buy price, to prevent automl to pick up on those errors.
### TODO: Use a multi-columns levels DF as single data source and put the data loading stuff outside the WRRunner class.
#         Then put the fit method contents in the predict(X), where predict expects a scile of that multi-columns DF.
#         The score(X, y=None) then calls predict(X) and returns the yield only - like the current one does with the fit.
#         Conccern: isn't averaging accross folds the same as optimizing over a larger window?
#         Since train/validation split is meaningless here (no train); isn't it worthless to have an Estimator score
#         being optimized instead of a single function output, without any explicit data handling by the HPO?


# Backtesting loop.
# V 1 - Positions dataframe with buy price+date
# V 2 - Trades dataframe with positions df columns + sell price+date+fees.
# V   - Sells are first removed from the positions and added to the trades. Then buys are added to positions.
# V 3 - Then, add P&L.
# V 4 - Balance: remove each transaction and its fees as they occur.
# V   - And assert that the balance - P&L.sum() + positions bought == start_balance
# V 5 - Unrealized P&L
# V 6 - Benchmark against the S&P500 + plotting
# V 7 - Metrics: Annualized yield, sharpe, sortino, ...
# V 8 - Refactor the position getter into a strategy framework that
# V   - takes start, end, lookback, strategy parameters, strategy data dict
# V   - returns iterables of dates and positions on those dates.
# V 9 - Refactor the backtesting loop into a class
# V 10 - Data loader file, metrics file, reporting file, ...
# V 11 - Passthrough all strategy parameters
# 12 - Use the open, high and low of the next day when executing trades in backtesting, to get more meaningful
#      estimate and confidence interval.
#    - (Requires) figure out the correction factor from AV and apply it to open, high, and low prices.
# 13 - HPO: do function optimization (in-sample)


class WeelkyRotation(BaseEstimator):

    def __init__(self,
                 start_date=None,
                 end_date=None,
                 lookback=200,

                 sma_tol=0.02,
                 volume_lookback=20,
                 volume_threshold=1e6,
                 price_min=1.,
                 rsi_lookback=3,
                 rsi_threshold=50.,
                 day_of_trade=4,
                 n_positions=10):
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback

        self.sma_tol = sma_tol
        self.volume_lookback = volume_lookback
        self.volume_threshold = volume_threshold
        self.price_min = price_min
        self.rsi_lookback = rsi_lookback
        self.rsi_threshold = rsi_threshold
        self.day_of_trade = day_of_trade
        self.n_positions = n_positions

    def fit(self, data_by_feature={}):
        adj_close_prices = data_by_feature[ADJUSTED_CLOSE_COLUMN]
        volumes = data_by_feature[VOLUME_COLUMN]

        # Filters for §8: Weekly rotation of the S&P 500 - The 30-Minute Stock Trader
        # 1 - SPY is above 0.98*SMA(200)
        spy = adj_close_prices['SPY']
        spy_sma = adj_close_prices['SPY'].rolling(self.lookback).mean()
        spy_masks = spy > spy_sma * (1 - self.sma_tol)

        # 2 - avg. vol. (20) >= 1M
        volumes_sma = volumes.rolling(self.volume_lookback).mean()
        volumes_masks = volumes_sma >= self.volume_threshold

        # 3 - Min price 1$
        price_masks = adj_close_prices > self.price_min

        # 4 - RSI(3) < 50
        prices_rsi = adj_close_prices.apply(lambda x: ta.momentum.rsi(x, n=self.rsi_lookback))
        rsi_masks = prices_rsi < self.rsi_threshold

        # Total masks 1-4
        # For pd.Series: .values[:, None] is to transform the series to a np.array and use its broadcasting capabilities,
        # specifying which axis needs to be repeated
        masks = volumes_masks & price_masks & rsi_masks & spy_masks.values[:, None]

        # 5a - Sort wrt the ROC(200)
        performances = adj_close_prices / adj_close_prices.shift(self.lookback - 1)

        # 5b - trade on Fridays in the backtesting window
        valid_dates = adj_close_prices.index
        day_of_week_triggers = valid_dates.dayofweek == self.day_of_trade
        backtesting_window_triggers = (valid_dates >= self.start_date) & (valid_dates <= self.end_date)
        triggers = day_of_week_triggers & backtesting_window_triggers
        dates = valid_dates[triggers]

        positions_list = pd.Series(index=dates, name='positions', dtype=object)
        for date in dates:
            mask = masks.loc[date]
            performance = performances.loc[date]
            positions_list[date] = performance[mask].sort_values(ascending=False)[:self.n_positions].index.tolist()

        self.positions_list = positions_list

        return self

    def predict(self, date):
        return self.positions_list[date]

    def get_dates(self):
        return self.positions_list.index


def weekly_rotation_runner_facotry(start_date_requested=pd.to_datetime('2019-01-31'),
                                   end_date_requested=pd.to_datetime('2019-10-31'),
                                   max_lookback=200,
                                   start_balance=DEFAULT_START_BALANCE,
                                   verbose=True,
                                   output_metric=None):
    data_by_ticker = load_pickled_dict(pkl_file=SP500_PKL)

    data_by_feature = {}
    data_by_feature[ADJUSTED_CLOSE_COLUMN] = get_feature(data_by_ticker, column=ADJUSTED_CLOSE_COLUMN,
                                                         debug=False, impute=False, verbose=verbose)
    data_by_feature[VOLUME_COLUMN] = get_feature(data_by_ticker, column=VOLUME_COLUMN,
                                                 debug=False, impute=False, verbose=verbose)
    data_by_feature, (start_date, end_date) = slice_backtesting_window(features=data_by_feature,
                                                                       start_date_requested=start_date_requested,
                                                                       end_date_requested=end_date_requested,
                                                                       lookback=max_lookback,
                                                                       verbose=verbose)

    prices = data_by_feature[ADJUSTED_CLOSE_COLUMN]

    def weekly_roration_runner(lookback=200,
                               sma_tol=0.02,
                               volume_lookback=20,
                               volume_threshold=1e6,
                               price_min=1.,
                               rsi_lookback=3,
                               rsi_threshold=50.,
                               day_of_trade=4,
                               n_positions=10):
        assert lookback <= max_lookback, f'lookback={lookback} should be smaller than max_lookback={max_lookback}.'

        time = timer()

        strategy = WeelkyRotation(start_date=start_date,
                                  end_date=end_date,
                                  lookback=lookback,

                                  sma_tol=sma_tol,
                                  volume_lookback=volume_lookback,
                                  volume_threshold=volume_threshold,
                                  price_min=price_min,
                                  rsi_lookback=rsi_lookback,
                                  rsi_threshold=rsi_threshold,
                                  day_of_trade=day_of_trade,
                                  n_positions=n_positions)
        strategy.fit(data_by_feature=data_by_feature)
        dates = strategy.get_dates()

        backtesting = Backtesting(start_balance=start_balance)
        backtesting.fit(strategy=strategy, prices=prices, dates=dates)

        results = make_backtesting_report(backtesting=backtesting, prices=prices, dates=dates,
                                          verbose=verbose, plotting=verbose)

        time = timer() - time
        if verbose:
            print(f'\nBacktesting time: {time:.3f} s.')

        if output_metric is None:
            output = results
        else:
            output = results[output_metric]

        return output

    return weekly_roration_runner


if __name__ == '__main__':
    runner = weekly_rotation_runner_facotry(start_date_requested=pd.to_datetime('2019-01-31'),  # '2000-11-19'
                                            end_date_requested=pd.to_datetime('2019-10-31'),  # '2019-11-22'
                                            max_lookback=200,
                                            verbose=True,
                                            output_metric='annualized_yield')
    annualized_yield = runner(lookback=200,
                              sma_tol=0.02,
                              volume_lookback=20,
                              volume_threshold=1e6,
                              price_min=1.,
                              rsi_lookback=3,
                              rsi_threshold=50.,
                              day_of_trade=4,
                              n_positions=10)

    REFERENCE_YIELD = 0.13514657590506485
    assert np.isclose(annualized_yield, REFERENCE_YIELD, atol=1e-8), \
        f'Yield is {annualized_yield}  but should be {REFERENCE_YIELD}'
