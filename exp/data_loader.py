import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pandas.plotting
from matplotlib import pyplot as plt

import ta.momentum
from sklearn.base import BaseEstimator

from exp import DATA_DIR, PLT_FILE_FORMAT, SP500_PKL
from exp.data_getter import load_pickled_dict
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN, VOLUME_COLUMN, POSITIONS_COLUMNS, TRADES_EXTRA_COLUMNS, \
    TRADES_COLUMNS, BENCKMARK_TICKER, N_DAYS_IN_YEAR, DEFAULT_START_BALANCE


### TODO: Need the joiner and leaver data of the SP500 stocks and filtering functions
### TODO: Need to (re)-implement the RSI/EMA indicators.
### TODO: Forward pad: limit the number of padded values
### TODO: The price for the execution is currently the price for the decision making; maybe the next day open instead?
### TODO: Ticker changes, merger, acquisitions, and bankrupcies: currently selling at the buy price, to prevent automl to pick up on those errors.


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
# 8 - Refactor the position getter into a strategy framework that
#   - takes start, end, lookback, strategy parameters, strategy data dict
#   - returns iterables of dates and positions on those dates.
# 9 - Refactor the backtesting loop into a class
# 10 - Data loader file, metrics file, reporting file, ...
# 11 - Use the open, high and low of the next day when executing trades in backtesting, to get more meaningful
#      estimate and confidence interval.
#    - (Requires) figure out the correction factor from AV and apply it to open, high, and low prices.

def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None,
                verbose=True):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)

    first_date = feature_df.index[0]
    last_date = feature_df.index[-1]
    n_tickers = len(feature_df.columns)
    if verbose:
        print(f'Loaded {column} for {n_tickers} tickers from {first_date.date()} to {last_date.date()}.')

    if start_idx is not None and end_idx is not None:
        feature_df = feature_df.iloc[start_idx:end_idx + 1, :]
    if debug:
        report_nans(feature_df, feature_name=column, verbose=verbose)
    if impute is not None:
        feature_df = impute_time_series(feature_df, method=impute)

    return feature_df


def report_nans(df, feature_name=ADJUSTED_CLOSE_COLUMN, verbose=True):
    n_nans = df.isna().sum(axis=0).sort_values(ascending=False)

    df.isna().sum(axis=1).plot()
    plotfile = os.path.join(DATA_DIR, f'Nans_{feature_name}.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)

    if verbose:
        print(f'Number of Nans per ticker for {feature_name}:\n{n_nans}\n')
        print(f'Number of Nans per day for {feature_name} plotted in file: {plotfile}\n')


def impute_time_series(df, method=None):
    if method == 'pad':
        df = df.fillna(method=method)
    return df


def get_ib_fees(position, price):
    IB_FEE_PER_SHARE = 0.005
    IB_MAX_FEE_PERCENT = 0.01
    IB_MIN_FEE = 1.

    fee_min = IB_MIN_FEE
    fee_max = position * price * IB_MAX_FEE_PERCENT
    fee = IB_FEE_PER_SHARE * position
    fee = min(max(fee, fee_min), fee_max)

    return fee


def get_notional(balance, n_buys, price_min):
    n_buys = max(1, n_buys)
    notional_max = balance / n_buys
    position_max = notional_max / price_min
    max_buy_fees = get_ib_fees(position=position_max, price=price_min)
    budget_fees = n_buys * max_buy_fees
    notional = (balance - budget_fees) / n_buys

    return notional


def get_p_and_l(row):
    fees = row['fees']
    position = row['position']
    price_buy = row['price_buy']
    price_sell = row['price_sell']
    p_and_l = position * (price_sell - price_buy) - fees
    return p_and_l


def get_balance(start_balance, trades_df, positions_df):
    p_and_l = trades_df['P&L'].sum()
    fees = positions_df['fees_buy'].sum()
    invested = (positions_df['position'] * positions_df['price_buy']).sum()
    balance = start_balance + p_and_l - invested - fees

    return balance


def get_total_unrealized_p_and_l(trades_df, positions_df, prices_series):
    realized_pl = trades_df['P&L'].sum()

    invested = (positions_df['position'] * positions_df['price_buy']).sum()
    prices_current = prices_series[positions_df['ticker']].values
    value_current = (positions_df['position'] * prices_current).sum()
    fees = positions_df['fees_buy'].sum()
    unrealized_pl = value_current - invested - fees

    p_and_l = realized_pl + unrealized_pl

    return p_and_l


def get_annualized_yield(unrealized_pl, start_balance=0.):
    dates = unrealized_pl.index
    backtesting_duration = (dates[-1] - dates[0]).days
    start_value = start_balance + unrealized_pl.iloc[0]
    end_value = start_balance + unrealized_pl.iloc[-1]
    annualized_yield = N_DAYS_IN_YEAR / backtesting_duration * np.log(end_value / start_value)

    return annualized_yield


def get_sharpe_ratio(unrealized_pl, start_balance=0.):
    normalized_returns = (unrealized_pl - unrealized_pl.shift(1)) / (unrealized_pl.shift(1) + start_balance)

    duration_in_years = (unrealized_pl.index[-1] - unrealized_pl.index[0]).days / N_DAYS_IN_YEAR
    mean_steps_per_year = unrealized_pl.size / duration_in_years

    norm_returns_mean_annualized = normalized_returns.mean() * mean_steps_per_year
    norm_returns_std_annualized = normalized_returns.std() * np.sqrt(
        mean_steps_per_year)  # Assuming a Weiner process / random walk; mean total distance = distance per step * sqrt(number of steps)
    sharpe_ratio = norm_returns_mean_annualized / norm_returns_std_annualized

    return sharpe_ratio


def get_sortino_ratio(unrealized_pl, start_balance=0.):
    normalized_returns = (unrealized_pl - unrealized_pl.shift(1)) / (unrealized_pl.shift(1) + start_balance)

    duration_in_years = (unrealized_pl.index[-1] - unrealized_pl.index[0]).days / N_DAYS_IN_YEAR
    mean_steps_per_year = unrealized_pl.size / duration_in_years

    norm_returns_mean_annualized = normalized_returns.mean() * mean_steps_per_year

    neg_returns_mask = normalized_returns < 0.
    neg_returns_std = normalized_returns[neg_returns_mask].std()
    norm_returns_std_annualized = neg_returns_std * np.sqrt(
        mean_steps_per_year)  # Assuming a Weiner process / random walk; mean total distance = distance per step * sqrt(number of steps)

    sharpe_ratio = norm_returns_mean_annualized / norm_returns_std_annualized

    return sharpe_ratio


def slice_backtesting_window(features={}, start_date_requested=None, end_date_requested=None, lookback=None,
                             verbose=True):
    assert all([isinstance(feature, pd.DataFrame) for feature in features.values()]), \
        f'Passed features must all be pd.DataFrames!'
    indexes = [df.index for df in features.values()]
    assert all([indexes[0].equals(index) for index in indexes[1:]]), \
        f'All features DataFrames must have the same indexing.'

    df = list(features.values())[0]
    valid_dates = df[start_date_requested:end_date_requested].index
    start_date = valid_dates[0]
    end_date = valid_dates[-1]
    if verbose:
        print(f'Backtesting date range adjusted from {start_date_requested.date()} - {end_date_requested.date()}'
              f' to {start_date.date()} - {end_date.date()}')

    start_idx = df.index.get_loc(start_date) - (lookback - 1)
    end_idx = df.index.get_loc(end_date)
    assert start_idx >= 0, \
        f'Lookback window could not be fulfilled, need {abs(start_idx)} additional data points before {start_date.date()}'

    features_sliced = {k: df.iloc[start_idx:end_idx + 1] for k, df in features.items()}

    return features_sliced, (start_date, end_date)


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


class Backtesting(object):

    def __init__(self, start_balance=DEFAULT_START_BALANCE):
        self.start_balance = start_balance

    def fit(self, strategy, prices, dates=None):
        if dates is None:
            dates = prices.index
        positions_df = pd.DataFrame(columns=POSITIONS_COLUMNS)
        errors_df = pd.DataFrame(columns=POSITIONS_COLUMNS)
        trades_df = pd.DataFrame(columns=TRADES_COLUMNS)
        unrealized_pl = pd.Series(index=dates, name=f'Unrealized P&L')
        pos_counter = 0
        balance = self.start_balance
        old_position = set()
        for i, date in enumerate(dates):
            new_position = set(strategy.predict(date))
            buys = new_position.difference(old_position)
            sells = old_position.difference(new_position)
            old_position = new_position

            for sell in sells:
                mask = positions_df['ticker'] == sell
                assert mask.sum() == 1, f'The number of current positions matching {sell} is {mask.sum()} but should be ==1'
                pos_id = positions_df[mask].index[0]

                price = prices.loc[date, sell]
                position = positions_df.loc[pos_id, 'position']
                if pd.isna(price):
                    price = positions_df.loc[pos_id, 'price_buy']
                    errors_df.loc[pos_id, POSITIONS_COLUMNS] = positions_df.loc[pos_id, POSITIONS_COLUMNS]
                    errors_df.loc[pos_id, 'date_sell'] = date
                fee = get_ib_fees(position=position, price=price)

                balance += position * price - fee

                trades_df.loc[pos_id, POSITIONS_COLUMNS] = positions_df.loc[pos_id, POSITIONS_COLUMNS]
                positions_df = positions_df.drop(index=pos_id)

                trades_df.loc[pos_id, TRADES_EXTRA_COLUMNS] = [date, price, fee]
                trades_df.loc[pos_id, 'fees'] = trades_df.loc[pos_id, 'fees_buy'] + trades_df.loc[pos_id, 'fees_sell']
                trades_df.loc[pos_id, 'P&L'] = get_p_and_l(row=trades_df.loc[pos_id])

            notional = get_notional(balance=balance, n_buys=len(buys), price_min=strategy.price_min)
            for buy in buys:
                price = prices.loc[date, buy]
                position = notional // price
                fee = get_ib_fees(position=position, price=price)

                balance += -position * price - fee

                positions_df.loc[pos_counter, POSITIONS_COLUMNS] = [date, buy, position, price, fee]
                pos_counter += 1

            unrealized_pl[date] = get_total_unrealized_p_and_l(trades_df=trades_df, positions_df=positions_df,
                                                               prices_series=prices.loc[date])

        balance_theoric = get_balance(start_balance=self.start_balance, trades_df=trades_df, positions_df=positions_df)
        assert np.isclose(balance, balance_theoric, atol=0.01), \
            f'Backtesting end balance does not check out against transactions: ' \
            f'end balance = {balance:.2f}$ VS theoric balance = {balance_theoric:.2f}$.'

        date = dates[-1]
        prices_current = prices.loc[date, positions_df['ticker']].values
        value_current = (positions_df['position'] * prices_current).sum()
        balance_pl = self.start_balance + unrealized_pl[date] - value_current
        assert np.isclose(balance, balance_pl, atol=0.01), \
            f'Backtesting end balance does not check out against balance deduced from (last P&L - investments): ' \
            f'end balance = {balance:.2f}$ VS P&L derived balance = {balance_pl:.2f}$.'

        self.trades_df = trades_df
        self.positions_df = positions_df
        self.errors_df = errors_df
        self.unrealized_pl = unrealized_pl

        return self

    def get_trades(self):
        return self.trades_df, self.positions_df, self.errors_df

    def get_unrealized_pl(self):
        return self.unrealized_pl


def make_backtesting_report(backtesting, prices, dates=None, verbose=True, plotting=True):
    assert isinstance(backtesting, Backtesting), \
        f'This function requires an instance of the Backtesting class to work.'
    trades_df, positions_df, errors_df = backtesting.get_trades()
    unrealized_pl = backtesting.get_unrealized_pl()
    start_balance = backtesting.start_balance

    if verbose > 1:
        print(f'\nTrades DataFrame:\n{trades_df}')
        print(f'\nPositions DataFrame:\n{positions_df}')
        print(f'\nErrors DataFrame:\n{errors_df}')
        print(f'\nUnrealized P&L Series:\n{unrealized_pl}')

    benckmark_prices = prices.loc[dates, BENCKMARK_TICKER]
    benckmark_position = start_balance / benckmark_prices.iloc[0]
    benckmark_fee = get_ib_fees(position=benckmark_position, price=benckmark_prices.iloc[0])
    benckmark_pl = benckmark_position * benckmark_prices - start_balance - benckmark_fee

    realized_pl = trades_df['P&L'].sum()
    annualized_yield = get_annualized_yield(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sharpe_ratio = get_sharpe_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sortino_ratio = get_sortino_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    benchmark_annualized_yield = get_annualized_yield(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sharpe_ratio = get_sharpe_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sortino_ratio = get_sortino_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)

    if verbose:
        print(f'\nRealized P&L: {realized_pl:.2f}$')
        print(f'Annualized yield = {100 * annualized_yield:.1f}%; benckmark = {100 * benchmark_annualized_yield:.1f}%')
        print(f'Sharpe ratio: {sharpe_ratio:.2f}; benckmark = {benchmark_sharpe_ratio:.2f}')
        print(f'Sortino ratio: {sortino_ratio:.2f}; benckmark = {benchmark_sortino_ratio:.2f}')

    if plotting:
        pd.plotting.register_matplotlib_converters()
        plt.figure(figsize=(10, 5))
        label = f'{unrealized_pl.name} - yield={100 * annualized_yield:.1f}%; sharpe={sharpe_ratio:.2f}; sortino={sortino_ratio:.2f}'
        plt.plot(unrealized_pl, '--r', label=label)
        benchmark_label = f'{benckmark_pl.name} - yield={100 * benchmark_annualized_yield:.1f}%; sharpe={benchmark_sharpe_ratio:.2f}; sortino={benchmark_sortino_ratio:.2f}'
        plt.plot(benckmark_pl, '--b', label=benchmark_label)
        xlim = plt.xlim()
        plt.xlim(xlim)
        plt.plot(xlim, [0.] * 2, '--k')
        plt.title(f'Strategy performance')
        plt.legend()
        plotfile = os.path.join(DATA_DIR, f'unrealized_pl.{PLT_FILE_FORMAT}')
        plt.savefig(plotfile)
        if verbose:
            print(f'\nPlotted unrealized P&L to file: {plotfile}')

    results = {'realized_pl': realized_pl,
               'annualized_yield': annualized_yield,
               'sharpe_ratio': sharpe_ratio,
               'sortino_ratio': sortino_ratio}

    return results


class WeeklyRotationRunner(BaseEstimator):

    def __init__(self,
                 start_date_requested=pd.to_datetime('2019-01-31'),
                 end_date_requested=pd.to_datetime('2019-10-31'),
                 lookback=200,
                 start_balance=DEFAULT_START_BALANCE,
                 verbose=True):
        self.start_date_requested = start_date_requested
        self.end_date_requested = end_date_requested
        self.lookback = lookback
        self.start_balance = start_balance
        self.verbose = verbose

    def fit(self, data_by_ticker):
        time = timer()

        data_by_feature = {}
        data_by_feature[ADJUSTED_CLOSE_COLUMN] = get_feature(data_by_ticker, column=ADJUSTED_CLOSE_COLUMN,
                                                             debug=False, impute=False, verbose=self.verbose)
        data_by_feature[VOLUME_COLUMN] = get_feature(data_by_ticker, column=VOLUME_COLUMN,
                                                     debug=False, impute=False, verbose=self.verbose)
        data_by_feature, (start_date, end_date) = slice_backtesting_window(features=data_by_feature,
                                                                           start_date_requested=self.start_date_requested,
                                                                           end_date_requested=self.end_date_requested,
                                                                           lookback=self.lookback,
                                                                           verbose=self.verbose)

        prices = data_by_feature[ADJUSTED_CLOSE_COLUMN]

        strategy = WeelkyRotation(start_date=start_date, end_date=end_date, lookback=self.lookback)
        strategy.fit(data_by_feature=data_by_feature)
        dates = strategy.get_dates()

        backtesting = Backtesting(start_balance=self.start_balance)
        backtesting.fit(strategy=strategy, prices=prices, dates=dates)

        self.results = make_backtesting_report(backtesting=backtesting, prices=prices, dates=dates,
                                               verbose=self.verbose, plotting=self.verbose)

        time = timer() - time
        if self.verbose:
            print(f'\nBacktesting time: {time:.3f} s.')

        return self

    def score(self, x_test=None, y_test=None):
        return self.results["annualized_yield"]


if __name__ == '__main__':
    data_by_ticker = load_pickled_dict(pkl_file=SP500_PKL)

    runner = WeeklyRotationRunner(start_date_requested=pd.to_datetime('2019-01-31'),  # '2000-11-19'
                                  end_date_requested=pd.to_datetime('2019-10-31'),  # '2019-11-22'
                                  lookback=200,
                                  verbose=True)
    runner.fit(data_by_ticker=data_by_ticker)

    REFERENCE_YIELD = 0.13514657590506485
    annualized_yield = runner.score()
    assert np.isclose(annualized_yield, REFERENCE_YIELD, atol=1e-8), \
        f'Yield is {annualized_yield}  but should be {REFERENCE_YIELD}'
