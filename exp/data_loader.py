import os
from timeit import default_timer as timer

import pandas as pd
from matplotlib import pyplot as plt

import ta.momentum

from exp import DATA_DIR, PLT_FILE_FORMAT, SP500_PKL
from exp.data_getter import load_pickled_dict

ADJUSTED_CLOSE_COLUMN = 'adjusted_close'
OPEN_COLUMN = 'open'
HIGH_COLUMN = 'high'
LOW_COLUMN = 'low'
CLOSE_COLUMN = 'close'
VOLUME_COLUMN = 'volume'
DIVIDENT_COLUMN = 'dividend_amount'
SPLIT_COLUMN = 'split_coefficient'


### TODO: Need the joiner and leaver data of the SP500 stocks and filtering functions
### TODO: Need to (re)-implement the RSI/EMA indicators.
### TODO: Forward pad: limit the number of padded values

def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)
    if start_idx is not None and end_idx is not None:
        feature_df = feature_df.iloc[start_idx:end_idx + 1, :]

    if debug:
        report_nans(feature_df, feature_name=column)
    if impute is not None:
        feature_df = impute_time_series(feature_df, method=impute)

    return feature_df


def report_nans(df, feature_name=ADJUSTED_CLOSE_COLUMN):
    n_nans = df.isna().sum(axis=0).sort_values(ascending=False)
    print(f'Number of Nans per ticker for {feature_name}:\n{n_nans}\n')

    df.isna().sum(axis=1).plot()
    plotfile = os.path.join(DATA_DIR, f'Nans_{feature_name}.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)
    print(f'Number of Nans per day for {feature_name} plotted in file: {plotfile}\n')


def impute_time_series(df, method=None):
    if method == 'pad':
        df = df.fillna(method=method)
    return df


# def backtesting(start_date, end_date, lookback, prices_dict):
#     return None


if __name__ == '__main__':
    time = timer()

    pkl_file = SP500_PKL  # PAR data

    start_date = pd.to_datetime('2019-01-31')  # PAR backtesting
    end_date = pd.to_datetime('2019-10-31')  # PAR backtesting
    lookback = 200  # PAR backtesting

    sma_tol = 0.02  # PAR strategy
    volume_lookback = 20  # PAR strategy
    volume_threshold = 1e6  # PAR strategy
    price_threshold = 1.  # PAR strategy
    rsi_lookback = 3  # PAR strategy
    rsi_threshold = 50.  # PAR strategy
    day_of_trade = 4  # PAR strategy - pandas DatetimeIndex.dayofweek value for Friday
    n_positions = 10  # PAR strategy

    data_by_ticker = load_pickled_dict(pkl_file=pkl_file)

    data_by_feature = {}
    adj_close_prices = get_feature(data_by_ticker, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False)
    start_idx = adj_close_prices.index.get_loc(start_date) - (lookback - 1)
    end_idx = adj_close_prices.index.get_loc(end_date)
    adj_close_prices = adj_close_prices.iloc[start_idx:end_idx + 1]

    volumes = get_feature(data_by_ticker, column=VOLUME_COLUMN, start_idx=start_idx, end_idx=end_idx, debug=False,
                          impute=False)

    # Filters for §8: Weekly rotation of the S&P 500 - The 30-Minute Stock Trader
    # 1 - SPY is above 0.98*SMA(200)
    spy = adj_close_prices['SPY']
    spy_sma = adj_close_prices['SPY'].rolling(lookback).mean()
    spy_masks = spy > spy_sma * (1 - sma_tol)

    # 2 - avg. vol. (20) >= 1M
    volumes_sma = volumes.rolling(volume_lookback).mean()
    volumes_masks = volumes_sma >= volume_threshold

    # 3 - Min price 1$
    price_masks = adj_close_prices > price_threshold

    # 4 - RSI(3) < 50
    prices_rsi = adj_close_prices.apply(lambda x: ta.momentum.rsi(x, n=rsi_lookback))
    rsi_masks = prices_rsi < rsi_threshold

    # Total masks 1-4
    # For pd.Series: .values[:, None] is to transform the series to a np.array and use its broadcasting capabilities,
    # specifying which axis needs to be repeated
    masks = volumes_masks & price_masks & rsi_masks & spy_masks.values[:, None]

    # 5a - Sort wrt the ROC(200)
    performances = adj_close_prices / adj_close_prices.shift(lookback - 1)

    # 5b - trade on Fridays in the backtesting window
    valid_dates = adj_close_prices.index
    day_of_week_triggers = valid_dates.dayofweek == day_of_trade
    backtesting_window_triggers = (valid_dates >= start_date) & (valid_dates <= end_date)
    triggers = day_of_week_triggers & backtesting_window_triggers
    dates = valid_dates[triggers]

    positions = pd.Series(index=dates, name='positions')
    for date in dates:
        mask = masks.loc[date]
        performance = performances.loc[date]
        positions[date] = performance[mask].sort_values(ascending=False)[:n_positions].index.tolist()

    buys = pd.Series(index=dates, name='buys')
    sells = pd.Series(index=dates, name='sells')
    buys.iloc[0] = set(positions.iloc[0])
    sells.iloc[0] = set()
    for i in range(1, dates.size):
        new_position = set(positions.iloc[i])
        old_position = set(positions.iloc[i-1])
        buys.iloc[i] = new_position.difference(old_position)
        sells.iloc[i] = old_position.difference(new_position)

    for i in range(dates.size):
        print(f'{dates[i].date()} - Positions: {positions.iloc[i]}, Buys: {buys.iloc[i]}, Sells: {sells.iloc[i]}')

    time = timer() - time
    print(f'Loop time: {time} s.')
